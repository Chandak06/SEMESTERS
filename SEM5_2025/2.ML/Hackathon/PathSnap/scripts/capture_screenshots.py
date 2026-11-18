"""Capture genuine screenshots of the running system with headless Microsoft Edge
(Chrome DevTools Protocol). Requires the backend and the Streamlit UI to be running.

    .\.venv\Scripts\python.exe scripts\capture_screenshots.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --trip 00000000

Writes results/report/screenshots/*.png and capture_log.json. Nothing is drawn
or edited: each PNG is the browser's own rendering of the live page.
"""
import argparse,base64,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import websocket
from src.config import ROOT,save_json
EDGE=[r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
class Browser:
    def __init__(self,port=9333):
        exe=next(p for p in EDGE if os.path.exists(p))
        self.proc=subprocess.Popen([exe,"--headless=new",f"--remote-debugging-port={port}",f"--user-data-dir={ROOT/'.cache/screenshot_browser'}",
            "--no-first-run","--disable-extensions","--hide-scrollbars","about:blank"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                pages=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
            except OSError: time.sleep(.5)
        page=next(p for p in pages if p["type"]=="page")
        self.ws=websocket.create_connection(page["webSocketDebuggerUrl"],timeout=120); self.n=0
        self.call("Page.enable")
    def call(self,method,**params):
        self.n+=1; self.ws.send(json.dumps({"id":self.n,"method":method,"params":params}))
        while True:
            msg=json.loads(self.ws.recv())
            if msg.get("id")==self.n:
                if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result",{})
    def js(self,expression):
        r=self.call("Runtime.evaluate",expression=expression,returnByValue=True,awaitPromise=True)
        return r.get("result",{}).get("value")
    def wait(self,expression,timeout=180,what=""):
        end=time.time()+timeout
        while time.time()<end:
            if self.js(expression): return
            time.sleep(.5)
        raise TimeoutError(f"Timed out waiting for {what or expression}")
    def open(self,url,width=1440,height=1000):
        self.call("Emulation.setDeviceMetricsOverride",width=width,height=height,deviceScaleFactor=1,mobile=False)
        self.call("Page.navigate",url=url)
        self.wait("document.readyState==='complete'",60,"page load")
    def shot(self,path,full=True):
        if full:
            h=self.js("Math.max(document.body.scrollHeight,document.documentElement.scrollHeight,...[...document.querySelectorAll('[data-testid=stMain],[data-testid=stAppViewContainer],section.main')].map(e=>e.scrollHeight))") or 1000
            w=self.js("document.documentElement.clientWidth") or 1440
            self.call("Emulation.setDeviceMetricsOverride",width=w,height=min(int(h),6000),deviceScaleFactor=1,mobile=False); time.sleep(1.5)
        data=self.call("Page.captureScreenshot",format="png")["data"]
        Path(path).write_bytes(base64.b64decode(data)); print("saved",path,flush=True)
    def close(self):
        try: self.ws.close()
        finally: self.proc.terminate()
TILES_DONE="""(()=>{const d=document; const t=[...d.querySelectorAll('.leaflet-tile')];
 return t.length>0 && t.every(i=>i.complete && i.naturalWidth>0)})()"""
def main():
    a=argparse.ArgumentParser(); a.add_argument("--api",default="http://127.0.0.1:8000"); a.add_argument("--ui",default="http://127.0.0.1:8501")
    a.add_argument("--trip",default="",help="optional; typed into the selectbox (best effort)"); args=a.parse_args()
    out=ROOT/"results/report/screenshots"; out.mkdir(parents=True,exist_ok=True); log={"api":args.api,"ui":args.ui,"trip_requested":args.trip or "first in list","shots":[]}
    b=Browser()
    try:
        b.open(args.api+"/health",900,300); b.shot(out/"api_health.png",False); log["shots"].append("api_health.png")
        b.open(args.api+"/trips",1200,300); b.shot(out/"api_trips.png",False); log["shots"].append("api_trips.png")
        b.open(args.api+"/docs",1440,900); b.wait("!!document.querySelector('.opblock')",60,"Swagger UI"); time.sleep(1); b.shot(out/"api_docs.png"); log["shots"].append("api_docs.png")
        b.open(args.ui,1440,1000)
        b.wait("[...document.querySelectorAll('button')].some(x=>x.innerText.trim()==='Predict Route')",120,"Predict button")
        if args.trip:
            # choose the trip in the Streamlit selectbox by typing into its combobox input
            b.js("document.querySelector('[data-testid=\"stSelectbox\"] input').focus()")
            b.call("Input.insertText",text=args.trip); time.sleep(.8)
            b.call("Input.dispatchKeyEvent",type="keyDown",key="Enter",code="Enter",windowsVirtualKeyCode=13)
            b.call("Input.dispatchKeyEvent",type="keyUp",key="Enter",code="Enter",windowsVirtualKeyCode=13); time.sleep(1.5)
        b.shot(out/"ui_before_prediction.png",False); log["shots"].append("ui_before_prediction.png")
        b.js("[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Predict Route').click()")
        b.wait("document.body.innerText.includes('Observation Top-1 accuracy')",300,"prediction result")
        b.wait("document.querySelectorAll('iframe').length>0",60,"map iframe")
        MAPDOC="[...document.querySelectorAll('iframe')].map(f=>{try{return f.contentDocument}catch(e){return null}}).find(d=>d&&d.querySelector('.leaflet-container'))"
        try:
            b.wait(f"!!({MAPDOC})",30,"map")
            b.wait(f"(()=>{{const d={MAPDOC};const t=[...d.querySelectorAll('.leaflet-tile')];return t.length>0&&t.every(i=>i.complete&&i.naturalWidth>0)}})()",60,"map tiles")
        except TimeoutError as exc:
            log["map_wait_warning"]=f"{exc}; iframe not inspectable or tiles slow, used fixed 12 s wait"; time.sleep(12)
        log["ui_text"]=b.js("document.querySelector('[data-testid=\"stMain\"]')?.innerText.slice(0,800) || document.body.innerText.slice(0,800)")
        import re
        shown=re.search(r"Predicted route for trip (\d{8})",log["ui_text"] or ""); log["trip_shown_in_ui"]=shown.group(1) if shown else None
        b.shot(out/"ui_prediction.png"); log["shots"].append("ui_prediction.png")
        for trip in ["00000000","00000052"]:
            html=ROOT/"results/maps/student_test"/f"{trip}.html"
            b.open(html.as_uri(),1400,900); b.wait("!!document.querySelector('.leaflet-container')",60,"folium map")
            try: b.wait(TILES_DONE,60,"tiles")
            except TimeoutError: log.setdefault("map_tile_warnings",[]).append(trip)
            b.shot(out/f"folium_{trip}.png",False); log["shots"].append(f"folium_{trip}.png")
    finally:
        b.close()
    log["captured_at"]=time.strftime("%Y-%m-%d %H:%M:%S"); log["cross_laptop"]=False
    save_json(out/"capture_log.json",log); print(json.dumps({k:v for k,v in log.items() if k!="ui_text"}))
if __name__=="__main__": main()
