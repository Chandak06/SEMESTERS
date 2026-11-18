import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src.config import ROOT,save_json
app=AppTest.from_file(str(ROOT/"ui/app.py"),default_timeout=180).run()
assert not app.exception,app.exception
assert len(app.selectbox)==1 and len(app.selectbox[0].options)==10
next(b for b in app.button if b.label=="Predict Route").click().run(timeout=180)
assert not app.exception,app.exception
assert len(app.metric)==4
assert app.session_state["result"]["predicted_geometry"]
assert app.session_state["result"]["network_geometry"]
assert app.sidebar.success
metrics=[{"label":m.label,"value":m.value} for m in app.metric]
save_json(Path(os.getenv("PATHSNAP_UI_EVIDENCE",str(ROOT/"results/logs/ui_smoke.json"))),{"ui":"Streamlit AppTest","real_http":os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000"),"trip":app.session_state["result"]["trip_id"],"metrics":metrics,"reference_keys_present":any("reference" in k or "label" in k for k in app.session_state["result"])})
print(metrics)

