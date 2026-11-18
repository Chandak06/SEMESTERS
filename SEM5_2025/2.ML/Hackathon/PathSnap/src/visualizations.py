import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from .config import ROOT
def model_plots():
    table=pd.read_csv(ROOT/"results/models/comparison.csv")
    for metric in ["cv_macro_f1","top1_accuracy","macro_f1","log_loss"]:
        fig,ax=plt.subplots(figsize=(9,5)); ax.bar(table.model,table[metric],color="#217a91")
        ax.set_ylabel(metric.replace("_"," ")); ax.set_title("Trip-grouped CV" if metric.startswith("cv") else "Student testing (aligned reference proxy)")
        ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(ROOT/"results/models"/f"{metric}.png",dpi=150); plt.close(fig)
    for name in table.model:
        result=json.loads((ROOT/"results/models"/f"{name}.json").read_text())
        cm=np.asarray(result["student_test"]["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="red")
        ax.set(xlabel="Predicted",ylabel="Reference proxy",xticks=[0,1],yticks=[0,1],title=name)
        fig.tight_layout(); fig.savefig(ROOT/"results/models"/f"{name}_confusion.png",dpi=150); plt.close(fig)
COLOURS=dict(classifier="#f39c12",gps="#2479d4",prediction="#e34b36",network="#6b6b6b",reference="#1c9e54",start="#0a7d32",end="#111111")
def zoom_for(lower,upper,width_px=900,height_px=560):
    """Web-Mercator zoom that fits a lat/lon box in a map of the given pixel size.
    Used instead of relying only on fitBounds, which can run before an embedded
    iframe has its final size and then zooms out too far."""
    lat=np.deg2rad(np.clip([lower[0],upper[0]],-85,85))
    mercator=np.log(np.tan(np.pi/4+lat/2))
    dlat=max(abs(mercator[1]-mercator[0])/(2*np.pi)*360,1e-6); dlon=max(upper[1]-lower[1],1e-6)
    zoom=min(np.log2(width_px*360/(256*dlon)),np.log2(height_px*360/(256*dlat)))
    return int(np.clip(np.floor(zoom-.15),3,18))
def route_map(gps,segments,network=None,reference=None,fit_bounds=True,width_px=900,height_px=560,classifier=None):
    """Folium map of GPS observations, road geometry and the predicted route.
    gps: list of [lat, lon] or None; segments/network/reference: lists of [[lat, lon], ...] polylines."""
    clean=np.asarray([p for p in gps if p is not None],dtype=float).reshape(-1,2)
    if len(clean):
        lower=clean.min(axis=0); upper=clean.max(axis=0)
        m=folium.Map(location=((lower+upper)/2).tolist(),zoom_start=zoom_for(lower,upper,width_px,height_px),tiles="OpenStreetMap",prefer_canvas=True)
    else:
        m=folium.Map(location=[0,0],zoom_start=2,tiles="OpenStreetMap",prefer_canvas=True)
    if network is not None:
        g=folium.FeatureGroup(name="Road network (candidate corridor)",show=True)
        for s in network: folium.PolyLine(s,color=COLOURS["network"],weight=1.5,opacity=.45).add_to(g)
        g.add_to(m)
    g=folium.FeatureGroup(name="GPS observations")
    for p in clean: folium.CircleMarker(p.tolist(),radius=2,color=COLOURS["gps"],fill=True,weight=1).add_to(g)
    g.add_to(m)
    if reference is not None:
        g=folium.FeatureGroup(name="Reference route (offline only)")
        for s in reference: folium.PolyLine(s,color=COLOURS["reference"],weight=7,opacity=.55).add_to(g)
        g.add_to(m)
    if classifier is not None:
        g=folium.FeatureGroup(name="Classifier-only route (before HMM)",show=False)
        for s in classifier: folium.PolyLine(s,color=COLOURS["classifier"],weight=3,opacity=.9,dash_array="6 6").add_to(g)
        g.add_to(m)
    g=folium.FeatureGroup(name="Predicted route")
    for s in segments: folium.PolyLine(s,color=COLOURS["prediction"],weight=4,opacity=.9).add_to(g)
    g.add_to(m)
    if len(clean):
        g=folium.FeatureGroup(name="Trip start / end")
        folium.CircleMarker(clean[0].tolist(),radius=7,color="white",weight=2,fill=True,fill_color=COLOURS["start"],fill_opacity=1,tooltip="First GPS observation").add_to(g)
        folium.CircleMarker(clean[-1].tolist(),radius=7,color="white",weight=2,fill=True,fill_color=COLOURS["end"],fill_opacity=1,tooltip="Last GPS observation").add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    rows=[("gps","GPS observations","dot"),("prediction","Predicted route","line")]
    if network is not None: rows.insert(1,("network","Road network","line"))
    if reference is not None: rows.append(("reference","Reference route (offline only)","line"))
    if classifier is not None: rows.append(("classifier","Classifier-only route (toggle)","line"))
    rows+= [("start","Start","dot"),("end","End","dot")]
    swatch=lambda colour,kind:(f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{colour};margin-right:6px"></span>' if kind=="dot"
        else f'<span style="display:inline-block;width:18px;height:4px;background:{colour};margin:0 6px 3px 0;vertical-align:middle"></span>')
    legend="".join(f'<div>{swatch(COLOURS[key],kind)}{label}</div>' for key,label,kind in rows)
    m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.92);padding:8px 10px;border-radius:4px;font:12px/1.5 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)"><b>PathSnap legend</b>'+legend+'</div>'))
    if fit_bounds and len(clean): m.fit_bounds([lower.tolist(),upper.tolist()])
    return m
def geometry(trip,route):
    return [trip.nodes[trip.arcs[int(a)]][:,::-1].tolist() for a in route if a>=0]
def eda_plots():
    table=pd.read_csv(ROOT/"results/eda/trips.csv")
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for part,group in table.groupby("partition"):
        axes[0].hist(group.observations,bins=15,alpha=.6,label=part)
        axes[1].scatter(group.longitude_min,group.latitude_min,s=12,label=part)
        axes[2].scatter(group.nodes,group.weak_components,s=12,label=part)
    for ax,title in zip(axes,["Trip observation counts","GPS extent origins (lon/lat)","Network nodes vs weak components"]): ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT/"results/eda/exploration.png",dpi=150); plt.close(fig)
if __name__=="__main__": model_plots(); eda_plots()

