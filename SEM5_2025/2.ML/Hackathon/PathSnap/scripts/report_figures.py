"""Figures and tables for the technical report, computed only from saved
experiment outputs, the raw training/student-testing files and the saved model
artifacts. No model is fitted, tuned or overwritten.

    .\.venv\Scripts\python.exe scripts\report_figures.py

Outputs: results/report/*.png, results/report/*.csv and
results/models/fit_diagnostics.csv (in-sample vs CV vs student-test metrics).
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from src.config import ROOT,FEATURES,save_json
from src.data_loader import load_trip,load_reference,list_trips
from src.train_models import SPECS,training_data
from src.evaluate_models import candidate_metrics,score,top1
OUT=ROOT/"results/report"; OUT.mkdir(parents=True,exist_ok=True)
NAMES={"logistic":"Logistic Regression","naive_bayes":"Gaussian Naive Bayes","decision_tree":"Decision Tree",
       "random_forest":"Random Forest","hist_gradient_boosting":"HistGradientBoosting","tuned_model":"HGB (tuned)"}
def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/name,dpi=150); plt.close(fig); print("saved",name,flush=True)
def data_quality():
    trips=pd.read_csv(ROOT/"results/eda/trips.csv")
    cols=["observations","invalid_gps","missing_values","duplicate_timestamps","reversed_timestamps","gaps_over_30s","alignment_over_80m","reference_breaks","zero_length_arcs","isolated_nodes"]
    table=trips.groupby("partition")[cols].sum().T
    table["evaluator_test"]="not supplied"
    table.to_csv(OUT/"data_quality.csv"); print(table)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for part,colour in [("train","#217a91"),("student_test","#e3893b")]:
        g=trips[trips.partition==part]
        axes[0].hist(np.log10(g.observations),bins=15,alpha=.65,color=colour,label=f"{part} ({len(g)} trips)")
        axes[1].hist(np.log10(g.max_interval_s.clip(lower=1)),bins=15,alpha=.65,color=colour,label=part)
        axes[2].hist(g.alignment_p95_m.clip(upper=100),bins=15,alpha=.65,color=colour,label=part)
    axes[0].set(title="GPS observations per trip",xlabel="log10(observations)",ylabel="trips")
    axes[1].set(title="Largest sampling gap per trip (median gap = 1 s)",xlabel="log10(max interval, s)")
    axes[2].set(title="95th pct GPS-to-reference distance",xlabel="metres (clipped at 100)")
    for ax in axes: ax.legend(fontsize=8)
    save(fig,"eda_distributions.png")
def trajectory_plot(partition,trip_id,ax,show_reference):
    trip=load_trip(partition,trip_id); gps=trip.track[trip.valid]
    pad=.004; lo=gps[:,:2].min(axis=0)-pad; hi=gps[:,:2].max(axis=0)+pad
    a=trip.nodes[trip.arcs[:,0]]; b=trip.nodes[trip.arcs[:,1]]
    inside=((np.minimum(a,b)<=hi)&(np.maximum(a,b)>=lo)).all(axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([a[inside],b[inside]],axis=1),colors="#9a9a9a",linewidths=.5,label="road network arcs"))
    if show_reference:
        r=load_reference(trip); ax.add_collection(LineCollection(np.stack([trip.nodes[trip.arcs[r,0]],trip.nodes[trip.arcs[r,1]]],axis=1),colors="#1c9e54",linewidths=2.5,label="reference route"))
    ax.scatter(gps[:,0],gps[:,1],s=1.5,c=gps[:,2]-gps[0,2],cmap="viridis",label="GPS (colour = time)",zorder=3)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel="longitude",ylabel="latitude",
        title=f"{partition} trip {trip_id}: {len(gps)} GPS points, {len(trip.arcs)} arcs ({inside.sum()} in view)")
    ax.set_aspect(1/np.cos(np.deg2rad(gps[:,1].mean()))); ax.legend(fontsize=7,loc="best")
def trajectories():
    fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    trajectory_plot("train","00000038",axes[0],True)
    trajectory_plot("student_test","00000052",axes[1],False)
    save(fig,"gps_and_road_network.png")
def feature_distributions():
    data=training_data(400); x=data[0].reshape(-1,len(FEATURES)); y=data[1].ravel()
    keys=["distance_m","heading_difference","distance_rank","distance_margin_m","projection_fraction","speed_mps"]
    fig,axes=plt.subplots(2,3,figsize=(13,6.5))
    for ax,key in zip(axes.ravel(),keys):
        v=x[:,FEATURES.index(key)]; hi=np.percentile(v,99)
        bins=np.linspace(v.min(),hi,40) if key!="distance_rank" else np.arange(-.5,8.5)
        ax.hist(v[y==0],bins=bins,alpha=.6,density=True,label="other candidates",color="#999999")
        ax.hist(v[y==1],bins=bins,alpha=.6,density=True,label="aligned (positive) candidate",color="#e34b36")
        ax.set_title(key); ax.legend(fontsize=7)
    fig.suptitle(f"Training sample: {len(y):,} candidate rows from 80 training trips ({int(y.sum()):,} positive)")
    save(fig,"feature_distributions.png")
    return data
def fit_diagnostics(data):
    """In-sample (training-sample) metrics of the saved artifacts vs. their
    trip-grouped CV and student-test results. Evidence for over/under-fitting."""
    x,y,groups,ids,labels,_=data; rows=[]
    comparison=pd.read_csv(ROOT/"results/models/comparison.csv").set_index("model")
    tuning=json.loads((ROOT/"results/tuning/summary.json").read_text())
    for name in list(SPECS)+["tuned_model"]:
        model=joblib.load(ROOT/"artifacts"/f"{name}.joblib")
        with threadpool_limits(limits=2): p,_=score(model,x,ids)
        train=candidate_metrics(y,p)
        if name in SPECS:
            cv=json.loads((ROOT/"results/models"/f"{name}.json").read_text())["cv"]; test=comparison.loc[name]
            test_top1,test_f1,test_ll=test.top1_accuracy,test.macro_f1,test.log_loss
        else:
            cv=tuning["cv"]; st=tuning["student_test"]; test_top1,test_f1,test_ll=st["top1_accuracy"],st["macro_f1"],st["log_loss"]
        rows.append(dict(model=name,train_top1=float(np.mean(top1(ids,p)==labels)),train_macro_f1=train["macro_f1"],train_log_loss=train["log_loss"],
            cv_top1=float(np.mean([f["top1_accuracy"] for f in cv])),cv_macro_f1=float(np.mean([f["macro_f1"] for f in cv])),cv_log_loss=float(np.mean([f["log_loss"] for f in cv])),
            test_top1=float(test_top1),test_macro_f1=float(test_f1),test_log_loss=float(test_ll)))
    frame=pd.DataFrame(rows); frame.to_csv(ROOT/"results/models/fit_diagnostics.csv",index=False); print(frame.round(4).to_string())
    fig,axes=plt.subplots(1,2,figsize=(13,4.5)); pos=np.arange(len(frame)); w=.27
    for ax,metric in zip(axes,["top1","log_loss"]):
        for i,(split,colour) in enumerate([("train","#7f7f7f"),("cv","#217a91"),("test","#e3893b")]):
            ax.bar(pos+(i-1)*w,frame[f"{split}_{metric}"],w,label={"train":"training sample (in-sample)","cv":"3-fold trip-grouped CV","test":"student test"}[split],color=colour)
        ax.set_xticks(pos,[NAMES[m] for m in frame.model],rotation=20,fontsize=8)
        ax.set_title("Observation Top-1 accuracy" if metric=="top1" else "Candidate log loss (lower is better)")
        if metric=="top1": ax.set_ylim(.75,1)
        ax.legend(fontsize=7)
    save(fig,"fit_diagnostics.png")
def model_selection():
    t=pd.read_csv(ROOT/"results/models/comparison.csv")
    fig,axes=plt.subplots(1,3,figsize=(14,4.2)); pos=np.arange(len(t)); labels=[NAMES[m] for m in t.model]
    axes[0].bar(pos-.2,t.cv_macro_f1,.4,yerr=t.cv_macro_f1_std,label="CV Macro F1 (± fold std)",color="#217a91"); axes[0].bar(pos+.2,t.macro_f1,.4,label="student-test Macro F1",color="#e3893b")
    axes[0].set_ylim(.75,.95); axes[0].set_title("Candidate Macro F1")
    axes[1].bar(pos-.2,t.cv_top1_accuracy,.4,label="CV Top-1",color="#217a91"); axes[1].bar(pos+.2,t.top1_accuracy,.4,label="student-test Top-1",color="#e3893b")
    axes[1].axhline(t.cv_top1_accuracy.max()-.005,ls="--",c="k",lw=1,label="selection threshold (CV max − 0.005)"); axes[1].set_ylim(.78,.88); axes[1].set_title("Observation Top-1 accuracy")
    axes[2].bar(pos,t.inference_seconds,color="#6b6b6b"); axes[2].set_yscale("log"); axes[2].set_title("Classifier inference time, 10 student trips (s, log)")
    for ax in axes: ax.set_xticks(pos,labels,rotation=25,fontsize=8)
    for ax in axes[:2]: ax.legend(fontsize=7)
    save(fig,"model_selection.png")
def feature_importance():
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,name in zip(axes,["decision_tree","random_forest"]):
        model=joblib.load(ROOT/"artifacts"/f"{name}.joblib"); imp=model.feature_importances_; order=np.argsort(imp)
        ax.barh(np.array(FEATURES)[order],imp[order],color="#217a91"); ax.set_title(f"{NAMES[name]}: impurity-based feature importance")
    save(fig,"feature_importance.png")
def tuning():
    history=json.loads((ROOT/"results/tuning/trial_history.json").read_text())
    summary=json.loads((ROOT/"results/tuning/summary.json").read_text())
    values=np.array([t["value"] for t in history])
    fig,axes=plt.subplots(1,2,figsize=(13,4.2))
    colours=["#9a9a9a" if t["number"]<summary["tpe_startup_trials"] else "#217a91" for t in history]
    axes[0].scatter([t["number"] for t in history],values,c=colours,zorder=3); axes[0].plot(np.maximum.accumulate(values),c="#e34b36",label="best so far")
    axes[0].axhline(summary["baseline_objective"],ls="--",c="k",lw=1,label="untuned baseline objective")
    axes[0].set(xlabel="trial (grey = random startup, blue = TPE)",ylabel="CV Top-1 − 0.02·log loss",title="Optuna TPE trials (HistGradientBoosting)"); axes[0].legend(fontsize=8)
    comp=pd.read_csv(ROOT/"results/tuning/comparison.csv").set_index("variant")
    metrics=["cv_macro_f1","top1_accuracy","macro_f1"]; pos=np.arange(len(metrics))
    for i,v in enumerate(["baseline","tuned"]): axes[1].bar(pos+(i-.5)*.38,comp.loc[v,metrics],.38,label=v,color=["#9a9a9a","#217a91"][i])
    axes[1].set_xticks(pos,["CV Macro F1","Student Top-1","Student Macro F1"]); axes[1].set_ylim(.84,.92); axes[1].legend(); axes[1].set_title("Untuned vs tuned (same partitions)")
    save(fig,"tuning.png")
def hmm():
    t=pd.read_csv(ROOT/"results/hmm/per_trip.csv")
    p=t.pivot(index="trip_id",columns="method",values=["top1_accuracy","route_continuity","sequence_edit_similarity"])
    fig,axes=plt.subplots(1,3,figsize=(15,4.2)); pos=np.arange(len(p))
    for ax,key in zip(axes,["top1_accuracy","route_continuity","sequence_edit_similarity"]):
        ax.bar(pos-.2,p[key]["classifier"],.4,label="classifier only",color="#f39c12"); ax.bar(pos+.2,p[key]["hmm"],.4,label="classifier + HMM/Viterbi",color="#e34b36")
        ax.set_xticks(pos,[str(i) for i in p.index],rotation=45,fontsize=7); ax.set_title(key.replace("_"," ")+" per student trip"); ax.set_ylim(0 if key=="sequence_edit_similarity" else .6,1)
    axes[0].legend(fontsize=8); save(fig,"hmm_per_trip.png")
    p.columns=[f"{a}_{b}" for a,b in p.columns]; p.to_csv(OUT/"hmm_per_trip_wide.csv")
if __name__=="__main__":
    started=time.perf_counter()
    data_quality(); trajectories(); data=feature_distributions(); fit_diagnostics(data)
    model_selection(); feature_importance(); tuning(); hmm()
    print(f"done in {time.perf_counter()-started:.1f}s")
