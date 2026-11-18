"""Generate offline Folium maps for every student-testing trip.

Runs the deployed pipeline (best_model + HMM/Viterbi) on each trip, checks the
prediction equals the saved results/hmm/<trip>.json, and writes
results/maps/student_test/<trip>.html with layers for GPS, the candidate road
corridor, the classifier-only route, the HMM route and, because these maps are
offline development evidence (never sent to the evaluator UI), the reference route.

    .\.venv\Scripts\python.exe scripts\generate_maps.py [--trips 00000000 00000052]
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import ROOT,save_json
from src.data_loader import list_trips,load_trip,load_reference
from src.inference import Predictor
from src.evaluate_models import compress
from src.visualizations import route_map,geometry
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--trips",nargs="*"); args=parser.parse_args()
    out=ROOT/"results/maps/student_test"; out.mkdir(parents=True,exist_ok=True)
    predictor=Predictor(); rows=[]
    for trip_id in args.trips or list_trips("student_test"):
        started=time.perf_counter()
        trip=load_trip("student_test",trip_id); result=predictor.predict(trip)
        saved=json.loads((ROOT/"results/hmm"/f"{trip_id}.json").read_text())
        matches=result["pred"].tolist()==saved["predictions"] and result["route"]==saved["route"]
        corridor=np.unique(result["ids"][result["ids"]>=0])
        m=route_map(trip.track[:,1::-1].tolist() if trip.valid.all() else [trip.track[i,1::-1].tolist() if trip.valid[i] else None for i in range(len(trip.track))],
            geometry(trip,result["route"]),network=geometry(trip,corridor),reference=geometry(trip,load_reference(trip)),
            classifier=geometry(trip,compress(result["classifier_pred"])))
        path=out/f"{trip_id}.html"; m.save(str(path))
        rows.append(dict(trip_id=trip_id,file=path.relative_to(ROOT).as_posix(),observations=len(trip.track),route_arcs=len(result["route"]),
            corridor_segments=len(corridor),matches_saved_hmm_predictions=matches,seconds=time.perf_counter()-started))
        print(json.dumps(rows[-1]),flush=True)
    save_json(ROOT/"results/maps/student_test/index.json",rows)
    if not all(r["matches_saved_hmm_predictions"] for r in rows): raise SystemExit("Live predictions differ from saved HMM results")
if __name__=="__main__": main()
