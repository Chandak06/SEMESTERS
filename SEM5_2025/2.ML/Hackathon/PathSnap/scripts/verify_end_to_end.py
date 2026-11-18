from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ui.client import ApiClient
from src.config import ROOT,save_json
from src.data_loader import load_trip,load_reference
from src.visualizations import route_map,geometry
out=ROOT/"results/completion_audit"
client=ApiClient("http://127.0.0.1:8011"); assert client.health()["model_loaded"]
trips=client.trips(); assert trips["mode"]=="student_test" and len(trips["trips"])==10
rows=[]
for trip_id in trips["trips"]:
    start=time.perf_counter(); body=client.predict(trip_id)
    expected=json.loads((ROOT/"results/hmm"/f"{trip_id}.json").read_text())
    assert body["predicted_arc_ids"]==expected["predictions"]
    assert body["predicted_route_arc_ids"]==expected["route"]
    assert body["network_geometry"] and body["predicted_geometry"]
    assert not any("reference" in k or "label" in k for k in body)
    rows.append(dict(trip_id=trip_id,top1_accuracy=body["metrics"]["top1_accuracy"],route_continuity=body["metrics"]["route_continuity"],http_seconds=time.perf_counter()-start,processing_seconds=body["processing_seconds"],network_segments=body["network_segment_count"],matches_saved_predictions=True))
    if trip_id==trips["trips"][0]:
        trip=load_trip("student_test",trip_id)
        route_map(body["gps"],body["predicted_geometry"],body["network_geometry"],geometry(trip,load_reference(trip))).save(str(out/"representative_offline.html"))
        route_map(body["gps"],body["predicted_geometry"],body["network_geometry"]).save(str(out/"representative_evaluator.html"))
    print(json.dumps(rows[-1]),flush=True)
save_json(out/"http_verification.json",dict(trips=rows,all_ten_match_saved=True,cross_laptop_verified=False))

