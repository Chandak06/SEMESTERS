import json
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from src.config import ROOT
from src.inference import Predictor,ModelArtifactError
from src.data_loader import load_trip
from src.visualizations import route_map

def test_missing_artifact_is_actionable(tmp_path):
    with pytest.raises(ModelArtifactError,match="Missing inference artifact.*Restore"):
        Predictor(model_path=tmp_path/"missing.joblib")

def test_feature_order_mismatch_rejected(tmp_path):
    cfg=json.loads((ROOT/"configs/deployment.json").read_text())
    cfg["features"]=cfg["features"][::-1]
    path=tmp_path/"bad_config.json"; path.write_text(json.dumps(cfg))
    with pytest.raises(ModelArtifactError,match="feature names/order"):
        Predictor(config_path=path)

def test_corrupt_artifact_rejected(tmp_path):
    path=tmp_path/"corrupt.joblib"; path.write_bytes(b"not a model")
    with pytest.raises(ModelArtifactError,match="Cannot load compatible"):
        Predictor(model_path=path)

def test_degraded_health_and_prediction_error(monkeypatch):
    import app.api as api
    def missing(): raise ModelArtifactError("Missing inference artifact: best_model.joblib")
    monkeypatch.setattr(api,"Predictor",missing)
    with TestClient(api.app) as client:
        health=client.get("/health")
        assert health.status_code==503
        assert not health.json()["model_loaded"] and "Missing" in health.json()["error"]
        assert client.post("/predict",json={"trip_id":"00000052"}).status_code==503

def test_candidate_road_layer_without_references(monkeypatch):
    import app.api as api
    def missing_reference(trip): raise FileNotFoundError("Reference intentionally unavailable")
    monkeypatch.setattr(api,"load_reference",missing_reference)
    with TestClient(api.app) as client:
        result=client.post("/predict",json={"trip_id":"00000052"}).json()
    assert result["metrics"] is None
    assert result["network_geometry"] and result["predicted_geometry"]
    assert not any("reference" in key or "label" in key for key in result)
    trip=load_trip("student_test","00000052")
    allowed={tuple(trip.nodes[arc][:,::-1].ravel()) for arc in trip.arcs}
    assert all(tuple(np.array(segment).ravel()) in allowed for segment in result["network_geometry"])
    original=json.loads((ROOT/"results/hmm/00000052.json").read_text())
    assert result["predicted_arc_ids"]==original["predictions"]
    assert result["predicted_route_arc_ids"]==original["route"]
    html=route_map(result["gps"],result["predicted_geometry"],network=result["network_geometry"]).get_root().render()
    assert "Road network" in html and "GPS observations" in html and "Predicted route" in html
    assert "Reference route" not in html

def test_ui_connection_failure(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ui.client import ApiClient
    def timeout(self): raise requests.Timeout("Backend timeout")
    monkeypatch.setattr(ApiClient,"health",timeout)
    ui=AppTest.from_file(str(ROOT/"ui/app.py")).run()
    assert not ui.exception
    assert any("Cannot connect" in item.value for item in ui.error)

