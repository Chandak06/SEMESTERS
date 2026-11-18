"""UI client, evaluator UI and Folium map behaviour."""
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from src.config import ROOT
from src.visualizations import route_map,zoom_for
from ui.client import ApiClient

class FakeResponse:
    def __init__(self,status,body): self.status_code=status; self._body=body; self.ok=status<400; self.text=str(body)
    def json(self): return self._body

def test_client_rejects_bind_address_and_reports_degraded_health(monkeypatch):
    with pytest.raises(ValueError,match="0.0.0.0"): ApiClient("http://0.0.0.0:8000")
    with pytest.raises(ValueError): ApiClient("192.168.1.10:8000")
    monkeypatch.setattr(requests,"get",lambda url,timeout: FakeResponse(503,{"status":"degraded","model_loaded":False,"error":"Missing inference artifact"}))
    assert ApiClient("http://127.0.0.1:8000").health()["model_loaded"] is False
    monkeypatch.setattr(requests,"post",lambda url,json,timeout: FakeResponse(404,{"detail":"Unknown trip ID"}))
    with pytest.raises(requests.HTTPError,match="404: Unknown trip ID"): ApiClient("http://127.0.0.1:8000").predict("99999999")

def test_zoom_and_map_layers():
    assert zoom_for([50.0,8.0],[50.01,8.01])>zoom_for([50.0,8.0],[50.5,8.5])
    html=route_map([[50.0,8.0],None,[50.01,8.01]],[[[50.0,8.0],[50.01,8.01]]],network=[[[50.0,8.0],[50.0,8.01]]],
        classifier=[[[50.0,8.0],[50.01,8.0]]],fit_bounds=False).get_root().render()
    for text in ["Road network","GPS observations","Predicted route","Classifier-only route","Trip start / end","PathSnap legend"]: assert text in html
    assert "Reference route" not in html and "fitBounds" not in html

def test_ui_degraded_backend_message(monkeypatch):
    monkeypatch.setattr(ApiClient,"health",lambda self:{"status":"degraded","model_loaded":False,"error":"Missing inference artifact: best_model.joblib"})
    ui=AppTest.from_file(str(ROOT/"ui/app.py")).run()
    assert not ui.exception
    assert any("model is not ready" in e.value for e in ui.error)
    assert not any("Cannot connect" in e.value for e in ui.error)

def test_ui_full_prediction_through_api(monkeypatch):
    """Drive the Streamlit UI against the real FastAPI app (in-process)."""
    import app.api as api
    with TestClient(api.app) as server:
        def call(method,path,**kw):
            r=getattr(server,method)(path,**kw); assert r.status_code==200,r.text; return r.json()
        monkeypatch.setattr(ApiClient,"health",lambda self:call("get","/health"))
        monkeypatch.setattr(ApiClient,"trips",lambda self:call("get","/trips"))
        monkeypatch.setattr(ApiClient,"predict",lambda self,trip_id:call("post","/predict",json={"trip_id":trip_id}))
        ui=AppTest.from_file(str(ROOT/"ui/app.py"),default_timeout=180).run()
        assert not ui.exception and len(ui.selectbox[0].options)==10
        ui.selectbox[0].select("00000052").run()
        next(b for b in ui.button if b.label=="Predict Route").click().run(timeout=180)
    assert not ui.exception
    assert len(ui.metric)==4 and ui.metric[0].value.endswith("%")
    result=ui.session_state["result"]
    assert result["trip_id"]=="00000052" and result["predicted_route_arc_ids"]
    assert not any("reference" in k or "label" in k for k in result)
    assert any("STUDENT-TESTING" in w.value for w in ui.warning)
