"""PathSnap evaluator interface (Laptop 2).

Only talks to the Laptop 1 FastAPI backend over HTTP. It never loads the model,
trip files or reference routes; all predictions and metrics come from /predict.
"""
import os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
import streamlit as st

from ui.client import ApiClient
from src.visualizations import route_map
st.set_page_config(page_title="PathSnap evaluator",layout="wide")
st.title("PathSnap")
st.caption("Distributed map matching and route reconstruction — evaluator interface (Laptop 2)")

st.sidebar.header("Backend (Laptop 1)")
url=st.sidebar.text_input("Laptop 1 API address",os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000"),
    help="Use Laptop 1's Wi-Fi IPv4 address and port, e.g. http://192.168.1.10:8000")
st.sidebar.button("Re-check connection")
try:
    client=ApiClient(url); health=client.health()
except ValueError as exc:
    st.error(f"Cannot connect to Laptop 1: {exc}"); st.stop()
except requests.Timeout:
    st.error(f"Cannot connect to Laptop 1: no response from {url} within {10} s (timeout).")
    st.info("Check that the backend is running, that both laptops share the same network, and the firewall rule for TCP port 8000.")
    st.stop()
except requests.RequestException as exc:
    st.error(f"Cannot connect to Laptop 1: {exc}")
    st.info("Start the backend, check its Wi-Fi IPv4 address and port, then check the firewall.")
    st.stop()
if not health.get("model_loaded") or health.get("status")!="ok":
    st.sidebar.error("Backend reachable — model NOT ready")
    st.error("Backend is reachable but the model is not ready. Check Laptop 1's /health endpoint and terminal.")
    if health.get("error"): st.code(health["error"])
    st.stop()
try:
    available=client.trips()
except requests.RequestException as exc:
    st.error(f"Backend is reachable but /trips failed: {exc}"); st.stop()
st.sidebar.success("Backend connected — model ready")
st.sidebar.markdown(f"**Mode:** `{health.get('mode')}`  \n**Trips available:** {health.get('available_trips')}")
source=client.url
if st.session_state.get("result_source")!=source:
    st.session_state.pop("result",None)
mode=available["mode"]
if mode=="student_test":
    st.warning("STUDENT-TESTING DEVELOPMENT MODE — these are the ten supplied student-testing trips, not the reserved evaluator trips.")
else:
    st.info("RESERVED EVALUATOR MODE — trips are served from Laptop 1's evaluator_test folder; reference labels stay on Laptop 1.")
if not available["trips"]:
    st.info("No trips supplied for this mode. Place the authorised evaluator trip folders on Laptop 1 when provided, then restart the backend.")
    st.stop()
if mode=="evaluator_test" and len(available["trips"])!=10: st.warning(f"{len(available['trips'])} evaluator trips available; expected ten when the full set is supplied.")
selected=st.selectbox(f"Choose a trip ({len(available['trips'])} available)",available["trips"])
if st.button("Predict Route",type="primary"):
    try:
        with st.spinner(f"Laptop 1 is reconstructing the route for trip {selected}…"):
            started=time.perf_counter()
            st.session_state.result=client.predict(selected)
            st.session_state.result_http_seconds=time.perf_counter()-started
            st.session_state.result_source=source
    except requests.Timeout:
        st.error(f"Prediction request timed out after {client.predict_timeout} s. Check Laptop 1's terminal.")
        st.session_state.pop("result",None)
    except (requests.RequestException,ValueError) as exc:
        st.error(f"Prediction request failed: {exc}")
        st.session_state.pop("result",None)
result=st.session_state.get("result")
if result and result["trip_id"]==selected and result["mode"]==mode:
    metrics=result["metrics"]
    pct=lambda key:f"{metrics[key]:.2%}" if metrics and metrics.get(key) is not None else "Unavailable"
    st.subheader(f"Predicted route for trip {result['trip_id']}")
    a,b,c,d=st.columns(4)
    a.metric("Observation Top-1 accuracy",pct("top1_accuracy"))
    b.metric("Route continuity",pct("route_continuity"))
    c.metric("Server processing time",f"{result['processing_seconds']:.2f} s")
    d.metric("Round-trip time",f"{st.session_state.get('result_http_seconds',result['total_seconds']):.2f} s")
    st.caption(f"Metrics: {result['metric_status']}")
    route=result["predicted_route_arc_ids"]
    matched=sum(a>=0 for a in result["predicted_arc_ids"])
    st.write(f"Predicted route: **{len(route)} road arcs** (original arc IDs) · {matched} of {len(result['predicted_arc_ids'])} GPS observations matched · "
             f"route breaks: {len(result['route_boundaries'])} · candidate road segments shown: {result.get('network_segment_count','—')}")
    st.iframe(route_map(result["gps"],result["predicted_geometry"],network=result.get("network_geometry"),fit_bounds=False,width_px=1000,height_px=600).get_root().render(),
        height=640,alt="Predicted route and GPS observations")
    with st.expander("Predicted road sequence (original arc IDs)"):
        st.code(" → ".join(str(a) for a in route) if route else "No route reconstructed",wrap_lines=True)
    if metrics:
        with st.expander("All returned metrics"):
            st.json(metrics)
