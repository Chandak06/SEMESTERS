# PathSnap — Distributed Map Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon (PES University). PathSnap snaps noisy GPS observations to a directed road network and reconstructs the driven route:

```text
trip files (.track/.nodes/.arcs) → metric projection → k=8 nearest road candidates → 14 features
→ HistGradientBoosting candidate classifier (tuned with Optuna TPE) → HMM/Viterbi on the directed graph
→ route reconstruction (shortest-path bridging) → FastAPI (Laptop 1) → Streamlit + Folium UI (Laptop 2)
```

**Status:** Deliverable 1 is implemented and verified locally (see *Verification* below). Not yet verified: a physical two-laptop run, and the reserved evaluator trips (not supplied yet). Nothing has been pushed or submitted.

## Repository layout

```text
PathSnap/
  src/                  data_loader, preprocessing (metric projection), road_graph, candidate_generation,
                        feature_engineering, reference_alignment (training labels), build_features,
                        train_models (5 models), evaluate_models (metrics), tune_model (Optuna),
                        hmm_viterbi, inference (Predictor), evaluate_hmm, visualizations (plots + Folium)
  app/api.py            Laptop 1 FastAPI backend: GET /health, GET /trips, POST /predict
  ui/app.py             Laptop 2 Streamlit evaluator UI;  ui/client.py = its HTTP client
  artifacts/            logistic, naive_bayes, decision_tree, random_forest, hist_gradient_boosting,
                        tuned_model and best_model (= deployed, tuned HGB) .joblib files
  artifacts/features/   local per-trip feature caches (regenerable; not for publishing)
  configs/              deployment.json (model, feature order, HMM settings), experiment.json,
                        features.json, selection.json, training_samples.json, runtime_versions.json
  data/train|student_test|evaluator_test/   80 / 10 / reserved (empty until supplied) trip folders
  results/eda/          trip inventory, data-quality and feature statistics, projection audit
  results/models/       comparison.csv, per_trip.csv, per-model JSON (incl. CV folds), plots, confusion
                        matrices, fit_diagnostics.csv (train vs CV vs test)
  results/tuning/       trials.csv, trial_history.json, summary.json, comparison.csv (untuned vs tuned)
  results/hmm/          per_trip.csv, summary.json, per-trip predictions, comparison.png
  results/maps/         Folium HTML maps (student_test/<trip>.html for all ten trips)
  results/report/       report figures/tables and genuine browser screenshots (screenshots/)
  results/logs/         execution, test, HTTP, UI and API-server logs
  scripts/              data extraction, verification, map/figure/screenshot/report generation, launchers
  tests/                unit + integration tests (pytest)
  docs/                 methodology.md, completion_audit.md, report/ (Deliverable 2 PDF + source)
```

## 1. Installation (both laptops)

Python 3.11 (tested with 3.11.9) on Windows PowerShell. Always call the venv's `python.exe` (do not run `pip.exe` directly; Windows Application Control blocked it on the development machine).

```powershell
cd 'D:\SEMESTERS\SEM5_2025\2.ML\Hackathon\PathSnap'     # or wherever the project is copied
py -3.11 -m venv .venv                                    # skip if .venv already exists
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Data extraction (Laptop 1 only, once): the original ZIP files are in the parent folder.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
```

It verifies/extracts the 360 trip files into `data/` without overwriting differing copies (SHA-256 manifest in `results/eda/source_manifest.json`). Each trip folder holds headerless whitespace files: `.track` (lon lat seconds), `.nodes` (lon lat), `.arcs` (from-node to-node) and `.route` (chronological arc IDs); IDs are zero-based row numbers and are preserved end-to-end.

## 2. Laptop 1 — model + FastAPI backend

Needs: `src/`, `app/`, `configs/`, `artifacts/*.joblib` (at least `best_model.joblib`), `data/student_test` (and later `data/evaluator_test`). The feature caches are not needed for serving.

```powershell
cd 'D:\SEMESTERS\SEM5_2025\2.ML\Hackathon\PathSnap'
$env:PATHSNAP_MODE = 'student_test'          # development mode: the 10 student-testing trips
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Equivalent: `.\.venv\Scripts\python.exe -m app.api` (reads `PATHSNAP_HOST`, `PATHSNAP_PORT`). Keep this terminal visible during the demo: it prints `PathSnap model loaded; mode=...; available_trips=...` at start-up and one line per request, e.g.

```text
INFO:     PREDICT start trip=00000052 mode=student_test
INFO:     PREDICT trip=00000052 mode=student_test observations=262 route_arcs=93 top1=0.958... inference_seconds=0.233 total_seconds=0.250
INFO:     request=... method=POST path=/predict status=200 seconds=0.271
WARNING:  PREDICT rejected trip=99999999 reason=unknown_trip
```

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `200 {"status":"ok","mode":..,"model_loaded":true,"available_trips":10}`; `503 {"status":"degraded",..,"error":..}` if the model/config is missing, corrupt or incompatible (feature order, parameters) |
| `GET /trips` | trip IDs discovered from `data/<mode>/` at request time |
| `POST /predict` `{"trip_id":"00000052"}` | 422 for malformed IDs, 404 for unknown IDs, 503 if the model is unavailable. Otherwise runs projection → candidates → features → `best_model` → HMM/Viterbi → route, then (server-side only, after prediction) scores against the reference. Returns predicted arc per GPS point, reconstructed route arc IDs and geometry, candidate-corridor road geometry, GPS points, metrics and timings. Reference arcs/labels are never returned. |

Interactive docs: `http://127.0.0.1:8000/docs`.

**Reserved evaluator trips** (when the organisers supply them): copy the ten trip folders into `data\evaluator_test\` on Laptop 1 and restart with

```powershell
$env:PATHSNAP_MODE = 'evaluator_test'
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

If `.route` files are absent, predictions still run and metrics show "Reference data unavailable". Never use this partition for training or tuning.

## 3. Laptop 2 — evaluator UI

Copy the project folder (the UI only uses `ui/`, `src/__init__.py`, `src/config.py`, `src/visualizations.py`; it never loads a model or trip data) and install requirements as in §1.

```powershell
cd 'D:\SEMESTERS\SEM5_2025\2.ML\Hackathon\PathSnap'
$env:PATHSNAP_API_URL = 'http://<LAPTOP1-IPv4>:8000'     # e.g. http://192.168.1.10:8000
.\.venv\Scripts\python.exe -m streamlit run ui\app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open `http://localhost:8501`. The sidebar shows the API address (editable at runtime), a health indicator and the mode. Choose a trip, press **Predict Route**; the UI shows Top-1 accuracy, route continuity, server and round-trip time, route/break counts, the predicted arc sequence and an interactive Folium map (GPS points, candidate road corridor, predicted route, start/end markers, legend, layer control). Student-testing mode is labelled with a yellow banner. Connection errors, timeouts, a degraded backend and API errors are shown as messages instead of crashing.

## 4. Two-laptop checklist

1. Connect both laptops to the same Wi-Fi or a phone hotspot (guest networks often block device-to-device traffic).
2. Laptop 1: find the IPv4 address of the Wi-Fi adapter: `ipconfig` → *Wireless LAN adapter Wi-Fi* → *IPv4 Address*. Do not use `0.0.0.0` or `127.0.0.1` on Laptop 2.
3. Laptop 1: start the backend (§2) and check locally: `Invoke-RestMethod http://127.0.0.1:8000/health`.
4. Laptop 2: test the network path:
   ```powershell
   Test-NetConnection -ComputerName 192.168.1.10 -Port 8000      # TcpTestSucceeded : True
   Invoke-RestMethod http://192.168.1.10:8000/health
   ```
5. If step 4 fails but step 3 works, it is almost always Windows Firewall on Laptop 1. Check the Wi-Fi profile with `Get-NetConnectionProfile` (should be *Private*) and, in an **Administrator** PowerShell on Laptop 1 (device owner's decision; not done by this project), allow the port:
   ```powershell
   New-NetFirewallRule -DisplayName 'PathSnap API 8000' -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
   ```
   Alternatively accept the "Allow access" prompt Windows shows the first time Python listens, ticking *Private networks*. Remove the rule after the event: `Remove-NetFirewallRule -DisplayName 'PathSnap API 8000'`.
6. Laptop 2: start the UI (§3) with that address, or paste it into the sidebar and press *Re-check connection*.
7. Demo: pick a trip → **Predict Route** → show the metrics/map on Laptop 2 and the matching `PREDICT trip=...` line in Laptop 1's terminal (screenshot both for the report).
8. Ports busy? Use another port consistently (`--port 8001` on Laptop 1, `:8001` in the UI URL; `--server.port 8502` for Streamlit).

## 5. Verification (run on Laptop 1)

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q                                  # 27 tests
.\.venv\Scripts\python.exe scripts\verify_existing.py                           # reloads all 7 artifacts, reproduces saved metrics, checks Optuna trials & feature parity
# with the backend running on port 8000 (and the UI on 8501 for the last command):
.\.venv\Scripts\python.exe scripts\smoke_http.py --url http://127.0.0.1:8000    # all 10 trips over real HTTP
$env:PATHSNAP_API_URL='http://127.0.0.1:8000'; .\.venv\Scripts\python.exe scripts\check_ui.py   # Streamlit UI → real API
.\.venv\Scripts\python.exe scripts\capture_screenshots.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501   # headless Edge screenshots
```

## 6. Reproducing experiments (overwrites results — back up `artifacts/`, `configs/`, `results/` first)

```powershell
.\.venv\Scripts\python.exe -m src.build_features                       # candidates, features, proxy labels, EDA tables
.\.venv\Scripts\python.exe scripts\audit_geometry.py                    # projection error audit
.\.venv\Scripts\python.exe -m src.train_models --folds 3 --max-observations 400   # 5 models, grouped CV, student test
.\.venv\Scripts\python.exe -m src.tune_model --trials 10                # Optuna TPE on the selected model
.\.venv\Scripts\python.exe -m src.evaluate_hmm                          # classifier-only vs HMM on 10 student trips
.\.venv\Scripts\python.exe -m src.visualizations                        # model + EDA plots
.\.venv\Scripts\python.exe scripts\generate_maps.py                     # Folium maps for all 10 student trips
.\.venv\Scripts\python.exe scripts\report_figures.py                    # report figures + fit diagnostics (no refitting)
.\.venv\Scripts\python.exe scripts\build_report.py                      # docs/report/PathSnap_Technical_Report.pdf
```

Seed 42 throughout. Trip-grouped 3-fold CV (`GroupKFold`) on the 80 training trips only; the 10 student trips are used only for reporting; the evaluator partition is never read by experiment scripts.

## 7. Results (actual, from `results/`)

Student testing = all 19,160 observations of the ten student trips. Top-1 = share of GPS points whose predicted arc equals the reference-alignment proxy label. Macro F1 / log loss are for the binary "is this candidate the aligned arc" task. Inference time = classifier scoring for all ten trips.

| Model | CV Macro F1 | CV Top-1 | Student Top-1 | Student Macro F1 | Log loss | Continuity | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9030 | 83.50% | 83.66% | 0.9032 | 0.1158 | 91.57% | 0.024 |
| Gaussian Naive Bayes | 0.8169 | 81.98% | 81.00% | 0.8159 | 0.4918 | 91.03% | 0.049 |
| Decision Tree | 0.9047 | 83.59% | 84.67% | 0.9086 | 0.1331 | 91.19% | 0.010 |
| Random Forest | 0.9115 | 84.48% | 84.61% | 0.9110 | 0.0974 | 91.97% | 0.984 |
| HistGradientBoosting | **0.9134** | **84.77%** | **85.28%** | **0.9123** | **0.0949** | 92.23% | 0.828 |
| HGB tuned (deployed) | 0.9125 | 84.90% | 85.25% | 0.9120 | 0.0957 | 92.02% | 1.303 |

Selection rule (`configs/selection.json`): CV Top-1 within 0.5 pp of the best → lowest CV log loss → inference time ⇒ HistGradientBoosting. Optuna TPE: 10 trials (3 random startup + 7 TPE), objective = mean CV (Top-1 − 0.02·log loss); best `max_iter=171, max_leaf_nodes=44, learning_rate=0.1193, l2_regularization=0.0178, min_samples_leaf=22`; objective 0.84573 → 0.84706. Tuning improved CV Top-1 slightly but **not** the student-test metrics.

| Deployed model, 10 student trips | Top-1 | Route continuity | Sequence edit similarity |
|---|---:|---:|---:|
| Classifier only | 85.25% | 92.02% | 0.582 |
| Classifier + HMM/Viterbi | 83.19% | 96.76% | 0.716 |

HMM decoding makes routes more continuous and closer to the reference sequence but lowers pointwise Top-1 on 6 of 10 trips (largest drops: trip 00000007, 92.5% → 84.9%; trip 00000077, 71.2% → 64.3%) and raises it on the other 4. Live API runtimes were 0.23–7.1 s per trip on the development laptop.

## 8. Limitations

- Per-point labels are **proxies** from a monotone geometric alignment of GPS points to the reference route, not independent annotations; 21,128 training points with >80 m residual were excluded from fitting.
- The HMM's emissions are normalised classifier scores, not a calibrated generative model; parameters (β = 30 m, 30 s gap reset, 3 km path cap) were not tuned.
- Ten tuning trials is a small budget. Training uses at most 400 evenly spaced eligible points per training trip.
- Folium/Leaflet tiles need Internet access on Laptop 2; without it the route still draws on a blank background.
- Only trusted, locally produced `.joblib` files should be loaded (joblib uses pickle).

## 9. AI assistance

This project was developed with AI coding assistants (OpenAI Codex earlier, Anthropic Claude Code for the completion audit, UI/map fixes, figures, screenshots and report draft). All metrics come from scripts in this repository. The team must review the code, results and report and be able to explain them before submission, and disclose this assistance as the event policy requires.
