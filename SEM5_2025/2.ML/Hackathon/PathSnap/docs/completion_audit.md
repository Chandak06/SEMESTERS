# Completion audit - 18 November 2026

Audit conducted before implementation changes. All Python implementation files in src/, scripts/, app/ and ui/, the README, tests and five official PDFs were inspected.

## Already working
- Original-ID loaders, metre-based geometry, candidate index and reference alignment.
- Fourteen deterministic features; learned scaling embedded in logistic and naive Bayes pipelines. The selected tree model requires no fitted scaler.
- Five baseline artifacts plus tuned/deployed artifacts load and predict on real cached student features.
- 80 training trips and ten student-testing trips; evaluator directory empty.
- Ten recorded Optuna TPE trials, grouped folds and tuned/untuned comparisons.
- Log-space HMM/Viterbi with directed distances, backtracking and gap/disconnection handling.
- app.api:app FastAPI backend, ui/app.py Streamlit client and existing offline Folium map.
- Existing dependencies pass pip check.

## Gaps to fix
- UI does not show backend health or model readiness.
- Evaluator map omits relevant road-network geometry, although offline maps include it.
- A previously displayed prediction can survive a backend URL change.
- Startup lacks explicit model/feature-schema compatibility checks and a clear degraded-health response.
- Regression coverage does not explicitly test missing/corrupt artifacts, bad feature order or new UI/network behavior.
- Technical report and actual browser-rendered screenshots are absent.
- Historical completion scripts contain stale hardcoded status and could overwrite newer audit conclusions.

No evidence yet requires model retraining, feature changes, decoder changes or replacement of stored experiments. Verification outputs and source backups will be kept under results/completion_audit/. Existing models and experiment CSV/JSON/plots will remain unchanged.

## External evidence gaps
Reserved evaluator trips are not supplied. Two physical laptops and the team's identification details are unavailable. Cross-device Wi-Fi/firewall evidence and the receiving laptop's terminal screenshot require the team. They will be disclosed rather than fabricated.


## Second completion pass (Claude Code, 18 November 2026, afternoon)
Re-audited all code, results and artifacts; re-ran `scripts/verify_existing.py` (all 7 artifacts reproduce saved metrics; features match cache). No retraining, retuning or HMM re-runs; no experiment outputs changed.
Fixed/added: UI client now distinguishes a degraded backend (503) from an unreachable one and surfaces API error details; rejects 0.0.0.0; UI map uses a computed zoom (fitBounds inside the Streamlit iframe zoomed out too far), start/end markers, swatch legend, expanded layer control; UI shows predicted arc sequence and round-trip time; button renamed "Predict Route". New: `scripts/generate_maps.py` (10 maps), `scripts/report_figures.py` (EDA, fit diagnostics without refitting, tuning, HMM plots), `scripts/capture_screenshots.py` (headless Edge, CDP), `scripts/build_report.py` (Deliverable 2 PDF), `tests/test_ui_maps.py` (4 tests). One-off patch scripts moved to `scripts/legacy/`. README rewritten with exact commands.
Live check used ports 8010/8521 because older servers from the morning pass still occupy 8000, 8011, 8501 and 8511 (they run older code and should be stopped before the demo).
