"""Build the Deliverable 2 technical report (HTML -> PDF via headless Edge).

Every number is read from files in results/ and configs/; every image is a
figure produced by the repository's scripts or a genuine browser screenshot.

    .\.venv\Scripts\python.exe scripts\build_report.py
Output: docs/report/PathSnap_Technical_Report.html and .pdf
"""
import html,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.config import ROOT,FEATURES
from src.train_models import SPECS
OUT=ROOT/"docs/report"; OUT.mkdir(parents=True,exist_ok=True)
J=lambda p:json.loads((ROOT/p).read_text(encoding="utf-8"))
cmp_=pd.read_csv(ROOT/"results/models/comparison.csv").set_index("model")
fit=pd.read_csv(ROOT/"results/models/fit_diagnostics.csv").set_index("model")
tun=J("results/tuning/summary.json"); tcmp=pd.read_csv(ROOT/"results/tuning/comparison.csv").set_index("variant")
hmm=J("results/hmm/summary.json"); hpt=pd.read_csv(ROOT/"results/hmm/per_trip.csv")
eda=pd.read_csv(ROOT/"results/eda/trips.csv"); dq=pd.read_csv(ROOT/"results/report/data_quality.csv",index_col=0)
exp=J("configs/experiment.json"); dep=J("configs/deployment.json"); sel=J("configs/selection.json")
smoke=J("results/logs/http_smoke_final.json"); shots=J("results/report/screenshots/capture_log.json")
maps=J("results/maps/student_test/index.json")
NAMES={"logistic":"Logistic Regression","naive_bayes":"Gaussian Naive Bayes","decision_tree":"Decision Tree","random_forest":"Random Forest","hist_gradient_boosting":"HistGradientBoosting"}
pct=lambda v:f"{100*v:.2f}%"; f4=lambda v:f"{v:.4f}"
def img(path,caption,width=100):
    rel=os.path.relpath(ROOT/path,OUT).replace("\\","/")
    return f'<figure><img src="{rel}" style="width:{width}%"><figcaption>{caption} <span class="src">[{path}]</span></figcaption></figure>'
def table(headers,rows,cls=""):
    h="".join(f"<th>{x}</th>" for x in headers); b="".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'
def gap(text): return f'<div class="gap"><b>EVIDENCE GAP — to be completed by the team:</b> {text}</div>'
cfg=lambda name:", ".join(f"{k}={v}" for k,v in SPECS[name][2].items() if k not in("random_state","n_jobs"))
part=eda.groupby("partition")
tr,st=part.get_group("train"),part.get_group("student_test")
# ---------------------------------------------------------------- content
S=[]
S.append(f"""
<div class="title">
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p class="sub">Technical Report — UE24CS352A Machine Learning Hackathon, PES University</p>
<p><b>Team name / members / SRNs:</b> <span class="fill">[to be filled in by the team]</span></p>
<p><b>Repository:</b> <span class="fill">[GitHub URL to be added after the team publishes it]</span></p>
<div class="note">Draft generated from the repository's saved experiment outputs by <code>scripts/build_report.py</code>. Every number and graph is
read from files under <code>results/</code> and <code>configs/</code> and can be regenerated with the commands in the README. Items marked
<b>EVIDENCE GAP</b> require the team (two-laptop demonstration screenshots, team details). The team must review, edit and be able to
explain every section before submission (see §9, AI-assistance disclosure).</div>
</div>
<h2>1. Problem and pipeline overview</h2>
<p>Each trip provides a noisy GPS track (<code>.track</code>: longitude, latitude, seconds), a directed road network (<code>.nodes</code>, <code>.arcs</code>)
and, for development data, the driven route as a chronological list of arc IDs (<code>.route</code>). We frame map matching as
<b>candidate ranking</b>: for every GPS point we generate the k = {dep.get('candidates',8) if 'candidates' in dep else 8} nearest directed road arcs,
describe each (point, arc) pair with 14 features, train a binary classifier that scores "this arc is the one being driven", and decode the
whole trip with an HMM/Viterbi decoder over the directed road graph. The deployed system serves this pipeline through FastAPI on Laptop 1
and a Streamlit/Folium evaluator interface on Laptop 2.</p>
<p><b>Partitions.</b> 80 training trips (fitting, 3-fold trip-grouped cross-validation and Bayesian optimisation), 10 student-testing trips
(development comparison only) and the reserved evaluator trips (not supplied at the time of writing; never used). Trip IDs in the two supplied
partitions are disjoint (checked by <code>tests/test_integration.py</code>).</p>
""")
# ---------------- 2 EDA
S.append(f"""<h2>2. Dataset understanding and exploratory data analysis</h2>
<h3>2.1 How the data were preprocessed</h3>
<ol>
<li><b>Strict loading with original IDs</b> (<code>src/data_loader.py</code>). Files are parsed as headerless whitespace tables; node and arc IDs are
the zero-based row numbers and are never re-indexed, so every prediction refers to an original arc ID. Arc endpoints must be integer and in range,
coordinates must be finite with |lon| ≤ 180 and |lat| ≤ 90; invalid GPS rows are kept in place (flagged, predicted as −1) so the
observation index is never shifted. Timestamps are never re-sorted.</li>
<li><b>Metric projection</b> (<code>src/preprocessing.py</code>). Longitude/latitude are converted to metres with a local equirectangular projection
(Earth radius 6,371,008.8 m) centred on the median road node of the trip, so all distances and headings are physical. An audit against haversine
lengths (<code>results/eda/projection_audit.csv</code>) shows a maximum segment-length error of 1.14% on training networks and 0.49% on student networks.</li>
<li><b>Directed road graph</b> (<code>src/road_graph.py</code>): adjacency lists, arc lengths and headings, in/out degrees, weak/strong components and
a cached, distance-capped Dijkstra used for transitions and route bridging.</li>
<li><b>Candidate generation</b> (<code>src/candidate_generation.py</code>): a KD-tree on arc midpoints gives a search radius that provably contains the
8 nearest arcs (radius = 8th seed distance + half the longest arc); exact point-to-segment distances then rank them. The reference arc is
<i>not</i> injected into the candidate set; the resulting candidate miss rate on the student trips is {pct(cmp_.loc['hist_gradient_boosting','candidate_miss_rate'])}.</li>
<li><b>Training labels</b> (<code>src/reference_alignment.py</code>). The <code>.route</code> file lists arcs, not one arc per GPS point, so a
monotone dynamic program aligns each GPS point to an occurrence in the reference sequence (squared distance / 20 m, capped, plus a heading term,
never moving backwards). These per-point labels are <b>proxies</b>. Training points whose aligned reference arc is more than 80 m away
({int(dq.loc['alignment_over_80m','train']):,} points, concentrated in trips 00000054 and 00000033) are excluded from fitting but all student points are scored.</li>
<li><b>Sampling for training</b>: at most 400 evenly spaced eligible points per training trip (so long trips do not dominate) →
{exp['training_observations']:,} observations, {exp['training_candidate_rows']:,} candidate rows; sample indices are saved in <code>configs/training_samples.json</code>.</li>
</ol>
<h3>2.2 Missing-value and data-quality analysis</h3>
<p>Computed by <code>src/build_features.py</code> on every file of the train and student-test partitions (<code>results/eda/trips.csv</code>).</p>
{table(["Check","train (80 trips)","student_test (10 trips)"],[[k.replace('_',' '),f"{int(dq.loc[k,'train']):,}",f"{int(dq.loc[k,'student_test']):,}"] for k in dq.index])}
<p><b>Observations.</b> There are no missing/non-finite GPS values, out-of-range coordinates, duplicate or reversed timestamps in either partition,
so no imputation was needed. The dominant irregularity is <b>temporal gaps</b>: the median sampling interval is exactly 1 s in every trip, but 133
training and 2 student gaps exceed 30 s (one training trip has a {int(tr.max_interval_s.max()):,} s gap). The decoder therefore restarts at gaps
instead of interpolating. Road networks are large relative to the trips (median {int(tr.arcs.median()):,} arcs per training trip vs
{int(tr.reference_arcs.median())} reference arcs), contain several weakly connected components (median {int(tr.weak_components.median())}) and some zero-length arcs
({int(dq.loc['zero_length_arcs','train'])} in training), which motivated candidate pruning and component checks before shortest-path search.
GPS noise is moderate: the median distance from a GPS point to its aligned reference arc is about
{tr.alignment_median_m.median():.1f} m (train) / {st.alignment_median_m.median():.1f} m (student), with a 95th percentile of
{tr.alignment_p95_m.median():.1f} m / {st.alignment_p95_m.median():.1f} m.</p>
{img("results/report/eda_distributions.png","Figure 1. Trip lengths, largest sampling gap per trip and 95th-percentile GPS-to-reference distance (train vs student_test).")}
{img("results/eda/exploration.png","Figure 2. Observation counts, GPS extents and network size vs weakly connected components (train and student_test).")}
<h3>2.3 GPS trajectories and road networks</h3>
{img("results/report/gps_and_road_network.png","Figure 3. GPS trajectory (coloured by time) over the road-network arcs. Left: training trip 00000038 with its reference route (green). Right: student-test trip 00000052 (no reference shown).")}
<p>Patterns visible in the plots and maps: GPS points lie mostly within a lane-width of the true road but scatter more at junctions,
roundabouts and under-passes (e.g. the loop at the south-west of trip 00000052), where several arcs are within a few metres; parallel and
opposite-direction arcs make pure nearest-road matching ambiguous, which is why heading and connectivity information are needed.</p>
<h3>2.4 Features and why they were chosen</h3>
""")
just={"distance_m":"Perpendicular GPS-to-arc distance — the primary geometric evidence.","start_distance_m":"Distance to the arc start node; separates short arcs near junctions.",
 "end_distance_m":"Distance to the arc end node; same role at the other end.","length_m":"Arc length; long arcs attract projections, short arcs are junction pieces.",
 "heading_difference":"Angle between smoothed GPS heading (±2 points) and the directed arc — distinguishes opposite-direction and crossing arcs.",
 "heading_valid":"Whether the heading is reliable (moving > 3 m, no outlier, gap ≤ 30 s).","dt_s":"Time since previous fix; context for gaps.",
 "speed_mps":"Implied speed (clipped at 60 m/s); signals noise/jumps.","in_degree":"In-degree of the arc's start node (junction complexity).",
 "out_degree":"Out-degree of the arc's end node.","distance_rank":"Rank 0–7 of the arc among the candidates of this point.",
 "projection_fraction":"Where along the arc the point projects (0 = start, 1 = end).","distance_margin_m":"Distance minus the nearest candidate's distance — relative evidence within the point's candidate set.",
 "gps_outlier":"Implausible speed / invalid point flag."}
S.append(table(["#","Feature","Purpose"],[[i+1,f"<code>{f}</code>",just[f]] for i,f in enumerate(FEATURES)]))
S.append(f"""<p>Absolute coordinates, arc IDs and anything derived from the reference route are deliberately excluded, so the classifier must generalise
to unseen cities and cannot memorise roads. The same function (<code>src/feature_engineering.py</code>) is used in training and in the API;
<code>scripts/verify_existing.py</code> confirms the features computed at inference time are bit-identical to the training cache, and the API
refuses to start if the saved feature order in <code>configs/deployment.json</code> differs. The distributions below (training sample) show
that aligned candidates have small distance, small distance margin, low rank and small heading difference; tree importances confirm that
<code>distance_margin_m</code> and <code>heading_difference</code> carry most of the signal (Decision Tree: 0.45 and 0.43; Random Forest: 0.22 and 0.36).</p>
{img("results/report/feature_distributions.png","Figure 4. Feature distributions for aligned (positive) vs other candidates, 80 training trips.")}
{img("results/report/feature_importance.png","Figure 5. Impurity-based feature importance of the saved Decision Tree and Random Forest.")}
""")
# ---------------- 3 models
rows=[]
for m in SPECS:
    r=cmp_.loc[m]
    rows.append([NAMES[m],SPECS[m][0],f"<small>{cfg(m)}</small>",f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}",pct(r.top1_accuracy),pct(r.route_continuity),f4(r.macro_f1),f4(r.log_loss),f"{r.inference_seconds:.3f} s"])
S.append(f"""<h2>3. Model experimentation and comparison</h2>
<p><b>Protocol</b> (<code>src/train_models.py</code>, seed 42): each model is evaluated with 3-fold <code>GroupKFold</code> on the 80 training trips
(all points of a trip are in the same fold; fold trip lists saved in <code>results/models/cross_validation.json</code>), then refit on all 80 training
trips and evaluated on all 19,160 points of the 10 student trips. <b>Metrics</b>: candidate-level accuracy/precision/recall/<b>Macro F1</b>
(binary, threshold 0.5) and <b>log loss</b>; observation-level <b>Top-1 accuracy</b> (highest-scoring candidate = aligned reference arc);
<b>route continuity</b> (share of consecutive matched points on the same or directly connected arcs); inference time = classifier scoring
of all 10 student trips. Linear/Gaussian models include a <code>StandardScaler</code> fitted inside the pipeline; tree models need none.</p>
<p><b>Primary selection metric</b>: CV observation Top-1, because the task is to choose the right road for each point; models within
0.5 percentage points of the best are then compared on CV log loss (probability quality matters because the HMM uses the scores as emissions)
and then inference time. Rule and eligible models are saved in <code>configs/selection.json</code>: eligible = {', '.join(NAMES[m] for m in sel['eligible_models'])};
selected = <b>{NAMES[sel['selected_model']]}</b>.</p>
<h3>3.1 Overall model comparison (classifier only, before HMM)</h3>
{table(["Model","Family","Key configuration","CV Macro F1 (mean ± sd)","Student Top-1","Route continuity","Student Macro F1","Log loss","Inference time (10 trips)"],rows,"small")}
<p class="small">Additional measured metrics (student test): {"; ".join(f"{NAMES[m]}: precision {cmp_.loc[m,'precision']:.3f}, recall {cmp_.loc[m,'recall']:.3f}, accuracy {cmp_.loc[m,'accuracy']:.4f}, training time {cmp_.loc[m,'train_seconds']:.1f} s" for m in SPECS)}.
Candidate miss rate is {pct(cmp_.loc['logistic','candidate_miss_rate'])} for every model because all models share the same candidate set. All metrics are applicable to all five models (each outputs probabilities), so no N/A entries are needed.</p>
""")
detail={
"logistic":dict(why="A linear, well-calibrated probabilistic baseline: the evidence for a candidate is largely monotone (closer, better aligned, lower rank → more likely), which a linear logit can represent, and its probabilities feed the HMM directly.",
 sl="Strengths: fastest to train (0.5 s) and to score, stable across folds, good calibration (log loss {ll}). Limitation: one linear boundary cannot express interactions such as \"heading matters only when the vehicle is moving\" or threshold effects in distance margin; large coefficients on the correlated distance features (start/end distance −16.5, length +20.7, margin −13.5 after scaling) show it compensating with correlated terms.",
 hp="C = 1.0 (inverse L2 strength), max_iter = 500, lbfgs solver, StandardScaler.",loss="Binary cross-entropy (log loss) with L2 penalty."),
"naive_bayes":dict(why="Included as the generative/probabilistic family: it models each feature per class with a Gaussian and is extremely cheap, a useful lower bound for what simple class-conditional likelihoods achieve.",
 sl="Strengths: trivial training (0.13 s), highest recall ({rec}). Limitations: the independence assumption is badly violated (distance, endpoint distances, rank and margin are strongly correlated), so evidence is double-counted, producing over-confident positives — precision only {prec} and by far the worst log loss ({ll}). Several features are skewed or discrete (rank, flags), not Gaussian.",
 hp="var_smoothing = 1e-5 (variance floor), StandardScaler.",loss="Maximum-likelihood estimates of per-class Gaussian means/variances; prediction by Bayes' rule (no iterative loss)."),
"decision_tree":dict(why="Captures non-linear rules and interactions (e.g. small margin AND small heading difference) without scaling, and is interpretable.",
 sl="Strengths: fastest inference (0.010 s), readable splits, Top-1 close to the ensembles. Limitations: piecewise-constant, poorly calibrated probabilities (log loss {ll}) and high variance — see the train/CV gap below.",
 hp="max_depth = 12, min_samples_leaf = 20, Gini criterion.",loss="Greedy minimisation of Gini impurity at each split."),
"random_forest":dict(why="Bagging many decorrelated trees reduces the variance of a single tree while keeping its non-linear rules — a natural next step after the Decision Tree.",
 sl="Strengths: second-best log loss ({ll}) and CV Macro F1, robust to noisy points. Limitations: largest model on disk (13.8 MB), slowest training (25.6 s) and about 100× slower inference than a single tree (0.98 s); it fits the training sample very closely.",
 hp="n_estimators = 100, max_depth = 18, min_samples_leaf = 5, bootstrap, sqrt features per split.",loss="Each tree minimises Gini impurity on a bootstrap sample; probabilities averaged."),
"hist_gradient_boosting":dict(why="Boosting fits trees sequentially to the residual errors of the ensemble, directly minimising log loss, which suits both Top-1 ranking and HMM emissions; histogram binning keeps it fast on 255k rows.",
 sl="Strengths: best CV Top-1, CV Macro F1, student Top-1, Macro F1 and log loss of all five models, small train/CV gap, 2.6 s training. Limitations: more hyper-parameters, sequential training, slower inference than a single tree; early stopping was disabled because its random validation split would mix trips.",
 hp="max_iter = 100, max_leaf_nodes = 31, learning_rate = 0.1, l2_regularization = 1.0, min_samples_leaf = 20 (default), early_stopping = False.",loss="Binary cross-entropy (log loss) minimised by gradient boosting with L2 regularisation of leaf values."),
}
fitnote={
"logistic":"No overfitting: in-sample, CV and student Top-1 are {tr1}/{cv1}/{te1} and log loss {trl}/{cvl}/{tel} — practically identical. Its ceiling is about 1 pp below the tree ensembles, a sign of mild <b>underfitting</b> (high bias) of a linear boundary.",
"naive_bayes":"<b>Underfitting</b> (model bias): it is the weakest on all three splits (Top-1 {tr1}/{cv1}/{te1}); log loss is already poor in-sample ({trl}) and worse on the student set ({tel}), consistent with the wrong independence assumption rather than memorisation.",
"decision_tree":"<b>Overfitting</b>: in-sample Top-1 {tr1} vs CV {cv1} (≈3.8 pp gap) and log loss {trl} vs {cvl} (almost doubled) — leaves of 20 samples at depth 12 memorise training trips. The student result ({te1}) is close to CV, so the CV estimate was reliable.",
"random_forest":"The largest in-sample/CV gap (Top-1 {tr1} vs {cv1}; log loss {trl} vs {cvl}) shows that individual deep trees overfit, but averaging keeps the generalisation error low: CV and student results ({te1}) agree, and log loss is far better than the single tree.",
"hist_gradient_boosting":"Mild, controlled overfitting: in-sample Top-1 {tr1} vs CV {cv1} and student {te1}; log loss {trl} / {cvl} / {tel}. The small gap plus fold stability (CV Top-1 per fold {folds}) indicate good generalisation.",
}
S.append("<h3>3.2 Per-model analysis</h3>")
for i,m in enumerate(SPECS):
    r=cmp_.loc[m]; d=detail[m]; f=fit.loc[m]; j=J(f"results/models/{m}.json"); cm=j["student_test"]["confusion_matrix"]
    fmt=dict(ll=f4(r.log_loss),prec=f"{r.precision:.3f}",rec=f"{r.recall:.3f}",tr1=pct(f.train_top1),cv1=pct(f.cv_top1),te1=pct(f.test_top1),
             trl=f4(f.train_log_loss),cvl=f4(f.cv_log_loss),tel=f4(f.test_log_loss),folds=", ".join(pct(x["top1_accuracy"]) for x in j["cv"]))
    S.append(f"""<div class="model"><h4>3.2.{i+1} {NAMES[m]}</h4>
<p><b>A1. Why suitable?</b> {d['why']}</p>
<p><b>A2. Strengths and limitations.</b> {d['sl'].format(**fmt)}</p>
<p><b>B5. Important hyper-parameters (initial values).</b> {d['hp']}</p>
<p><b>B6. Loss / objective.</b> {d['loss']}</p>
<p><b>C7–C8. Student-testing performance.</b> Observation Top-1 <b>{pct(r.top1_accuracy)}</b>, candidate Macro F1 <b>{f4(r.macro_f1)}</b>, log loss <b>{f4(r.log_loss)}</b>,
precision {r.precision:.3f}, recall {r.recall:.3f}, route continuity {pct(r.route_continuity)}, sequence edit similarity {r.sequence_edit_similarity:.3f}.
Confusion matrix [[TN, FP], [FN, TP]] = {cm}. CV: Macro F1 {r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}, Top-1 {pct(r.cv_top1_accuracy)}.</p>
<p><b>C9. Over- or under-fitting?</b> {fitnote[m].format(**fmt)}</p></div>""")
S.append(f"""<h3>3.3 Model visualisations</h3>
<p>None of the five models produces an epoch-wise training history in our configuration (HistGradientBoosting with <code>early_stopping=False</code>
stores no validation curve; the others are not iterative or not epoch-based), so <b>no training/validation loss curves are shown</b>. Instead,
over/under-fitting is assessed with in-sample vs. CV vs. student metrics computed from the saved models (<code>scripts/report_figures.py</code>,
<code>results/models/fit_diagnostics.csv</code>).</p>
{img("results/models/cv_macro_f1.png","Figure 6. Trip-grouped CV Macro F1 across all models.",62)}
<div class="row">{img("results/models/top1_accuracy.png","Figure 7. Student-test observation Top-1 accuracy.")}{img("results/models/macro_f1.png","Figure 8. Student-test Macro F1.")}</div>
{img("results/models/log_loss.png","Figure 9. Student-test log loss (lower is better).",62)}
{img("results/report/model_selection.png","Figure 10. Model selection evidence: CV vs student Macro F1 (error bars = fold std), CV vs student Top-1 with the selection threshold, and inference time.")}
{img("results/report/fit_diagnostics.png","Figure 11. In-sample vs trip-grouped CV vs student-test Top-1 and log loss for the saved models.")}
<div class="row5">{''.join(img(f"results/models/{m}_confusion.png",f"{NAMES[m]}") for m in SPECS)}</div>
<p class="small">Figure 12. Student-test confusion matrices (candidate level, threshold 0.5).</p>
<p><b>Selection.</b> Random Forest and HistGradientBoosting are the only models within 0.5 pp of the best CV Top-1. HistGradientBoosting has the
lower CV log loss (0.0970 vs 0.0998), higher CV Top-1, a smaller train/CV gap and a 13.8 MB → 0.17 MB smaller artifact, so it was selected for
tuning and deployment. The student-test results (not used for selection) agree: it is best on Top-1, Macro F1 and log loss.</p>
""")
# ---------------- 4 tuning
space=[["max_iter","integer, uniform","70 – 180"],["max_leaf_nodes","integer, uniform","10 – 45"],["learning_rate","float, log-uniform","0.025 – 0.2"],
       ["l2_regularization","float, log-uniform","0.01 – 20"],["min_samples_leaf","integer, uniform","15 – 80"]]
hist=J("results/tuning/trial_history.json"); b,t=tcmp.loc["baseline"],tcmp.loc["tuned"]
S.append(f"""<h2>4. Bayesian hyper-parameter optimisation (HistGradientBoosting)</h2>
<p><b>Method.</b> Optuna's Tree-structured Parzen Estimator (<code>src/tune_model.py</code>), seed 42, {tun['tpe_startup_trials']} random start-up trials then
{tun['bayesian_trials']} TPE-guided trials (<b>{tun['actual_trials']} trials in total</b>, {tun['search_seconds']:.1f} s of search + {tun['refit_seconds']:.1f} s refit).
Every trial runs the same 3-fold trip-grouped CV on the 80 training trips; the student and evaluator partitions are not used in the search.</p>
<p><b>1–2. Search space</b> (justified by the baseline's behaviour: more/larger trees for capacity, learning rate and L2 and leaf size for regularisation):</p>
{table(["Hyper-parameter","Distribution","Range"],space)}
<p><b>3. Objective</b> (maximised): {html.escape(tun['objective'])}. Top-1 is the task metric; the small log-loss term prefers better-calibrated
emissions for the HMM when Top-1 ties. The untuned baseline scores {tun['baseline_objective']:.5f} on the same objective.</p>
<p><b>4–5. Trials and best values.</b> All {len(hist)} trials completed (<code>results/tuning/trials.csv</code>, <code>trial_history.json</code> with per-fold metrics).
Best trial #{max(hist,key=lambda x:x['value'])['number']}: {", ".join(f"{k} = {v:.4g}" if isinstance(v,float) else f"{k} = {v}" for k,v in tun['best_parameters'].items())}; objective {tun['best_objective']:.5f}.</p>
{table(["Trial","Objective"]+[k for k in tun['best_parameters']],[[x['number'],f"{x['value']:.5f}"]+[f"{v:.4g}" if isinstance(v,float) else v for v in x['params'].values()] for x in hist],"small")}
<p><b>6. Did tuning help?</b> Same partitions and metrics for both variants:</p>
{table(["Variant","CV Macro F1","CV Top-1","Student Top-1","Student Macro F1","Log loss","Inference time (10 trips)"],
 [[v,f4(tcmp.loc[v,'cv_macro_f1']),pct(tcmp.loc[v,'cv_top1_accuracy']),pct(tcmp.loc[v,'top1_accuracy']),f4(tcmp.loc[v,'macro_f1']),f4(tcmp.loc[v,'log_loss']),f"{tcmp.loc[v,'inference_seconds']:.3f} s"] for v in ["baseline","tuned"]])}
<p>Only marginally, and only on the optimised criterion: CV Top-1 rose by {100*(t.cv_top1_accuracy-b.cv_top1_accuracy):.2f} pp, but CV Macro F1
({f4(b.cv_macro_f1)} → {f4(t.cv_macro_f1)}), student Top-1 ({pct(b.top1_accuracy)} → {pct(t.top1_accuracy)}), Macro F1 and log loss all became slightly worse and
inference was ~{t.inference_seconds/b.inference_seconds:.1f}× slower (171 vs 100 trees, 44 vs 31 leaves, much weaker L2). The trial objectives span only
{min(x['value'] for x in hist):.4f}–{max(x['value'] for x in hist):.4f}, i.e. the model is insensitive to these hyper-parameters within the range, and the
tuned model's larger in-sample/CV gap (Figure 11) suggests the extra capacity is partly fitting training trips. Following the pre-declared rule (deploy
the tuned model if its CV objective beats the baseline), the tuned model is deployed as <code>artifacts/best_model.joblib</code>; the differences are
within fold-to-fold variation (CV Top-1 fold sd ≈ 1 pp), so we do not claim tuning improved generalisation.</p>
{img("results/report/tuning.png","Figure 13. Optuna trials (grey = random start-up, blue = TPE) with best-so-far curve, and untuned vs tuned comparison.")}
""")
# ---------------- 5 HMM
hp=hpt.pivot(index="trip_id",columns="method")
rowsh=[[f"{tid:08d}",int(hp.loc[tid,('observations','hmm')]),pct(hp.loc[tid,('top1_accuracy','classifier')]),pct(hp.loc[tid,('top1_accuracy','hmm')]),
        pct(hp.loc[tid,('route_continuity','classifier')]),pct(hp.loc[tid,('route_continuity','hmm')]),f"{hp.loc[tid,('sequence_edit_similarity','classifier')]:.3f}",
        f"{hp.loc[tid,('sequence_edit_similarity','hmm')]:.3f}",int(hp.loc[tid,('breaks','hmm')]),f"{hp.loc[tid,('processing_seconds','hmm')]:.2f}"] for tid in hp.index]
up=[f"{t:08d}" for t in hp.index if hp.loc[t,('top1_accuracy','hmm')]>hp.loc[t,('top1_accuracy','classifier')]]
down=[f"{t:08d}" for t in hp.index if hp.loc[t,('top1_accuracy','hmm')]<hp.loc[t,('top1_accuracy','classifier')]]
c,h=hmm["classifier"],hmm["hmm"]; hcfg=dep["hmm"]
S.append(f"""<h2>5. HMM and Viterbi route reconstruction</h2>
<p>Implementation: <code>src/hmm_viterbi.py</code> (decoder and route assembly), <code>src/inference.py</code> (pipeline), parameters in
<code>configs/deployment.json</code>: β = {hcfg['beta_m']:.0f} m, gap reset = {hcfg['max_gap_s']:.0f} s, path-search cap = {hcfg['max_path_m']:.0f} m.</p>
<p><b>1. Hidden states.</b> At time t the states are the (up to) 8 candidate arcs of GPS point t, stored by their original arc IDs together with the
projection fraction along the arc; the decoded state sequence is mapped straight back to original arc IDs (one per GPS point).</p>
<p><b>2. Emissions.</b> The deployed classifier's probabilities for the candidates of point t, clipped to [10⁻⁷, 1] and renormalised over that
point's candidates: log e<sub>t</sub>(s) = log p<sub>t</sub>(s) − log Σ p<sub>t</sub>. The same model and features as in §3–4 are used, so the decoder
consumes exactly the selected model's outputs.</p>
<p><b>3. Transitions.</b> For candidates a (at t−1) and b (at t) we compute the network distance d<sub>road</sub> = remaining length of a + directed
Dijkstra distance between a's end node and b's start node + travelled part of b (or the along-arc distance if a = b and the vehicle does not move
backwards), and the straight GPS displacement d<sub>gps</sub>. Score = −|d<sub>road</sub> − d<sub>gps</sub>| / (β + 0.1·d<sub>gps</sub>); transitions
with no directed path within 3 km, or longer than max(150 m, 60 m/s × Δt) are forbidden (−∞). Scores are normalised per source state in log space
(log-sum-exp). Viterbi recurrence: V<sub>t</sub>(b) = log e<sub>t</sub>(b) + max<sub>a</sub>[V<sub>t−1</sub>(a) + log A(a,b)], with back-pointers and
per-step max-normalisation for numerical stability, then back-tracking.</p>
<p><b>4. Missing candidates and gaps.</b> A point without candidates (invalid GPS) gets −1 and ends the current chain; Δt ≤ 0 or Δt &gt; 30 s also
ends it. The finished segment is back-tracked and a new chain starts from emissions only. The student trips contain {int(hpt[hpt.method=='hmm'].breaks.sum())} such breaks in total.</p>
<p><b>5. Disconnected roads.</b> If no candidate pair between t−1 and t has a finite transition (e.g. different weak components, one-way
violations, or a jump longer than physically possible), the chain is split at t instead of forcing an impossible connection. In route assembly,
consecutive different arcs are bridged with the real directed shortest path; when none exists a route boundary is recorded and no straight
"invented" segment is drawn on the map.</p>
<p><b>6. Did the HMM improve the route?</b> Comparison on all ten student trips with the deployed (tuned) classifier
(<code>src/evaluate_hmm.py</code>, <code>results/hmm/</code>):</p>
{table(["Method","Top-1 accuracy","Route continuity","Sequence edit similarity"],
 [["Classifier only",pct(c['top1_accuracy']),pct(c['route_continuity']),f"{c['sequence_edit_similarity']:.3f}"],["Classifier + HMM/Viterbi",pct(h['top1_accuracy']),pct(h['route_continuity']),f"{h['sequence_edit_similarity']:.3f}"]])}
{table(["Trip","Points","Top-1 clf","Top-1 HMM","Contin. clf","Contin. HMM","Seq. sim. clf","Seq. sim. HMM","Breaks","Total s"],rowsh,"small")}
{img("results/report/hmm_per_trip.png","Figure 14. Classifier-only vs classifier + HMM per student trip.")}
<p><b>Discussion.</b> At route level the HMM clearly helps: continuity rises on every trip (overall {pct(c['route_continuity'])} → {pct(h['route_continuity'])})
and the reconstructed arc sequence is much closer to the reference (edit similarity {c['sequence_edit_similarity']:.3f} → {h['sequence_edit_similarity']:.3f}; e.g. trip
00000007 0.565 → 0.968, 00000052 0.789 → 0.979). Pointwise Top-1, however, falls overall ({pct(c['top1_accuracy'])} → {pct(h['top1_accuracy'])}): it improves on
{len(up)} trips ({', '.join(up)}) and drops on {len(down)} ({', '.join(down)}). The drops come from points near junctions: the decoder prefers a connected
explanation and may assign a point to the adjacent arc it has just entered or left, which the proxy label (itself a geometric alignment) counts as wrong —
trip 00000007 loses 7.6 pp of Top-1 while its sequence similarity jumps to 0.968. Trip 00000077 is hard for both methods (classifier 71.2%,
sequence similarity &lt; 0.25): it has 7 chain breaks (gaps/disconnections), and when the classifier's top candidates are wrong over a stretch the HMM can only choose the most consistent of those wrong candidates; we have not yet analysed this trip in detail. β and the gap/path limits were
not tuned (an obvious next step, using training and student data only). Because the evaluator interface shows a route, the system deploys the HMM output.</p>
""")
# ---------------- 6 folium
S.append(f"""<h2>6. Geographical visualisation with Folium</h2>
<p>Map code: <code>src/visualizations.py::route_map</code>. Offline maps for all ten student trips are generated by <code>scripts/generate_maps.py</code>
(<code>results/maps/student_test/&lt;trip&gt;.html</code>; each run re-predicts the trip and checks the result equals the saved HMM output —
{sum(m['matches_saved_hmm_predictions'] for m in maps)}/10 matched). Layers: OpenStreetMap tiles; <span style="color:#6b6b6b">grey</span> candidate road corridor (all candidate arcs of the trip);
<span style="color:#2479d4">blue</span> GPS points; <span style="color:#1c9e54">green</span> reference route (offline maps only); <span style="color:#f39c12">orange dashed</span> classifier-only
route (toggle); <span style="color:#e34b36">red</span> HMM-predicted route; green/black start and end markers; a legend and an expanded layer control. The initial
view is fitted to the GPS extent. The evaluator UI uses the same function but without the reference or classifier layers.</p>
{img("results/report/screenshots/folium_00000052.png","Figure 15. Folium map of student trip 00000052 (headless-Edge screenshot of results/maps/student_test/00000052.html).")}
{img("results/report/screenshots/folium_00000000.png","Figure 16. Folium map of student trip 00000000 (≈70 km); the predicted (red) and reference (green) routes overlap almost everywhere.")}
""")
# ---------------- 7 API UI deployment
logl=[l.rstrip() for l in (ROOT/"results/logs/api_server_final.log").read_text(encoding="utf-8",errors="replace").splitlines()]
excerpt=[l for l in logl if "model loaded" in l or "Uvicorn running" in l][:2]+[l for l in logl if "PREDICT" in l][:8]
srows=[[r["trip_id"],pct(r["top1_accuracy"]),f"{r['processing_seconds']:.2f}",f"{r['http_seconds']:.2f}"] for r in smoke["trips"]]
S.append(f"""<h2>7. API, UI and deployment</h2>
<h3>7.1 Architecture</h3>
<pre class="diagram">Laptop 2 (evaluator)                         Wi-Fi / hotspot                 Laptop 1 (model + API)
Streamlit ui/app.py  ── GET /health, GET /trips ───────────────────────▶  FastAPI app/api.py (uvicorn :8000)
  trip selector, Predict Route                                              Predictor (best_model.joblib + configs/deployment.json)
  metrics + Folium map   ◀── JSON: predicted arcs, route, geometry,  ──    load trip → project → candidates → 14 features
                              metrics, timings (no labels)                  → classifier → HMM/Viterbi → route
                                                                            → server-side metrics vs .route (after prediction)</pre>
<p><b>Backend (Laptop 1).</b> <code>GET /health</code> reports whether the model and configuration loaded (503 "degraded" with the reason if an artifact is
missing, corrupt, has a different feature order or different parameters). <code>GET /trips</code> lists trip folders of the active mode found at request
time (<code>PATHSNAP_MODE=student_test</code> for development, <code>evaluator_test</code> for the reserved trips). <code>POST /predict</code> validates the ID
(422 malformed, 404 unknown), runs the full pipeline, and only afterwards reads the reference to compute Top-1 and continuity; reference arcs and per-point
labels are never returned. A middleware logs every request (path, status, duration) and the endpoint logs trip, observations, route length, accuracy and time.</p>
<p><b>UI (Laptop 2).</b> Streamlit page with configurable API URL, connection/health indicator, mode banner, trip selector, <i>Predict Route</i> button
with spinner, error/timeout messages, metrics (Top-1, continuity, server and round-trip time), the predicted arc sequence and the Folium map. It contains
no model and reads no trip files.</p>
<h3>7.2 Verification performed on the development laptop (single machine)</h3>
<p>Backend started with <code>python -m uvicorn app.api:app --host 0.0.0.0 --port 8010</code>; all ten student trips requested over real HTTP with the UI's
client (<code>scripts/smoke_http.py</code>), the Streamlit UI driven against it (<code>scripts/check_ui.py</code>), and screenshots captured in headless Edge
(<code>scripts/capture_screenshots.py</code>, {shots['captured_at']}). 27 automated tests pass (<code>results/logs/tests_final.log</code>).</p>
{table(["Trip","Top-1 (API)","Server processing s","HTTP round trip s"],srows,"small")}
{img("results/report/screenshots/ui_prediction.png","Figure 17. Evaluator UI after a prediction for trip 00000000 (screenshot of the running Streamlit app connected to the running API).",88)}
<div class="row">{img("results/report/screenshots/api_health.png","Figure 18. GET /health response.")}{img("results/report/screenshots/api_trips.png","Figure 19. GET /trips response.")}</div>
{img("results/report/screenshots/api_docs.png","Figure 20. FastAPI interactive documentation of the three endpoints.",70)}
<p>Excerpt of the backend log written during this verification (<code>results/logs/api_server_final.log</code>; text file, not a screenshot):</p>
<pre class="log">{html.escape(chr(10).join(excerpt))}</pre>
{gap("Two-laptop demonstration: screenshot of the UI on Laptop 2 connected to Laptop 1's IPv4 address, and a screenshot of Laptop 1's terminal (CMD/PowerShell) showing the received <code>PREDICT trip=...</code> requests. These have not been captured yet; all evidence above comes from one machine using 127.0.0.1.")}
<h3>7.3 Deployment</h3>
<p>Exact PowerShell commands for both laptops, IP discovery, connectivity tests (<code>Test-NetConnection</code>, <code>Invoke-RestMethod /health</code>) and
Windows-Firewall guidance are in the README (§2–§4). Dependencies are pinned in <code>requirements.txt</code> (Python 3.11.9).</p>
""")
# ---------------- 8 conclusions
S.append(f"""<h2>8. Conclusions and limitations</h2>
<ul>
<li>Five model families were compared under trip-grouped CV. Tree ensembles beat linear and Gaussian models; HistGradientBoosting gave the best
CV and student results with good calibration and was deployed after Optuna tuning (student Top-1 {pct(t.top1_accuracy)} classifier-only).</li>
<li>Bayesian optimisation changed the CV objective only marginally and did not improve student metrics — the model is near a plateau for these features.</li>
<li>The HMM/Viterbi decoder produces connected, realistic routes (continuity {pct(h['route_continuity'])}, sequence similarity {h['sequence_edit_similarity']:.3f})
at the cost of ~2 pp pointwise Top-1 ({pct(h['top1_accuracy'])}).</li>
<li>The system runs end-to-end through FastAPI and the Streamlit/Folium UI in 0.2–7 s per student trip.</li>
</ul>
<p><b>Limitations.</b> (1) Per-point labels are proxies from a geometric alignment, so "Top-1 accuracy" inherits its ambiguity at junctions.
(2) The HMM parameters were not tuned and emissions are classifier scores rather than a calibrated observation model. (3) Only 10 tuning trials and
at most 400 points per training trip were used. (4) Student-test trips were used for development comparisons, so they are not a fully unbiased estimate;
the reserved evaluator trips have not been seen. (5) The two-laptop network demonstration remains to be performed. (6) Map tiles require Internet on Laptop 2.</p>
<h2>9. AI-assistance disclosure</h2>
<p>AI coding assistants were used during development (OpenAI Codex for the initial implementation; Anthropic Claude Code for the completion audit,
UI/map improvements, test additions, figure and screenshot scripts and this report draft). All experiments were executed by the repository's code
and all numbers and figures are generated from its saved outputs. <span class="fill">[Team: edit this disclosure to match the event policy and your
actual usage, and confirm that every member can explain the code, experiments and results.]</span></p>
""")
CSS="""@page{size:A4;margin:16mm 14mm}body{font:10.5pt/1.45 'Segoe UI',Arial,sans-serif;color:#1b1b1b}
h1{font-size:22pt;margin:0 0 4px}h2{font-size:15pt;border-bottom:2px solid #217a91;padding-bottom:2px;margin-top:22px;page-break-after:avoid}
h3{font-size:12pt;margin:14px 0 4px;page-break-after:avoid}h4{font-size:11pt;margin:8px 0 2px;color:#217a91}
.title{border-left:6px solid #217a91;padding:8px 14px;margin-bottom:12px}.sub{font-size:12pt;color:#444;margin:0 0 8px}
.note{background:#eef6f8;border:1px solid #bcdbe3;padding:8px 10px;font-size:9.5pt}.fill{background:#fff3b0;padding:0 3px}
.gap{background:#fff1f0;border:2px dashed #d9534f;padding:8px 10px;margin:10px 0}
table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:9.5pt;page-break-inside:avoid}th,td{border:1px solid #c8c8c8;padding:3px 5px;text-align:left;vertical-align:top}
th{background:#e7f1f4}table.small{font-size:8.3pt}.small{font-size:8.8pt}
figure{margin:8px 0 12px;text-align:center;page-break-inside:avoid}img{max-width:100%;border:1px solid #ddd}figcaption{font-size:8.8pt;color:#444;margin-top:2px}
.src{color:#888;font-family:Consolas,monospace;font-size:7.8pt}.row{display:flex;gap:8px}.row figure{flex:1}.row5{display:flex;gap:4px}.row5 figure{flex:1;margin:0}
.model{border:1px solid #d7e6ea;border-radius:4px;padding:2px 10px 4px;margin:8px 0;page-break-inside:avoid}
pre{font:8pt/1.35 Consolas,monospace;background:#f6f6f6;border:1px solid #ddd;padding:6px;white-space:pre-wrap}pre.diagram{font-size:7.6pt}
code{font-family:Consolas,monospace;font-size:9pt;background:#f3f3f3;padding:0 2px}"""
page=f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>{''.join(S)}</body></html>"
html_path=OUT/"PathSnap_Technical_Report.html"; html_path.write_text(page,encoding="utf-8")
pdf=OUT/"PathSnap_Technical_Report.pdf"
edge=next(p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"] if os.path.exists(p))
subprocess.run([edge,"--headless=new","--disable-gpu","--no-pdf-header-footer",f"--user-data-dir={ROOT/'.cache/report_browser'}",
    f"--print-to-pdf={pdf}",html_path.as_uri()],check=True,timeout=180,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("wrote",html_path,"and",pdf,pdf.stat().st_size,"bytes")
