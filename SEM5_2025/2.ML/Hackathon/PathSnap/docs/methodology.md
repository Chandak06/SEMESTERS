# Implementation and metric definitions

This is implementation documentation for Deliverable 1, not the Technical Report.

The five supplied PDFs were read before implementation. Source archives are untouched. Source hashes and extracted-file hashes are recorded locally. No evaluator data were available or used. No Git operations are part of the workflow.

## Data and labels

Files are whitespace separated with no headers; road node and arc IDs are original zero-based row indices. Loading rejects malformed road networks, noninteger/out-of-range IDs and invalid reference IDs. Invalid GPS rows remain in place with an unmatched prediction (-1), rather than shifting correspondence. Timestamps are never silently sorted. Duplicate/nonpositive time steps and gaps over 30 seconds restart the decoder.

Features use a local equirectangular projection with radius 6371008.8 m and median network coordinates as origin. This works without native GIS binaries; an all-network audit compares segment lengths with spherical haversine distances. Projection errors and spherical-Earth approximation remain limitations, especially for unusually broad networks. The original intended pyproj implementation could not run under this machine's application-control policy; that failed execution is retained in the logs.

Reference labels are **proxies**, not independently verified observation annotations. A dynamic program aligns observations to chronological occurrences in the reference arc sequence, minimizing squared segment distance (scaled by 20 m and capped at 2500 per observation) plus a heading disagreement term. It can stay on a route occurrence or move forward; it cannot move backward. Backtracking yields the per-observation reference arc. It is independent of generated candidates. Routes may have incomplete coverage or discrepancies, and intersections/repeated roads can be ambiguous. No reference label is inserted into a candidate set. No row-by-row track/route matching is used.

Only training observations with alignment residual at most 80 m are eligible for fitting. Up to 400 evenly distributed eligible observations from **every one of the 80 trips** are used (31,903 observations, 255,224 candidate rows in this run). Saved sample indices reproduce this choice and reduce training cost and long-trip dominance. Validation uses the same eligible training pool. All valid student-test observations, including difficult alignments and candidate misses, are scored. Therefore CV and student-test metrics differ in both partition and quality filtering. Student-testing is a development set, not a final unbiased evaluation.

## Candidates and features

Eight nearest directed road segments are found per GPS observation. A cKDTree over segment midpoints supplies an initial bound. Searching all midpoints within that bound plus maximum segment half-length and refining with exact point-to-segment distances guarantees inclusion of nearer segments. Ties use original arc ID. There is no label-aware candidate expansion; far candidates remain visible and are flagged through distance features. Invalid observations have no candidates. Very long road segments can reduce index efficiency.

Fourteen features encode segment distance, endpoint distances, road length, directed heading difference, heading reliability, time difference, clipped speed, endpoint in/out degree, distance rank, projection fraction, distance margin and an outlier flag. Five-observation movement windows stabilize heading. Stationary movement, implausible speeds over 60 m/s and long gaps disable heading. This uses future GPS within a trip and is suitable for whole-trip reconstruction, not claimed to be causal streaming. Road IDs, absolute GPS positions and reference-route properties are excluded from predictors. Feature summaries are saved from real data. Distances distinguish nearby roads; headings distinguish reverse arcs; degrees and temporal signals provide ambiguity/noise context. No feature ablation claim is made.

## Models, folds and selection

Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest and HistGradientBoosting are trained. Gaussian Naive Bayes provides an inexpensive generative family baseline in place of the optional KNN suggestion. Its correlated-feature independence assumption can hurt probability quality. Linear and Gaussian models include StandardScaler fitted inside each fold and serialized in their pipeline. Trees need no fitted scaling. HistGradientBoosting disables random-row early stopping to preserve trip grouping.

Three GroupKFold splits keep all observations of each trip together. Fold trip membership, parameters, objectives, timings and full-fit models are saved. Primary model selection uses mean group-CV observation Top-1; models within 0.005 of the best are compared by lowest CV candidate log loss, then measured inference time. This considers matching quality, probability quality and speed; the models are not chosen by training accuracy.

Optuna TPE searches the selected family. Ten trials include three startup samples and seven TPE trials. The objective is mean fold (Top-1 - 0.02 * log loss). JSON is written after every trial; CSV, best configuration and full metrics are also retained. Search ranges are executable in tune_model.py. The original model is retained for deployment if its CV objective beats the tuned result. A fresh invocation starts a new reproducible study; archive current results before rerunning. Native SQL storage was unavailable, so the supported in-memory Optuna storage is used with explicit file persistence.

## Metrics

Candidate metrics flatten valid candidate pairs; accuracy, positive-class precision/recall, **binary-class macro F1**, 0.5 threshold confusion matrices and clipped binary log loss are reported. This macro F1 is not a macro average over thousands of road IDs.

Observation Top-1 accuracy chooses the maximum candidate probability and compares original arc IDs with aligned reference proxies. Missing reference candidates count as failures. Candidate miss rate is the fraction of valid observations whose aligned reference is absent. Route continuity is the fraction of adjacent matched observations on the same arc or directly connected directed arcs; it does not credit a multi-arc jump unless those bridges are separately reconstructed. Aggregate continuity weights transition-pair counts; aggregate accuracy weights observations.

Sequence edit similarity is 1 minus Levenshtein distance divided by the longer sequence length. Classifier sequences collapse consecutive duplicate predictions. HMM sequences additionally include actual shortest-path bridge arcs. Both compare against the supplied chronological route. This is a route-reconstruction comparison, not only a decoder-state comparison; per-trip observation metrics provide the common pointwise measure. Exact sequence match is also recorded. No artificial epoch curves are generated.

## Decoder and runtime separation

HMM states are current candidate original arc IDs. Clipped classifier probabilities normalized within an observation supply emission scores. Transition scores use the difference between directed shortest-path distance between projected candidate positions and GPS movement, scaled by 30 m + 0.1 times GPS movement. Available transitions are row-normalized in log space. Directed Dijkstra preserves arc IDs, caches paths, and limits searches to 3000 m. Transitions above max(150 m, 60 m/s times dt) are rejected. Viterbi uses log-space dynamic programming and explicit backtracking.

No-candidate rows, disconnected candidate sets, nonpositive time and gaps over 30 seconds break the chain. Each new block starts from emissions. Missing rows output -1. Route assembly inserts real directed shortest-path bridges where available and records boundaries elsewhere; maps do not draw invented straight bridges across disconnected roads. Projection-fraction reversal can require a real loop in the directed graph.

The inference module never loads a reference route. The API completes prediction first, then loads reference data solely for server-side metrics. It never returns labels or reference geometry. Evaluator mode only lists actual supplied directories. Student mode is prominently labelled. Trip IDs are allowlisted and path traversal is rejected; symlink escapes are checked by resolved path. API calls log path, status, timings, trip and aggregate accuracy.

Folium offline maps may include permitted reference layers. UI maps contain only returned GPS and predicted geometry. Folium's map tiles and JavaScript normally require Internet access; API communication itself is local. Two-laptop Wi-Fi/firewall verification and reserved evaluator evaluation remain pending.

