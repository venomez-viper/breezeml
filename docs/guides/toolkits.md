# Toolkits: track, anomaly, semi-supervised, native explain, and the CLI

Five more v1.7 additions, all on the 4 core dependencies: experiment
tracking without a server, anomaly detection with a consensus report,
self-training when most labels are missing, model-agnostic explanations
without SHAP, and a terminal command for the whole workflow.

## `track`: experiment tracking in one JSON file

Every serious ML project ends with "wait, which run was the good one?"
`breezeml.track` answers it with a plain JSON file: no server, no account,
no MLflow install. Runs live in `.breezeml/runs.json` next to your data,
human-readable and git-committable.

```python
model, report = breezeml.auto(df, "churn")
breezeml.track.log(model, report, name="baseline")

# ... days later ...
breezeml.track.leaderboard()          # every run, ranked by headline metric
best = breezeml.track.best()          # the top run, or None
breezeml.track.clear()                # wipe the log, returns the count
```

Each entry records the run name, timestamp, task, target, estimator, row
count, and the numeric parts of the report; AutoML runs also store the
best model, CV score, and time used. `leaderboard()` ranks score metrics
(accuracy, R2, F1) descending, then MAE runs ascending after them.

## `anomaly`: find the rows that do not belong

```python
from breezeml import anomaly

result = anomaly.isolation_forest(df)   # scores + flags per row
report = anomaly.compare(df)            # all 4 detectors, agreement report
print(report["n_unanimous"], report["unanimous_indices"])
```

Four detectors: `isolation_forest` (the strong tabular default),
`local_outlier_factor` (density-based), `one_class_svm` (boundary-based,
slower on large data), and `elliptic_envelope` (Gaussian assumption, fast).
Each returns per-row labels, scores, flagged indices, and the anomaly
rate. All run on the numeric view of your data with median imputation and
scaling, mirroring the supervised pipeline.

Anomaly detection has no ground truth, so `compare()` reports **agreement
between detectors** instead of an accuracy score: `consensus_votes` counts
how many detectors flagged each row, with `majority_indices` and
`unanimous_indices` shortcuts. Rows flagged by every detector deserve a
human look first.

## `semisupervised`: use the labels you do not have

Labeling is expensive. If you have 200 labeled rows and 5,000 unlabeled
ones, self-training can use the unlabeled data instead of throwing it away.
Rows where the target is NaN are the unlabeled pool:

```python
from breezeml import semisupervised

model, report = semisupervised.self_train(df, "label")
print(report["helped"], report["gain_over_baseline"])
```

The base classifier (random forest by default) trains on the labeled rows,
pseudo-labels the unlabeled rows it is confident about (probability >=
0.75, tunable via `threshold=`), retrains, and repeats. The holdout is
carved from *labeled* data only, and the report always includes
`supervised_baseline_accuracy`: the score using labeled rows alone. When
the unlabeled data did not help, `helped` is `False` and the printout says
"keep the supervised baseline" instead of hiding it. Needs at least 20
labeled rows.

## Native explainability: no SHAP required

`breezeml.explain` gained two model-agnostic tools that run on the core
dependencies (SHAP stays available via the `[explain]` extra):

```python
from breezeml import explain

rows = explain.permutation_importance(model, df, "target")
pd_curve = explain.partial_dependence(model, df, "target", feature="age")
```

`permutation_importance()` shuffles each feature and measures how much the
score drops: features the model truly relies on hurt the most when
scrambled. It works for any model, unlike tree-impurity importances.
`partial_dependence()` returns `{"grid", "average_prediction"}` for one
feature with everything else held constant; rising values mean the feature
pushes predictions up.

## The CLI: sound ML without opening Python

Installing breezeml puts a `breezeml` command on your PATH:

```bash
breezeml train data.csv --target churn            # auto model -> model.joblib
breezeml compare data.csv --target churn          # full leaderboard
breezeml automl data.csv --target churn --budget 120
breezeml audit data.csv --target churn            # exit 1 on critical findings
breezeml deploy model.joblib --out api/           # FastAPI + Docker directory
breezeml card model.joblib --out MODEL_CARD.md
breezeml zen                                      # the Zen of BreezeML
breezeml guide                                    # the garden path map
```

`train` accepts `--explain` to narrate every pipeline decision, and the
`audit` exit code makes it a natural CI gate before a scheduled retrain.

## When NOT to use these

- **track** is single-machine and file-based: no concurrent writers, no
  artifact storage, no UI. Teams with shared tracking servers should keep
  MLflow or W&B; this is for the other 95% of projects.
- **anomaly** detectors only see numeric columns (categoricals are
  dropped in the numeric prep), and `contamination=0.05` is an assumption,
  not a measurement: adjust it to your expected outlier rate.
- **semisupervised.self_train()** is classification-only and can amplify
  errors when the base model's confident predictions are confidently wrong.
  That is exactly why the baseline is in the report: trust `helped`.
- **permutation_importance()** is misleading when features are strongly
  correlated (shuffling one leaks through its twin), and
  `partial_dependence()` averages over the data, hiding interactions.
- **The CLI** covers the happy path on CSVs. Custom preprocessing, text
  embeddings, or time series still need the Python API.
