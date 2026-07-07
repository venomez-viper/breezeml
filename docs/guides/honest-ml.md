# Honest ML: audit, fairness, imbalance, blend

Four v1.7 tools with one job: stop your model from lying to you. Audit the
data before training, audit the model for group disparities after, handle
imbalanced targets with thresholds instead of accuracy theater, and ensemble
only when the ensemble actually earns it.

## `audit()`: catch the mistakes before they reach production

```python
import breezeml
from breezeml import datasets

df = datasets.breast_cancer()
result = breezeml.audit(df, "label")
print(result["ok"], result["summary"])
```

`audit()` checks for ID-like columns (models memorize IDs instead of
learning), constant columns, exact duplicate rows, contradictory labels
(identical features, different targets), high-cardinality categoricals,
columns that are more than half missing, and severe class imbalance. The
big one is **target leakage**: every feature is probed by training a tiny
depth-3 decision tree on that feature *alone*. A single feature that
predicts the target with a cross-validated score of 0.98 or better almost
always means the target leaked in (a post-outcome column, an encoded copy
of the label, a proxy). Leakage and ID columns are `critical` findings and
flip `ok` to `False`; everything else is a `warning`.

There is also a split checker:

```python
breezeml.contamination(train_df, test_df)
# CONTAMINATION: 12 row(s) appear in both train and test (2.4% of the test set).
```

Rows shared between train and test silently inflate every metric you
report. The CLI version, `breezeml audit data.csv --target churn`, exits
with code 1 on critical findings, so you can gate a pipeline on it.

## `fairness.report()`: per-group performance, measured not assumed

Every BreezeML model card warns "no fairness audit was performed." This is
how you perform one:

```python
model, report = breezeml.auto(df, "approved")
result = breezeml.fairness.report(model, holdout_df, sensitive="gender")
print(result["demographic_parity_ratio"], result["passes_four_fifths"])
```

For classification it reports, per group: size, accuracy, F1, selection
rate (share predicted positive), TPR, and FPR, plus the demographic parity
ratio and a four-fifths (80%) rule verdict, and the largest TPR and FPR
gaps between groups. For regression: per-group MAE and mean error, so you
see which direction the model is biased for each group. Groups under 10
rows get called out as noise, not evidence. Use a holdout set, never
training data. The sensitive column does not need to be a model feature;
it only has to exist in the DataFrame.

A passing report is evidence, not absolution: parity on one attribute does
not make a model fair. But it beats not looking.

## `imbalance`: stop judging rare-event models by accuracy

A 99:1 fraud dataset gives any model 99% accuracy for predicting "no fraud"
every time. The imbalance toolkit is the honest path, on the 4 core
dependencies only (no SMOTE clones, no synthetic data):

```python
from breezeml import imbalance

imbalance.summary(df["fraud"])                       # how bad is it?
model, report = breezeml.classify(df, "fraud", balanced=True)  # class weights
thr = imbalance.tune_threshold(model, df, "fraud")   # stop using 0.5
cal, cal_report = imbalance.calibrate(model, df, "fraud")  # honest probabilities
cost = imbalance.cost_report(model, df, "fraud", fp_cost=1, fn_cost=25)
```

- `summary()` reports the imbalance ratio, minority share, and a severity
  verdict (mild / moderate / severe).
- `balanced=True` on `classify()` / `auto()` trains with class weights.
- `tune_threshold()` finds the decision threshold that maximizes minority
  F1 on a held-out 25% split, and shows F1 at 0.5 versus F1 at the best
  threshold. Apply it with `imbalance.predict_with_threshold(model, X, thr["best_threshold"])`.
- `calibrate()` wraps the model in isotonic (or sigmoid) calibration and
  reports the Brier score before and after; if calibration did not help,
  it says so.
- `cost_report()` picks the threshold that minimizes *your* costs: tell it
  a missed fraud (FN) costs 25x a false alarm (FP) and it sweeps 99
  thresholds to find the cheapest one, reporting savings versus 0.5.

All threshold helpers are binary-only and require `predict_proba`.

## `blend()`: ensembles that admit when they lose

```python
model, report = breezeml.blend(df, "target")                  # soft-voting top 3
model, report = breezeml.blend(df, "target", method="stack")  # stacking
```

`blend()` runs `compare()` under the hood, takes the top `top_k` models
(default 3), and combines them with soft voting or a stacking meta-learner.
The report always contains `best_single_model`, `best_single_score`,
`blend_score`, and `beats_best_single`. When the blend loses, it prints:

```text
Honest call: keep the single model. Simpler wins ties.
```

The result is a normal `EasyModel`: `save()`, `card()`, and `deploy()` all
work on it.

## When NOT to use these

- **audit()** probes each feature independently, so it cannot catch leakage
  spread across a *combination* of features, and 0.98 single-feature scores
  can be legitimate on genuinely easy targets. Treat findings as leads to
  investigate, not verdicts.
- **fairness.report()** measures one attribute at a time and needs enough
  rows per group; it does not detect intersectional disparities (for
  example, disparities that only appear for one gender within one age band).
- **imbalance** threshold and calibration helpers are for binary targets
  with probability outputs. For multi-class imbalance, use `balanced=True`
  and judge by macro F1.
- **blend()** costs `top_k` extra model fits and slower inference for a
  usually small gain. If `beats_best_single` is `False`, believe it: ship
  the single model.
