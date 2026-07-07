# Architecture: The Four Breaths

BreezeML is intentionally large in capability and intentionally small in
what you must learn. The design follows a layered API (in the spirit of
[fastai's layered design](https://arxiv.org/abs/2002.04688)): each layer is
complete on its own, and nothing above it is required.

Run `breezeml.guide()` in any terminal for this map.

## Breath 1 - one line

```python
from breezeml import datasets, fit, predict
model = fit(datasets.iris(), "species")
```

`fit`, `predict`, `auto`, `from_csv`, `save`, `load`. Behind the single
call: type detection, median/mode imputation, scaling, one-hot encoding,
stratified splitting, a seeded estimator, and honest holdout metrics. A
beginner's first day never needs anything else.

## Breath 2 - understand and choose

`compare()` leaderboards (22 classifiers / 22 regressors),
`quick_tune()`, `detailed_report()`, `explain_decisions=True` narration,
and `card()` model cards. This is where BreezeML teaches: every automatic
decision can explain itself, computed from your data.

## Breath 3 - automate and ship

`automl()` (budget-aware search), `export()` (zero lock-in sklearn
codegen), `deploy()` (FastAPI + Docker), `drift.check()` and the live
`/drift` endpoint, `timeseries.forecast()` with the naive-baseline
honesty check. Since v1.7: `audit()` leakage detection, `fairness.report()`,
the `imbalance` toolkit, `blend()`, `track` experiment logging, and the
`breezeml` CLI. Production-aware, still 4 dependencies.

## Breath 4 - extend

`features`, `clustering`, `text`, `explain`, `plot` toolkits; the
`breezeml-mcp` server for AI agents; the zen garden (`zen()`, `haiku()`,
`fortune()`, `sensei()`).

## Design rules that hold every layer together

1. **The dependency contract.** Core imports only sklearn, pandas, numpy,
   joblib - enforced by a CI test that fails on a fifth. Optional powers
   live behind extras and lazy imports.
2. **Same shapes everywhere.** Training helpers return `(model, report)`;
   reports are plain dicts; leaderboards are lists of dicts sorted by the
   headline metric. Learn the shape once, use it everywhere.
3. **Honesty is not optional.** Stratified splits, fixed seeds, holdout
   metrics separated from search scores, naive baselines in forecasting,
   and warnings when results look too good.
4. **No lock-in, ever.** `export()` produces breezeml-free sklearn code;
   `deploy()` serves the raw sklearn pipeline. Leaving must always be easy -
   that is why people stay.
5. **Metadata rides with the model.** Core-API models carry `model.meta`
   (data profile, decisions, reference distributions), which powers cards,
   narration, drift checks, and export without recomputation.
