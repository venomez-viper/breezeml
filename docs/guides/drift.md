# Drift monitoring: know when your model goes stale

Every model trained with the core API (v1.3+) stores compact reference
distributions of its training data. Comparing new data against them takes
one call:

```python
import breezeml
from breezeml import drift

model, report = breezeml.auto(df, "churn")

result = drift.check(model, new_df)      # or model.check_drift(new_df)
print(result["summary"])
# "2 column(s) drifted (PSI >= 0.25): ['monthly_charges', 'plan_type']"

if result["drifted"]:
    ...  # retrain, alert, investigate
```

## What it detects

| Signal | How |
|---|---|
| Numeric distribution shift | PSI over training decile bins |
| Values outside training range | `share_outside_training_range` per column |
| New categories | pooled unseen-category share + PSI |
| Vanished columns | `missing_columns` in the report |
| Missing-rate spikes | train vs now missing rate per column |

PSI conventions: < 0.10 stable, 0.10-0.25 moderate (`warning`),
>= 0.25 `drift`. Threshold configurable via `check(..., threshold=)`.

## The /drift endpoint

`deploy()` bakes the reference distributions into the serving directory
(`reference.json`). The generated FastAPI app buffers the last 1000
prediction requests in memory and serves a live report:

```bash
curl http://localhost:8000/drift
# {"drifted": false, "columns": {"age": {"psi": 0.03, "status": "ok"}, ...}}
```

Needs at least 30 buffered predictions before it reports. The endpoint,
like the rest of the app, has zero breezeml dependency.

## When NOT to rely on it

- PSI sees marginal distributions only: a shift in the *relationship*
  between features (concept drift) with unchanged marginals passes
  undetected. Track live accuracy against ground truth when you have it.
- The in-app buffer resets on restart and lives per-process; behind a
  multi-worker server, each worker sees only its own traffic. For serious
  monitoring, log requests and run `drift.check()` on the full log.
