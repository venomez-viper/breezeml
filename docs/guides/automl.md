# BreezeAutoML: model search on a time budget

One call screens every built-in model, then spends the remaining budget
tuning the most promising candidates:

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model, report = breezeml.automl(df, "species", time_budget=60)

print(report["best_model"])       # e.g. "random_forest"
print(report["best_params"])      # winning hyperparameters
print(report["holdout"])          # honest 20% holdout metrics
```

## How the budget is spent

1. **Screening (~35%)**: every model gets a quick cross-validated score.
   Slow models are skipped once the screening window closes.
2. **Tuning (~65%)**: the top 3 tunable models get `RandomizedSearchCV`
   slices sized from their measured fit cost, so a fast model tries many
   configs and a slow one tries few instead of blowing the budget.
3. **Final holdout**: the winner is re-evaluated on a fresh stratified 20%
   split, so the reported metrics are not the (optimistic) search scores.

The budget is soft: a running fit is allowed to finish.

## Optuna backend (optional)

```bash
pip install breezeml[automl]
```

```python
model, report = breezeml.automl(df, "species", time_budget=120, backend="optuna")
```

Same search space, but Optuna's TPE sampler decides which configs to try.
Worth it on larger budgets; on small budgets random search is competitive.

## Everything downstream still works

The returned model is a normal `EasyModel`: `card()`, `export()`,
`deploy()`, and `explain_decisions()` all work. `model.meta["automl"]`
records the search: models screened, best CV score, params, time used.

## When NOT to use it

- Tiny datasets (< 200 rows): search scores are noise; train
  `classifiers.compare()` and pick by hand with cross-validation.
- When you already know the model family: `quick_tune()` spends your
  whole budget on the right grid instead of re-discovering it.
