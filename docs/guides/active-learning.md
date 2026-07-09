# Active learning: spend your labeling budget where it helps

Labeling is expensive. If you have 50 labeled rows and 10,000 unlabeled ones,
you cannot afford to label them all, so which rows do you send to a human
next? Random picks waste effort on rows the model already understands. Active
learning ranks the unlabeled pool by how much a label would teach the model,
so you learn faster per label spent. `breezeml.active` gives you two functions:
`query()` to pick the next batch, and `simulate()` to check honestly whether
smart querying actually beats random on YOUR data.

## `query()`: pick the next rows to label

```python
import breezeml
from breezeml import active

model, _ = breezeml.classify(labeled_df, "label")
pick = active.query(model, unlabeled_df, k=20, strategy="uncertainty")
# -> label the rows in pick["indices"], add them to labeled_df, retrain
print(pick["indices"], pick["scores"])
```

`query(model, unlabeled_df, k, strategy)` scores every row in the unlabeled
pool and returns the `k` most informative ones (default `k=10`, capped at the
pool size). It returns a dict with `indices` (the top-k row labels of
`unlabeled_df`, ranked most to least informative), `scores`, `strategy`, and
`k`. Pass `target=` if the pool still carries the target column and you want it
dropped before scoring.

Four strategies, all computed from predicted class probabilities:

- **`uncertainty`** (default): least confident, `1 - probability of the top
  class`. The model is least sure about these rows.
- **`margin`**: smallest gap between the top two classes. The model is torn
  between two labels, which is often more useful than raw uncertainty.
- **`entropy`**: Shannon entropy of the full predicted distribution. The model
  spreads probability across many classes.
- **`random`**: a seeded random score. This is the honest baseline, not a real
  strategy, and it is what `simulate()` compares against.

## `simulate()`: did active learning actually beat random here?

Active learning is not free, and it does not always win. `simulate()` runs the
whole loop for you and pits it against a random baseline from the SAME starting
set, so you get an honest verdict before you build any labeling infrastructure.

```python
from breezeml import active, datasets

df = datasets.breast_cancer()
curve = active.simulate(df, "label", initial=20, budget=120, step=20)
print(curve["active_wins"], curve["area_between_curves"])
```

`simulate(df, target, initial, budget, step, strategy)` carves an honest 25%
holdout, starts from `initial` randomly labeled rows, then repeatedly trains,
queries the `step` most informative rows from the remaining pool, adds them,
and retrains, recording holdout accuracy at each budget point up to `budget`
total labels. The identical loop runs with `strategy="random"` from the same
starting set as the baseline. It prints the budget-vs-accuracy curve for both
alongside a verdict, and returns a dict with `budgets`, `active_accuracy`,
`random_accuracy` (equal-length lists), `area_between_curves` (positive means
active was ahead on average), and `active_wins`. When random is competitive the
report says so instead of hiding it.

## When NOT to use it

- **No `predict_proba`.** Every strategy scores rows from predicted class
  probabilities. A model that only emits hard labels (some SVMs, some custom
  estimators) cannot be used, and `query()` raises a clear `TypeError`.
- **Noisy data or outliers.** Uncertainty sampling chases the rows the model is
  least sure about, and pure noise or mislabeled outliers look exactly like
  that. You can spend your whole budget labeling garbage. Audit first (see
  [Honest ML](honest-ml.md)) and consider `margin` or `entropy` over raw
  `uncertainty`.
- **Labeling is cheap.** If a label costs a fraction of a cent, the machinery
  is not worth it: just label everything and train once.
- **Tiny pools.** With a few hundred unlabeled rows the overhead of iterative
  retraining rarely pays off over labeling them all. `simulate()` also warns
  when `initial` is under 5 rows, because the first model is then very unstable
  and the curve is noisy.
