# Significance: is this difference real, or is it noise?

`breezeml.significance` answers one question that every leaderboard begs and
almost no one asks: when model A scores 0.91 and model B scores 0.90, did A
actually win, or did you just get a lucky random seed? Two numbers on a
holdout are almost never distinguishable by eye. This module attaches a
p-value to the gap and, when the gap is not significant, tells you plainly to
keep the simpler, faster model.

```python
from breezeml import significance

# Two already-fitted classifiers on one shared holdout
result = significance.mcnemar(model_a, model_b, df, "target")
print(result["p_value"], result["significant"], result["verdict"])

# Two algorithms compared across the same CV folds
result = significance.paired_cv_ttest("random_forest", "logistic", df, "target")
print(result["mean_difference"], result["p_value"], result["verdict"])
```

Everything here (the error function, the chi-square tail, the Student t
two-sided p-value, the regularized incomplete beta) is implemented directly
in numpy. No scipy, no statsmodels. The four-dependency contract holds.

## `mcnemar()`: two trained models, one holdout

Give `mcnemar()` two already-fitted models (any object with `.predict`, such
as two `EasyModel`s) and it splits `df` once (stratified, `test_size=0.2`,
`random_state=42`, matching the core trainers' holdout), asks both models to
predict on that shared test slice, and builds the 2x2 agreement table.

McNemar looks only at the **discordant pairs**: the cases one model gets right
and the other gets wrong. The rows they both call correctly, or both blow, tell
you nothing about which is better. With the continuity correction the
statistic is `(|b - c| - 1)^2 / (b + c)`, where `b` and `c` are the two
discordant counts, and the p-value is the upper tail of a chi-square with 1
degree of freedom.

The returned dict carries `statistic`, `p_value`, `n_discordant`,
`only_a_correct`, `only_b_correct`, `significant` (True when `p < 0.05`), and a
plain-English `verdict`. When both models predict identically there are no
discordant pairs, so the statistic is 0 and the p-value is 1: no evidence of a
difference.

## `paired_cv_ttest()`: two algorithms, the same folds

Where `mcnemar()` compares two fitted models on one split,
`paired_cv_ttest()` compares two **algorithms** across cross-validation. Pass
two algorithm names (resolved through the same factories `classifiers` and
`regressors` use, so `"random_forest"`, `"logistic"`, `"gradient_boosting"`,
and the rest all work), and both are scored on the **same folds** from one
shared splitter with a fixed seed. That shared splitter is what makes the
fold-by-fold differences genuinely paired.

The paired t-statistic is `mean_diff / (std_diff / sqrt(n))` over the folds,
and the two-sided p-value comes from a Student t distribution with `cv - 1`
degrees of freedom. The task is auto-detected (classification scores on
accuracy with a `StratifiedKFold`, regression on R2 with a `KFold`); pass
`task="classification"` or `task="regression"` to force it, and `cv=` to
change the fold count (default 5).

```python
result = significance.paired_cv_ttest(
    "gradient_boosting", "logistic", df, "target", cv=10
)
# BreezeML Paired CV t-test - gradient_boosting vs logistic ...
#   Verdict: Not significant (p >= 0.05): the observed difference is likely
#   noise. Prefer the simpler or faster model ...
```

The dict carries `mean_score_a`, `mean_score_b`, `mean_difference`,
`t_statistic`, `p_value`, `dof`, `significant`, and the `verdict`. If every
per-fold difference is identical (zero variance, including two identical
models), there is no t evidence, so the statistic is 0 and the p-value is 1.

## The honest default

Both tests share one bias, and it is the whole point of the module: when the
result is **not** significant, the verdict says the observed difference is
likely noise and you should prefer the simpler or faster model, because the
extra complexity is not buying you a real accuracy gain. A p-value of 0.31 on
a fancier model is not a reason to ship the fancier model.

## When NOT to use it

- **`mcnemar()` needs a shared holdout and fitted classifiers.** It is a
  classification test built on a 2x2 agreement table; it has no meaning for
  regression, and both models must already be fitted on data that excludes the
  holdout it carves out.
- **A single test is not a multiple-comparison correction.** Run McNemar or a
  paired t-test across ten model pairs and roughly one in twenty "significant"
  results at `p < 0.05` is expected by chance. If you are screening many pairs,
  correct for it (Bonferroni or similar); this module reports one comparison at
  a time.
- **`p >= 0.05` is not proof the models are equal.** It means you do not have
  the evidence to call a winner, often because the holdout or fold count is too
  small. Absence of significance is absence of evidence, not evidence of
  absence.
- **Significant does not mean large.** With enough data a trivially small,
  operationally meaningless gap can clear `p < 0.05`. Read `mean_difference`
  (or the discordant counts) alongside the p-value and decide whether the
  effect is worth paying for.
