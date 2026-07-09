# Causal inference: separate correlation from causation

`breezeml.causal` estimates the EFFECT of a treatment, not just a correlation.
Every row is a unit (a customer, a patient, a user), one binary column says
whether the unit got the treatment (`1` = treated, `0` = control), and one
column is the outcome you care about. The causal question is: how much did the
treatment CHANGE the outcome?

The Average Treatment Effect is `E[outcome | treated] - E[outcome | control]`,
but that raw difference of means is honest ONLY when treatment was assigned at
random (a proper A/B test). In observational data, whoever ended up treated is
usually different to begin with: sicker patients get the drug, engaged users
get the email, bigger accounts get the discount. Those pre-existing differences
are CONFOUNDERS, and they contaminate the naive difference so it measures
"treated vs control units" instead of "the effect of treating the same unit".
This module always computes the naive difference AND an adjusted estimate and
puts them side by side. A large gap between them is your confounding alarm.

```python
from breezeml import causal

res     = causal.estimate_ate(df, "treated", "spend", method="t_learner")
pair, r = causal.uplift(df, "treated", "spend")
imb     = causal.check_confounding(df, "treated", "spend")
```

Every other column in `df` is used as a covariate to adjust for. The treatment
column must be binary 0/1.

## `estimate_ate()`: the adjusted effect, next to the confounded baseline

```python
res = causal.estimate_ate(df, "treated", "spend", method="t_learner")
print(res["ate"], res["naive_ate"], res["note"])
```

`estimate_ate(df, treatment, outcome, method=...)` returns a dict with `ate`,
`naive_ate`, `method`, `propensity_range` (ipw only, else None), `n_treated`,
`n_control`, and a plain-English `note`. Three methods:

- **`naive`**: the raw difference of group means. Confounded unless treatment
  was randomized. It is always computed and reported regardless of which method
  you choose.
- **`t_learner`** (default): fit one outcome model on the treated and one on
  the control, predict both potential outcomes for everyone, and average the
  difference. The default outcome model is a random forest (no linearity
  assumption).
- **`ipw`**: inverse-propensity weighting. Fit `P(treated | covariates)`,
  reweight so treated and control look alike, then take the difference. The
  report includes the `propensity_range` and warns when propensities hit the
  clip bounds (weak overlap, where IPW extrapolates).

When the adjusted ATE differs from the naive ATE by more than about 25%, the
`note` calls that gap the footprint of confounding. Binary outcomes are treated
as 0/1, so the effect is a difference in probability (a risk difference).

## `uplift()`: who benefits most (per-row CATE)

```python
pair, report = causal.uplift(df, "treated", "spend")
effects = pair.predict_uplift(new_X)   # per-row treatment effect (CATE)
print(report["top_decile_uplift"], report["overall_ate"])
```

Uplift (a.k.a. CATE) modeling ranks units by how much the treatment is
predicted to help THEM specifically, so you can treat the responders and spare
everyone else. `uplift()` fits a T-learner and returns a `(TLearnerPair, dict)`
pair. The `TLearnerPair` exposes `predict_uplift(X)`, the per-row treated-model
prediction minus control-model prediction. The report carries
`top_decile_uplift` (actual outcome lift among the top-ranked units),
`overall_ate`, `predicted_top_uplift`, `k`, `n_top`, and a `note` that says
whether the top-ranked units really do respond more than average, and reminds
you to validate the ranking on held-out (ideally randomized) data.

## `check_confounding()`: is this data even randomized?

```python
imb = causal.check_confounding(df, "treated", "spend")
print(imb["looks_randomized"], imb["max_abs_smd"], imb["imbalanced"])
```

For each covariate, `check_confounding()` reports the standardized mean
difference (SMD) between treated and control:
`SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)`,
a scale-free measure of how differently the two groups look on that covariate.
`|SMD| > 0.1` is the conventional flag for meaningful imbalance, evidence that
treatment was NOT randomly assigned and that a naive difference of means will
be biased. The dict returns `smd` (per covariate), `imbalanced` (the flagged
covariates, worst first), `max_abs_smd`, `n_imbalanced`, `looks_randomized`,
and a `note`. Run this before you trust any effect estimate.

## What adjustment can and cannot do

The T-learner and IPW adjust for the confounders you MEASURED and put in the
covariate columns. They lean on the untestable no-unmeasured-confounders
assumption (ignorability): that once you condition on the observed covariates,
treated and control units are comparable. You cannot test this with the data.
If an important confounder is missing (unrecorded severity, salesperson skill,
motivation), every number is biased and no amount of modeling fixes it. Both
`estimate_ate()` and `uplift()` print this honesty warning loudly.

## When NOT to use it

- **This is NOT a randomized experiment.** If you can run a proper A/B test, do
  that instead; a randomized difference of means beats any adjustment here.
- **If you suspect an unmeasured confounder**, treat every estimate as a
  correlation with better manners, not a causal effect.
- **Tiny samples or near-total overlap failure** (treated and control never
  look alike) make the adjusted estimates extrapolate wildly. Check
  `check_confounding()` and the propensity range first.
- **Interference** (one unit's treatment affects another's outcome) breaks the
  whole framework; none of this applies to viral or network effects.
