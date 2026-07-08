# Survival: when will the event happen?

`breezeml.survival` answers the time-to-event question (when does the customer
churn, the part fail, the patient relapse?) without letting you make the
classic mistake that ruins it. Survival data has two columns per subject:

- **duration**: how long until the event happened, OR how long the subject was
  observed before it dropped out of view without the event happening.
- **event**: 1 if the event actually happened at `duration`, 0 if the subject
  was **censored** (the event has not been seen yet).

The censored rows are the whole point. Their duration is a **lower bound** on
the true time-to-event, not the true time. Drop the event column and regress on
duration directly and every censored row silently pulls your estimate down,
because you are treating "still alive at 30 days" as if it meant "died at
exactly 30 days". BreezeML calls that out loudly before it lets you trust a
number.

```python
from breezeml import survival

# The honesty check, on its own
survival.check_censoring(df, "duration", "event")

# Kaplan-Meier survival curve for one sample
km = survival.kaplan_meier(df, "duration", "event")
print(km["median_survival"], km["censoring_rate"])

# Per-group curves plus a log-rank test
groups = survival.groups_kaplan_meier(df, "duration", "event", "arm")
print(groups["logrank_p_value"], groups["significant"])
```

Everything here (the Kaplan-Meier estimator, Greenwood variance, and the
chi-square tail behind the log-rank test) is pure numpy. No new dependencies.

## `check_censoring()`: the loud warning

This is the honesty check for time-to-event data. It reports how many subjects
are censored and prints a warning explaining exactly why regressing on
`duration` while ignoring `event` is wrong: naive regression treats every
censored subject as if the event happened at their last-seen time, which
under-estimates true survival time and over-estimates the hazard. The dict
returns `n_subjects`, `n_censored`, `n_observed_events`, `censoring_rate`, and
`heavily_censored` (True when more than half the rows are censored).

## `kaplan_meier()`: the honest, assumption-light answer

The Kaplan-Meier estimator uses every subject's information up to the moment
they leave the risk set, censored or not, and never pretends a censored subject
experienced the event. At each distinct event time `t` the estimate updates as
`S(t) = S(t_prev) * (1 - d_i / n_i)`, where `d_i` is the number of events at
`t` and `n_i` is the number still at risk just before `t`. Greenwood's formula
supplies the standard error, and `ci_lower` / `ci_upper` are the 95% pointwise
bounds clipped to `[0, 1]`.

`kaplan_meier()` runs `check_censoring()` first (so you see the warning before
you trust the curve), then returns a dict with `timeline`, `survival`,
`survival_se`, `ci_lower`, `ci_upper`, `n_at_risk`, `n_events_at_t`,
`median_survival` (the first time `S <= 0.5`, or `None` if the curve never
crosses 0.5), `n_observed_events`, `n_censored`, `n_subjects`, and
`censoring_rate`.

## `groups_kaplan_meier()`: compare curves with a log-rank test

Fit a separate KM curve per level of a group column and test whether the curves
actually differ. With exactly **2 groups** the log-rank statistic and its 1-df
p-value are exact (built from the hypergeometric mean and variance of the risk
set). With **more than 2 groups** the p-value uses a documented chi-square
approximation on `k - 1` degrees of freedom.

```python
groups = survival.groups_kaplan_meier(df, "duration", "event", "treatment")
# Verdict: survival curves DIFFER significantly (p < 0.05).
```

The dict carries `groups` (each label mapped to its full KM result),
`logrank_statistic`, `logrank_p_value`, `logrank_dof`, `significant`, and the
`observed_events` / `expected_events` per group.

## When NOT to use it

- **Kaplan-Meier is unadjusted.** It describes one population (or a handful of
  groups) and answers "what fraction survive past time t?". It cannot give you
  the effect of a covariate while holding others fixed, a hazard ratio, or
  individual risk predictions.
- **The log-rank test is a hypothesis test, not an effect estimate.** It tells
  you whether curves differ; it does not tell you by how much, and it ignores
  covariates just as the curves do.
- **The moment you need covariate adjustment, reach for a regression model.**
  For "does treatment help AFTER controlling for age and stage?" you want a Cox
  proportional-hazards or a parametric AFT model. Those are not implemented
  here (to keep the core on pure numpy); install the optional `lifelines`
  package for `CoxPHFitter` / `WeibullAFTFitter`.
- **The event column must be 0/1 and durations non-negative.** The validator
  rejects anything else, on purpose: a mislabeled event column is the fastest
  way to a confidently wrong survival curve.
