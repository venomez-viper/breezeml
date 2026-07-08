"""
BreezeML survival analysis: time-to-event modeling that respects censoring.

Survival data has two columns per subject:

    duration : how long until the event (churn, failure, death) happened,
               OR how long the subject was observed before it fell out of
               view without the event ever happening.
    event    : 1 if the event actually happened at ``duration``,
               0 if the subject was CENSORED (event not seen yet).

The censored rows are the whole point. Their duration is a LOWER BOUND on
the true time-to-event, not the true time. If you drop the event column and
regress on duration directly, every censored row silently pulls your
estimate DOWN, because you are treating "still alive at 30 days" as if it
meant "died at exactly 30 days". That is the classic censoring-blindness
mistake, and BreezeML calls it out loudly (see ``check_censoring``).

The Kaplan-Meier estimator is the honest, assumption-light answer: it
uses every subject's information up to the moment they leave the risk set,
censored or not, and never pretends a censored subject died.

    from breezeml import survival

    km = survival.kaplan_meier(df, "duration", "event")
    groups = survival.groups_kaplan_meier(df, "duration", "event", "arm")

When NOT to use / when you need more
------------------------------------
Kaplan-Meier is UNADJUSTED. It describes one population (or a handful of
groups) and answers "what fraction survive past time t?". It cannot tell
you the effect of a covariate while holding others fixed, it cannot give
you a hazard ratio, and it cannot produce individual risk predictions.

The moment you want covariate adjustment ("does treatment help AFTER
controlling for age and stage?") you need a regression model for survival:
a Cox proportional-hazards model or a parametric AFT model. Those are not
implemented here to keep the core on pure numpy; install the optional
``lifelines`` package for ``CoxPHFitter`` / ``WeibullAFTFitter``. The
log-rank test in ``groups_kaplan_meier`` is a hypothesis test, not an
adjusted effect estimate, and it too ignores covariates.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._validation import check_df_target

__all__ = ["kaplan_meier", "groups_kaplan_meier", "check_censoring"]

Z_95 = 1.959963984540054  # 1.96, the 97.5th percentile of the standard normal


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def _extract_survival(df: pd.DataFrame, duration_col: str, event_col: str):
    """Validate columns and return clean (durations, events) float/int arrays."""
    check_df_target(df, duration_col)
    check_df_target(df, event_col)
    if duration_col == event_col:
        raise ValueError("duration_col and event_col must be different columns.")

    frame = df[[duration_col, event_col]].dropna()
    if len(frame) == 0:
        raise ValueError("No non-missing rows in duration/event columns.")

    durations = frame[duration_col].to_numpy(dtype=float)
    if np.any(durations < 0):
        raise ValueError(f"Column '{duration_col}' contains negative durations.")

    events_raw = frame[event_col].to_numpy()
    events = events_raw.astype(float)
    unique_events = set(np.unique(events).tolist())
    if not unique_events.issubset({0.0, 1.0}):
        raise ValueError(
            f"Column '{event_col}' must be 0/1 (0 = censored, 1 = event); "
            f"found values {sorted(unique_events)}."
        )
    return durations, events.astype(int)


# --------------------------------------------------------------------------- #
# core Kaplan-Meier (pure numpy)
# --------------------------------------------------------------------------- #
def _km_curve(durations: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate with Greenwood variance. Pure numpy.

    Returns (timeline, survival, se, n_at_risk, n_events) as numpy arrays,
    where timeline starts at 0 with survival 1.0 and then lists every
    distinct EVENT time in ascending order.
    """
    n_total = len(durations)
    event_times = np.unique(durations[events == 1])

    # start the curve at time 0, everyone alive
    timeline = [0.0]
    survival = [1.0]
    se = [0.0]
    at_risk_out = [n_total]
    events_out = [0]

    surv = 1.0
    greenwood_sum = 0.0  # running sum of d_i / (n_i * (n_i - d_i))
    for t in event_times:
        n_i = int(np.sum(durations >= t))          # at risk just before t
        d_i = int(np.sum((durations == t) & (events == 1)))  # events at t
        if n_i == 0:
            continue
        surv = surv * (1.0 - d_i / n_i)
        # Greenwood term is undefined when n_i == d_i (survival hits 0);
        # skip that term so we do not divide by zero.
        if n_i - d_i > 0:
            greenwood_sum += d_i / (n_i * (n_i - d_i))
        se_t = surv * math.sqrt(greenwood_sum)

        timeline.append(float(t))
        survival.append(float(surv))
        se.append(float(se_t))
        at_risk_out.append(n_i)
        events_out.append(d_i)

    return (
        np.asarray(timeline, dtype=float),
        np.asarray(survival, dtype=float),
        np.asarray(se, dtype=float),
        np.asarray(at_risk_out, dtype=int),
        np.asarray(events_out, dtype=int),
    )


def _median_survival(timeline: np.ndarray, survival: np.ndarray):
    """First time at which the survival estimate drops to 0.5 or below."""
    below = np.where(survival <= 0.5)[0]
    if len(below) == 0:
        return None
    return float(timeline[below[0]])


def _summarize_km(durations: np.ndarray, events: np.ndarray) -> dict:
    """Build the full KM result dict for one sample (no printing)."""
    timeline, survival, se, at_risk, n_events = _km_curve(durations, events)

    lower = np.clip(survival - Z_95 * se, 0.0, 1.0)
    upper = np.clip(survival + Z_95 * se, 0.0, 1.0)

    n_obs_events = int(np.sum(events == 1))
    n_censored = int(np.sum(events == 0))
    n_total = len(events)

    return {
        "timeline": timeline.tolist(),
        "survival": survival.tolist(),
        "survival_se": se.tolist(),
        "ci_lower": lower.tolist(),
        "ci_upper": upper.tolist(),
        "n_at_risk": at_risk.tolist(),
        "n_events_at_t": n_events.tolist(),
        "median_survival": _median_survival(timeline, survival),
        "n_observed_events": n_obs_events,
        "n_censored": n_censored,
        "n_subjects": n_total,
        "censoring_rate": float(n_censored / n_total) if n_total else 0.0,
    }


# --------------------------------------------------------------------------- #
# honesty helper
# --------------------------------------------------------------------------- #
def check_censoring(df: pd.DataFrame, duration_col: str, event_col: str,
                    show: bool = True) -> dict:
    """Report the censoring rate and warn against naive duration regression.

    This is the honesty check for time-to-event data. It prints a LOUD
    warning explaining why regressing on ``duration`` while ignoring
    ``event`` is wrong: censored rows have a duration that is only a lower
    bound on the true time-to-event, so treating them as completed events
    systematically UNDER-estimates survival time (and over-estimates the
    hazard).
    """
    durations, events = _extract_survival(df, duration_col, event_col)
    n_total = len(events)
    n_censored = int(np.sum(events == 0))
    rate = float(n_censored / n_total) if n_total else 0.0

    result = {
        "n_subjects": n_total,
        "n_censored": n_censored,
        "n_observed_events": int(np.sum(events == 1)),
        "censoring_rate": rate,
        "heavily_censored": bool(rate > 0.5),
    }

    if show:
        print("\nBreezeML Censoring Check")
        print("-" * 64)
        print(
            f"  {n_censored:,} of {n_total:,} subjects are CENSORED "
            f"({rate:.1%}). Their '{duration_col}' is a lower bound on the"
        )
        print("  true time-to-event, NOT the event time itself.")
        print(
            f"  WARNING: do NOT regress on '{duration_col}' while ignoring "
            f"'{event_col}'."
        )
        print(
            "  Naive regression treats every censored subject as if the event"
        )
        print(
            "  happened at their last-seen time, which UNDER-estimates true"
        )
        print(
            "  survival time and OVER-estimates the hazard. Use Kaplan-Meier"
        )
        print(
            "  (or a Cox / AFT model) so censored rows count only up to when"
        )
        print("  they left the risk set.")
        print("-" * 64)

    return result


# --------------------------------------------------------------------------- #
# public: single-sample Kaplan-Meier
# --------------------------------------------------------------------------- #
def kaplan_meier(df: pd.DataFrame, duration_col: str, event_col: str,
                 show: bool = True) -> dict:
    """Kaplan-Meier survival curve for a single sample (pure numpy).

    At each distinct event time ``t`` the estimate updates as
    ``S(t) = S(t_prev) * (1 - d_i / n_i)`` where ``d_i`` is the number of
    events at ``t`` and ``n_i`` is the number still at risk just before
    ``t``. Greenwood's formula supplies the standard error, and the
    returned ``ci_lower`` / ``ci_upper`` are the 95% pointwise bounds
    ``S(t) +/- 1.96 * SE`` clipped to ``[0, 1]``.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the duration and event columns.
    duration_col : str
        Column of times-to-event or times-to-censoring (non-negative).
    event_col : str
        Column of 0/1 indicators (1 = event observed, 0 = censored).
    show : bool
        Print the censoring check and a curve summary (default True).

    Returns
    -------
    dict
        ``timeline``, ``survival``, ``survival_se``, ``ci_lower``,
        ``ci_upper``, ``n_at_risk``, ``n_events_at_t``, ``median_survival``
        (first time S <= 0.5, or None), ``n_observed_events``,
        ``n_censored``, ``n_subjects`` and ``censoring_rate``.
    """
    durations, events = _extract_survival(df, duration_col, event_col)

    # honesty first: shout about censoring before anyone trusts a number.
    if show:
        check_censoring(df, duration_col, event_col, show=True)

    result = _summarize_km(durations, events)

    if show:
        med = result["median_survival"]
        med_str = "not reached (curve never crosses 0.5)" if med is None else f"{med:g}"
        print(f"\nBreezeML Kaplan-Meier - '{duration_col}' / '{event_col}'")
        print("-" * 64)
        print(
            f"  subjects: {result['n_subjects']:,}   events: "
            f"{result['n_observed_events']:,}   censored: "
            f"{result['n_censored']:,} ({result['censoring_rate']:.1%})"
        )
        print(f"  median survival: {med_str}")
        print(f"  distinct event times: {len(result['timeline']) - 1}")
        if result["survival"]:
            print(
                f"  final survival estimate: {result['survival'][-1]:.4f} "
                f"at t = {result['timeline'][-1]:g}"
            )
        print("-" * 64)

    return result


# --------------------------------------------------------------------------- #
# chi-square survival function (no scipy)
# --------------------------------------------------------------------------- #
def _gammaincc_reg(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).

    Numerical Recipes style: a power series for the lower part when
    ``x < a + 1`` (then complement), a continued fraction for the upper
    part otherwise. Accurate to roughly 1e-10 for the a, x values used by
    the chi-square tail here.
    """
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0

    gln = math.lgamma(a)
    if x < a + 1.0:
        # lower series P(a, x), then Q = 1 - P
        ap = a
        total = 1.0 / a
        term = total
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return 1.0 - p

    # upper continued fraction Q(a, x) directly (Lentz's method)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return q


def _chi2_sf(stat: float, dof: int) -> float:
    """Upper-tail (survival) probability of a chi-square with ``dof`` df.

    For 1 df this is exact via ``erfc(sqrt(x/2))``. For general df it uses
    the regularized upper incomplete gamma ``Q(df/2, x/2)`` above.
    """
    if stat <= 0.0:
        return 1.0
    if dof == 1:
        return math.erfc(math.sqrt(stat / 2.0))
    return _gammaincc_reg(dof / 2.0, stat / 2.0)


# --------------------------------------------------------------------------- #
# log-rank test (pure numpy)
# --------------------------------------------------------------------------- #
def _logrank(durations: np.ndarray, events: np.ndarray, group_labels: np.ndarray,
             groups: list):
    """Log-rank chi-square across groups.

    At each distinct event time the observed events per group are compared
    to the events expected under the null (equal hazard), using the
    hypergeometric mean and variance for the risk set. For 2 groups the
    statistic ``(O1 - E1)^2 / V1`` is EXACT with 1 df. For more than 2
    groups this returns the ``sum_g (O_g - E_g)^2 / E_g`` approximation on
    ``k - 1`` df, which is a well-known but approximate form.
    """
    n_groups = len(groups)
    event_times = np.unique(durations[events == 1])

    observed = np.zeros(n_groups)
    expected = np.zeros(n_groups)
    var_g1 = 0.0  # variance accumulator for group 0 (used in the 2-group case)

    for t in event_times:
        at_risk_mask = durations >= t
        n_total = int(np.sum(at_risk_mask))
        d_total = int(np.sum((durations == t) & (events == 1)))
        if n_total == 0 or d_total == 0:
            continue
        for gi, g in enumerate(groups):
            in_g = group_labels == g
            n_g = int(np.sum(at_risk_mask & in_g))
            d_g = int(np.sum((durations == t) & (events == 1) & in_g))
            observed[gi] += d_g
            expected[gi] += d_total * n_g / n_total
        # variance of the events in group 0 (hypergeometric)
        n0 = int(np.sum(at_risk_mask & (group_labels == groups[0])))
        if n_total > 1:
            frac = n0 / n_total
            var_g1 += (
                d_total * frac * (1.0 - frac) * (n_total - d_total) / (n_total - 1.0)
            )

    if n_groups == 2:
        dof = 1
        if var_g1 <= 0.0:
            stat = 0.0
        else:
            stat = (observed[0] - expected[0]) ** 2 / var_g1
    else:
        dof = n_groups - 1
        safe = expected > 0
        stat = float(np.sum((observed[safe] - expected[safe]) ** 2 / expected[safe]))

    p_value = _chi2_sf(float(stat), dof)
    return float(stat), float(p_value), dof, observed, expected


# --------------------------------------------------------------------------- #
# public: grouped Kaplan-Meier + log-rank
# --------------------------------------------------------------------------- #
def groups_kaplan_meier(df: pd.DataFrame, duration_col: str, event_col: str,
                        group_col: str, show: bool = True) -> dict:
    """Kaplan-Meier per group plus a log-rank test comparing the groups.

    Fits a separate KM curve for every level of ``group_col`` and runs the
    log-rank test for whether the survival curves differ. With exactly 2
    groups the log-rank statistic and its 1-df p-value are exact; with more
    than 2 groups the p-value uses a documented chi-square approximation on
    ``k - 1`` degrees of freedom (see ``_logrank``).

    Parameters
    ----------
    df : pd.DataFrame
    duration_col, event_col : str
        As in :func:`kaplan_meier`.
    group_col : str
        Column whose levels define the groups to compare.
    show : bool
        Print per-group summaries and the log-rank verdict (default True).

    Returns
    -------
    dict
        ``groups`` (mapping each group label to its full KM result dict),
        ``logrank_statistic``, ``logrank_p_value``, ``logrank_dof`` and
        ``significant`` (True when p < 0.05).
    """
    check_df_target(df, group_col)
    durations, events = _extract_survival(df, duration_col, event_col)

    # align the group labels with the rows kept by _extract_survival
    kept = df[[duration_col, event_col]].dropna().index
    group_labels = df.loc[kept, group_col].to_numpy()

    groups = list(pd.unique(pd.Series(group_labels)))
    if len(groups) < 2:
        raise ValueError(
            f"group_col '{group_col}' has only {len(groups)} distinct value(s); "
            "need at least 2 groups to compare."
        )

    per_group = {}
    for g in groups:
        mask = group_labels == g
        per_group[g] = _summarize_km(durations[mask], events[mask])

    stat, p_value, dof, observed, expected = _logrank(
        durations, events, group_labels, groups
    )
    significant = bool(p_value < 0.05)

    result = {
        "groups": per_group,
        "logrank_statistic": stat,
        "logrank_p_value": p_value,
        "logrank_dof": dof,
        "significant": significant,
        "observed_events": {str(g): float(observed[i]) for i, g in enumerate(groups)},
        "expected_events": {str(g): float(expected[i]) for i, g in enumerate(groups)},
    }

    if show:
        print(f"\nBreezeML Grouped Kaplan-Meier - by '{group_col}'")
        print("-" * 64)
        for g in groups:
            res = per_group[g]
            med = res["median_survival"]
            med_str = "not reached" if med is None else f"{med:g}"
            print(
                f"  group {str(g)!s:<12} n={res['n_subjects']:<5} "
                f"events={res['n_observed_events']:<5} "
                f"censored={res['censoring_rate']:.0%}  median={med_str}"
            )
        print("-" * 64)
        approx = "" if len(groups) == 2 else " (>2 groups: approximate)"
        print(
            f"  log-rank chi-square = {stat:.4f} on {dof} df{approx}, "
            f"p = {p_value:.4g}"
        )
        if significant:
            print("  Verdict: survival curves DIFFER significantly (p < 0.05).")
        else:
            print("  Verdict: no significant difference in survival (p >= 0.05).")
        print("-" * 64)

    return result
