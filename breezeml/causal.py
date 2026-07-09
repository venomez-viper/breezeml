"""
BreezeML causal inference and uplift modeling: estimate the EFFECT of a
treatment, not just a correlation.

The setup
---------
Every row is a unit (a customer, a patient, a user). One binary column says
whether the unit got the treatment:

    treatment : 1 = treated (got the intervention), 0 = control (did not).
    outcome   : the result you care about (spend, conversion, recovery, ...).

The question is causal: how much did the treatment CHANGE the outcome? The
Average Treatment Effect is

    ATE = E[outcome | treated] - E[outcome | control]

but only if that difference is caused by the treatment and nothing else.

Why the naive difference of means lies
--------------------------------------
Subtracting the two group averages is honest ONLY when treatment was assigned
at random (a proper A/B test / randomized experiment). In observational data,
whoever ended up treated is usually different to begin with: sicker patients
get the drug, engaged users get the email, bigger accounts get the discount.
Those pre-existing differences are CONFOUNDERS. They contaminate the raw
difference of means so it measures "treated vs control units" instead of "the
effect of treating the same unit". This confounding-blindness is exactly the
kind of silent-but-fatal mistake BreezeML calls out loudly.

This module always computes the naive difference AND an adjusted estimate, and
puts them side by side. A large gap between them is your confounding alarm.

What adjustment can and cannot do
---------------------------------
The estimators here (T-learner, inverse-propensity weighting) adjust for the
confounders you MEASURED and put in the covariate columns. They lean on the
untestable no-unmeasured-confounders assumption (a.k.a. ignorability): that
once you condition on the observed covariates, treated and control units are
comparable. You cannot test this assumption with the data. If an important
confounder is missing (unrecorded severity, salesperson skill, motivation),
every number below is biased and no amount of modeling fixes it.

    from breezeml import causal

    res     = causal.estimate_ate(df, "treated", "spend", method="t_learner")
    pair, r = causal.uplift(df, "treated", "spend")
    imb     = causal.check_confounding(df, "treated", "spend")

When NOT to use it
------------------
- This is NOT a randomized experiment. If you can run a proper A/B test, do
  that instead; a randomized difference of means beats any adjustment here.
- If you suspect an unmeasured confounder, treat every estimate as a
  correlation with better manners, not a causal effect.
- Tiny samples or near-total overlap failure (treated and control never look
  alike) make the adjusted estimates extrapolate wildly. Check
  ``check_confounding`` and the propensity range first.
- Interference (one unit's treatment affects another's outcome) breaks the
  whole framework; none of this applies to viral / network effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ._validation import check_df_target

__all__ = ["estimate_ate", "uplift", "check_confounding", "TLearnerPair"]

_SMD_FLAG = 0.1          # |standardized mean difference| above this = imbalance
_PROPENSITY_CLIP = (0.02, 0.98)
_NAIVE_GAP_FLAG = 0.25   # relative gap between naive and adjusted that we warn on


# --------------------------------------------------------------------------- #
# input preparation
# --------------------------------------------------------------------------- #
def _validate(df: pd.DataFrame, treatment: str, outcome: str) -> None:
    check_df_target(df, outcome)
    if treatment not in df.columns:
        raise ValueError(
            f"Treatment column '{treatment}' not found. Available: {list(df.columns)}"
        )
    if treatment == outcome:
        raise ValueError("treatment and outcome must be different columns.")
    vals = pd.unique(df[treatment].dropna())
    extra = set(np.asarray(vals).astype(float).tolist()) - {0.0, 1.0}
    if extra:
        raise ValueError(
            f"Treatment column '{treatment}' must be binary 0/1 "
            f"(1 = treated, 0 = control). Found other values: {sorted(extra)}"
        )
    if df[treatment].isna().any():
        raise ValueError(f"Treatment column '{treatment}' contains missing values.")


def _split_masks(df: pd.DataFrame, treatment: str):
    t = np.asarray(df[treatment]).astype(int)
    treated = t == 1
    control = t == 0
    if treated.sum() == 0 or control.sum() == 0:
        raise ValueError(
            "Need both treated and control rows; one group is empty."
        )
    return treated, control


def _covariate_matrix(df: pd.DataFrame, treatment: str, outcome: str) -> pd.DataFrame:
    """All columns except treatment/outcome, one-hot encoded and numeric.

    Categorical covariates are dummy-encoded so any sklearn estimator can use
    them; the resulting column list is what ``predict_uplift`` re-aligns to.
    """
    cov = df.drop(columns=[treatment, outcome])
    if cov.shape[1] == 0:
        raise ValueError(
            "No covariate columns found (every column is treatment or outcome). "
            "Adjustment needs covariates to condition on; only 'naive' is "
            "meaningful with zero covariates."
        )
    X = pd.get_dummies(cov, drop_first=True)
    if X.shape[1] == 0:
        raise ValueError("Covariates produced no usable numeric columns.")
    return X.astype(float)


def _align(X: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(X).reindex(columns=columns, fill_value=0.0).astype(float)


def _default_regressor():
    # Flexible, low-tuning default. It makes no linearity assumption, which is
    # the honest choice when you do not know the outcome surface.
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)


def _make_regressor(base):
    if base is None:
        return _default_regressor()
    return clone(base)


# --------------------------------------------------------------------------- #
# T-learner
# --------------------------------------------------------------------------- #
class TLearnerPair:
    """Two fitted outcome models (treated + control) with a CATE predictor.

    ``predict_uplift(X)`` returns the per-row Conditional Average Treatment
    Effect: the treated-model prediction minus the control-model prediction,
    i.e. the modeled effect of treating THAT unit. Averaging it gives the ATE.
    """

    def __init__(self, model_treated, model_control, columns: list):
        self.model_treated = model_treated
        self.model_control = model_control
        self.columns = list(columns)
        self.task = "uplift"

    def predict_uplift(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.columns)
        Xd = _align(X, self.columns)
        pred_t = np.asarray(self.model_treated.predict(Xd), dtype=float)
        pred_c = np.asarray(self.model_control.predict(Xd), dtype=float)
        return pred_t - pred_c


def _fit_t_learner(df, treatment, outcome, base) -> TLearnerPair:
    X = _covariate_matrix(df, treatment, outcome)
    y = np.asarray(df[outcome], dtype=float)
    treated, control = _split_masks(df, treatment)
    model_treated = _make_regressor(base).fit(X[treated], y[treated])
    model_control = _make_regressor(base).fit(X[control], y[control])
    return TLearnerPair(model_treated, model_control, list(X.columns))


def _naive_ate(df, treatment, outcome) -> float:
    treated, control = _split_masks(df, treatment)
    y = np.asarray(df[outcome], dtype=float)
    return float(y[treated].mean() - y[control].mean())


def _ipw_ate(df, treatment, outcome):
    """Inverse-propensity-weighted ATE with clipping and self-normalization."""
    X = _covariate_matrix(df, treatment, outcome)
    y = np.asarray(df[outcome], dtype=float)
    t = np.asarray(df[treatment]).astype(int)

    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(X, t)
    p_raw = prop_model.predict_proba(X)[:, 1]
    lo, hi = _PROPENSITY_CLIP
    p = np.clip(p_raw, lo, hi)

    w_t = t / p
    w_c = (1 - t) / (1 - p)
    # self-normalized (Hajek) estimator: far more stable than raw Horvitz-Thompson
    mean_treated = np.sum(w_t * y) / np.sum(w_t)
    mean_control = np.sum(w_c * y) / np.sum(w_c)
    ate = float(mean_treated - mean_control)
    prop_range = (float(p_raw.min()), float(p_raw.max()))
    return ate, prop_range


# --------------------------------------------------------------------------- #
# public: estimate_ate
# --------------------------------------------------------------------------- #
def estimate_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    method: str = "t_learner",
    base=None,
    show: bool = True,
) -> dict:
    """Estimate the Average Treatment Effect, with the confounded baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the treatment column, the outcome column, and the
        covariates (every other column) to adjust for.
    treatment : str
        Binary column, 1 = treated, 0 = control.
    outcome : str
        The result column. Binary outcomes are treated as 0/1 numbers, so the
        effect is a difference in probability (a risk difference).
    method : {"naive", "t_learner", "ipw"}
        - "naive": raw difference of group means. Confounded unless treatment
          was randomized. Always computed and reported regardless of method.
        - "t_learner": fit one outcome model on the treated and one on the
          control, predict both potential outcomes for everyone, average the
          difference.
        - "ipw": inverse-propensity weighting. Fit P(treated | covariates),
          reweight so treated and control look alike, then difference.
    base : sklearn regressor, optional
        Estimator to clone for the T-learner outcome models. Default is a
        random forest (no linearity assumption).
    show : bool
        Print the report.

    Returns
    -------
    dict
        ``ate``, ``naive_ate``, ``method``, ``propensity_range`` (ipw only,
        else None), ``n_treated``, ``n_control``, ``note``.
    """
    _validate(df, treatment, outcome)
    treated, control = _split_masks(df, treatment)
    n_treated = int(treated.sum())
    n_control = int(control.sum())

    naive = _naive_ate(df, treatment, outcome)
    prop_range = None

    method = method.lower()
    if method == "naive":
        ate = naive
    elif method == "t_learner":
        pair = _fit_t_learner(df, treatment, outcome, base)
        X = _covariate_matrix(df, treatment, outcome)
        ate = float(np.mean(pair.predict_uplift(X)))
    elif method == "ipw":
        ate, prop_range = _ipw_ate(df, treatment, outcome)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: naive, t_learner, ipw."
        )

    gap = abs(ate - naive)
    scale = max(abs(naive), abs(ate), 1e-9)
    rel_gap = gap / scale
    confounding_flag = method != "naive" and rel_gap > _NAIVE_GAP_FLAG

    if method == "naive":
        note = (
            "This is the raw difference of means. It equals a causal effect "
            "ONLY if treatment was randomized. In observational data it is "
            "confounded. Re-run with method='t_learner' or 'ipw' to adjust."
        )
    elif confounding_flag:
        note = (
            f"Adjusted ATE ({ate:.4g}) differs from the naive ATE ({naive:.4g}) "
            f"by {rel_gap:.0%}. That gap is the footprint of confounding: the "
            "raw difference was measuring pre-existing differences between the "
            "groups, not the treatment effect. Trust the adjusted number ONLY "
            "if you believe no important confounder is missing."
        )
    else:
        note = (
            f"Adjusted ATE ({ate:.4g}) is close to the naive ATE ({naive:.4g}); "
            "little measured confounding was detected. This does NOT rule out "
            "unmeasured confounding."
        )

    result = {
        "ate": float(ate),
        "naive_ate": float(naive),
        "method": method,
        "propensity_range": prop_range,
        "n_treated": n_treated,
        "n_control": n_control,
        "note": note,
    }

    if show:
        print(f"\nBreezeML Causal ATE - treatment '{treatment}' on outcome '{outcome}'")
        print("-" * 68)
        print(f"  method                : {method}")
        print(f"  n treated / control   : {n_treated:,} / {n_control:,}")
        print(f"  naive ATE (confounded): {naive:.6g}")
        print(f"  adjusted ATE          : {ate:.6g}"
              f"{'' if method != 'naive' else '   (same as naive)'}")
        if prop_range is not None:
            print(f"  propensity range      : [{prop_range[0]:.3f}, {prop_range[1]:.3f}]"
                  f"  (clipped to [{_PROPENSITY_CLIP[0]}, {_PROPENSITY_CLIP[1]}])")
            if prop_range[0] < _PROPENSITY_CLIP[0] or prop_range[1] > _PROPENSITY_CLIP[1]:
                print("  ! Propensities hit the clip bounds: some units are almost")
                print("    always (or never) treated. Overlap is weak; IPW extrapolates.")
        print("-" * 68)
        print("  !! HONESTY WARNING: these estimates are valid ONLY under the")
        print("     no-unmeasured-confounders assumption, which CANNOT be tested")
        print("     from data. A large gap between the naive and adjusted ATE")
        print("     signals confounding among the MEASURED covariates; an")
        print("     unmeasured confounder would bias even the adjusted number.")
        print(f"  {note}")
        print("-" * 68 + "\n")

    return result


# --------------------------------------------------------------------------- #
# public: uplift
# --------------------------------------------------------------------------- #
def uplift(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    base=None,
    show: bool = True,
    k: float = 0.1,
):
    """Fit a T-learner uplift model and summarize who benefits most.

    Uplift (a.k.a. CATE) modeling ranks units by how much the treatment is
    predicted to help THEM specifically, so you can treat the responders and
    spare everyone else.

    Parameters
    ----------
    df, treatment, outcome, base : see ``estimate_ate``.
    show : bool
        Print the report.
    k : float
        Top fraction to summarize (default 0.1 = top decile).

    Returns
    -------
    (TLearnerPair, dict)
        The fitted pair (with ``.predict_uplift(X)``) and a report dict:
        ``top_decile_uplift`` (actual outcome lift among the top-k ranked
        units), ``overall_ate`` (naive lift over everyone), ``k``,
        ``n_top``, ``predicted_top_uplift``, and ``note``.
    """
    _validate(df, treatment, outcome)
    pair = _fit_t_learner(df, treatment, outcome, base)

    X = _covariate_matrix(df, treatment, outcome)
    y = np.asarray(df[outcome], dtype=float)
    t = np.asarray(df[treatment]).astype(int)
    cate = pair.predict_uplift(X)

    overall_ate = _naive_ate(df, treatment, outcome)

    n = len(df)
    n_top = max(1, int(round(n * k)))
    order = np.argsort(cate)[::-1]  # highest predicted uplift first
    top = order[:n_top]
    top_t = t[top] == 1
    top_c = t[top] == 0

    predicted_top_uplift = float(np.mean(cate[top]))
    if top_t.sum() > 0 and top_c.sum() > 0:
        top_decile_uplift = float(y[top][top_t].mean() - y[top][top_c].mean())
        measurable = True
    else:
        top_decile_uplift = None
        measurable = False

    if not measurable:
        note = (
            f"The top {k:.0%} ranked units contain only one treatment arm, so "
            "their actual lift cannot be measured directly. Use randomized "
            "(or well-overlapped) data to validate uplift ranking."
        )
    elif top_decile_uplift > overall_ate:
        note = (
            f"The top {k:.0%} by predicted uplift show an actual lift of "
            f"{top_decile_uplift:.4g} vs {overall_ate:.4g} overall: the model "
            "is finding units that respond more than average. Validate on a "
            "holdout before acting; in-sample uplift is optimistic."
        )
    else:
        note = (
            f"The top {k:.0%} do NOT show higher actual lift "
            f"({top_decile_uplift:.4g}) than the overall {overall_ate:.4g}. The "
            "uplift ranking may be noise, or the effect may be homogeneous."
        )

    report = {
        "top_decile_uplift": top_decile_uplift,
        "overall_ate": float(overall_ate),
        "predicted_top_uplift": predicted_top_uplift,
        "k": float(k),
        "n_top": int(n_top),
        "model": "t_learner",
        "note": note,
    }

    if show:
        print(f"\nBreezeML Uplift (T-learner) - treatment '{treatment}' on '{outcome}'")
        print("-" * 68)
        print(f"  overall ATE (naive)        : {overall_ate:.6g}")
        print(f"  top {k:.0%} predicted uplift : {predicted_top_uplift:.6g}")
        if top_decile_uplift is not None:
            print(f"  top {k:.0%} ACTUAL lift      : {top_decile_uplift:.6g}"
                  f"  (n={n_top:,})")
        else:
            print(f"  top {k:.0%} ACTUAL lift      : not measurable (single arm)")
        print("-" * 68)
        print("  !! Predicted uplift is only as trustworthy as the")
        print("     no-unmeasured-confounders assumption behind it. Ranking")
        print("     quality must be checked on held-out (ideally randomized) data.")
        print(f"  {note}")
        print("-" * 68 + "\n")

    return pair, report


# --------------------------------------------------------------------------- #
# public: check_confounding
# --------------------------------------------------------------------------- #
def check_confounding(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    show: bool = True,
) -> dict:
    """Measure covariate imbalance between treated and control (LOUD).

    For each covariate it reports the standardized mean difference (SMD)

        SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control)/2)

    a scale-free measure of how differently the two groups look on that
    covariate. |SMD| > 0.1 is the conventional flag for meaningful imbalance,
    i.e. evidence that treatment was NOT randomly assigned and that a naive
    difference of means will be biased.

    Returns
    -------
    dict
        ``smd`` (per covariate), ``imbalanced`` (flagged covariates),
        ``max_abs_smd``, ``n_imbalanced``, ``looks_randomized``, ``note``.
    """
    _validate(df, treatment, outcome)
    X = _covariate_matrix(df, treatment, outcome)
    treated, control = _split_masks(df, treatment)

    smd = {}
    for col in X.columns:
        a = np.asarray(X[col])[treated]
        b = np.asarray(X[col])[control]
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        if pooled < 1e-12:
            value = 0.0
        else:
            value = float((a.mean() - b.mean()) / pooled)
        smd[col] = round(value, 4)

    imbalanced = sorted(
        [c for c, v in smd.items() if abs(v) > _SMD_FLAG],
        key=lambda c: abs(smd[c]),
        reverse=True,
    )
    max_abs = max((abs(v) for v in smd.values()), default=0.0)
    looks_randomized = len(imbalanced) == 0

    if looks_randomized:
        note = (
            "No covariate exceeds |SMD| > 0.1. Treated and control look "
            "balanced, consistent with (but not proof of) random assignment. "
            "A naive difference of means is plausibly unbiased here."
        )
    else:
        worst = imbalanced[0]
        note = (
            f"{len(imbalanced)} covariate(s) are imbalanced (worst: '{worst}', "
            f"SMD {smd[worst]:+.2f}). Treated and control differ BEFORE any "
            "treatment, so a naive difference of means is confounded and "
            "biased. Adjust (t_learner / ipw), and remember adjustment only "
            "fixes the confounders you measured."
        )

    result = {
        "smd": smd,
        "imbalanced": imbalanced,
        "n_imbalanced": len(imbalanced),
        "max_abs_smd": round(float(max_abs), 4),
        "looks_randomized": bool(looks_randomized),
        "note": note,
    }

    if show:
        print(f"\nBreezeML Confounding Check - treatment '{treatment}'")
        print("-" * 68)
        print(f"  {'covariate':<32}{'SMD':>10}   flag")
        for col in sorted(smd, key=lambda c: abs(smd[c]), reverse=True):
            flag = "IMBALANCED" if abs(smd[col]) > _SMD_FLAG else "ok"
            print(f"  {str(col):<32}{smd[col]:>10.4f}   {flag}")
        print("-" * 68)
        if not looks_randomized:
            print("  !! WARNING: covariate imbalance detected. Treatment was")
            print("     almost certainly NOT randomly assigned. A naive difference")
            print("     of means WILL be biased by these pre-existing differences.")
        else:
            print("  Covariates look balanced (this is evidence, not proof, of")
            print("  randomization; unmeasured confounders are still invisible).")
        print(f"  {note}")
        print("-" * 68 + "\n")

    return result
