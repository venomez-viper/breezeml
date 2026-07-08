"""
Statistical significance testing for model comparisons.

Two models scoring 0.91 and 0.90 on your holdout are almost never
distinguishable. This module tells you, with a p-value, whether an
observed gap is real signal or the kind of wobble you get from a
different random seed. It sits squarely in the library's honesty
philosophy: when the difference is not significant, it says so out loud
and tells you to keep the simpler, faster model.

    breezeml.significance.mcnemar(model_a, model_b, df, "target")
    breezeml.significance.paired_cv_ttest("random_forest", "logistic", df, "target")

Everything here (the complementary error function, the chi-square
survival function, the Student t two-sided p-value, the regularized
incomplete beta) is implemented directly in numpy so the four-dependency
contract (scikit-learn, pandas, numpy, joblib) stays intact. No scipy,
no statsmodels.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ._preprocessing import _build_preprocessor, _detect_types
from ._validation import check_df_target

__all__ = ["mcnemar", "paired_cv_ttest"]


# ---------------------------------------------------------------------------
# Numerical building blocks (implemented here so we never import scipy).
# ---------------------------------------------------------------------------

def _erf(x: float) -> float:
    """Error function via Abramowitz and Stegun 7.1.26.

    Maximum absolute error about 1.5e-7, which is far tighter than the
    loose tolerances this module reports p-values to. The polynomial is
    only valid for x >= 0, so we fold negative inputs with erf(-x) = -erf(x).
    """
    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    y = 1.0 - poly * np.exp(-x * x)
    return sign * float(y)


def _erfc(x: float) -> float:
    """Complementary error function, erfc(x) = 1 - erf(x).

    Sanity checks: erfc(0) = 1, erfc(1) is about 0.1573.
    """
    return 1.0 - _erf(x)


def _chi2_1df_sf(stat: float) -> float:
    """Survival function (upper tail) of chi-square with 1 degree of freedom.

    For 1 df the survival function has the closed form
    P(X > stat) = erfc(sqrt(stat / 2)), which we evaluate with the erfc
    approximation above.
    """
    if stat <= 0.0:
        return 1.0
    return float(_erfc(np.sqrt(stat / 2.0)))


# Lanczos coefficients (g = 5), classic Numerical Recipes gammln.
_LANCZOS = (
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
)


def _gammln(xx: float) -> float:
    """Natural log of the gamma function for xx > 0 (Lanczos approximation)."""
    x = float(xx)
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015
    y = x
    for coef in _LANCZOS:
        y += 1.0
        ser += coef / y
    return float(-tmp + np.log(2.5066282746310005 * ser / x))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iter = 300
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = np.exp(
        _gammln(a + b) - _gammln(a) - _gammln(b)
        + a * np.log(x) + b * np.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return float(bt * _betacf(a, b, x) / a)
    return float(1.0 - bt * _betacf(b, a, 1.0 - x) / b)


def _t_two_sided_p(t: float, dof: float) -> float:
    """Two-sided p-value for a Student t statistic with ``dof`` degrees.

    Uses the identity that the two-tailed tail area equals the regularized
    incomplete beta I_x(dof/2, 1/2) evaluated at x = dof / (dof + t^2).
    Accuracy tracks the incomplete beta continued fraction (roughly 1e-10),
    comfortably inside the tolerances reported here. As a cross-check, for
    large dof this converges to the normal two-sided p, erfc(|t| / sqrt(2)),
    so a t of 1.96 lands near 0.05.
    """
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    return _betai(dof / 2.0, 0.5, x)


# ---------------------------------------------------------------------------
# Verdict helper (the honest, plain-English part).
# ---------------------------------------------------------------------------

def _verdict(label_a: str, label_b: str, score_a: float, score_b: float, significant: bool) -> str:
    """Plain-English call on a model comparison.

    When the result is not significant we say plainly that the observed
    difference is likely noise and the simpler or faster model should be
    preferred. This is the library's default bias: do not pay for
    complexity a p-value cannot justify.
    """
    if not significant:
        return (
            "Not significant (p >= 0.05): the observed difference is likely noise. "
            "Prefer the simpler or faster model, since the extra complexity is not "
            "buying you a real accuracy gain."
        )
    winner, loser = (label_a, label_b) if score_a >= score_b else (label_b, label_a)
    return (
        f"Significant (p < 0.05): '{winner}' really does beat '{loser}' on this data. "
        "The gap is unlikely to be a random-seed artifact."
    )


def _detect_task(y: pd.Series) -> str:
    return "classification" if (y.dtype == "object" or y.nunique() < 20) else "regression"


# ---------------------------------------------------------------------------
# 1. McNemar's test for two trained classifiers on a shared holdout.
# ---------------------------------------------------------------------------

def mcnemar(model_a, model_b, df: pd.DataFrame, target: str, show: bool = True) -> dict:
    """McNemar's test comparing two trained classifiers on one holdout.

    Both models must already be fitted (an EasyModel or any fitted object
    with a ``predict`` method) and must accept the raw feature frame. We
    split ``df`` once (stratified, test_size=0.2, random_state=42) to match
    the holdout the core trainers carve out, ask both models to predict on
    that shared test slice, and build the 2x2 table of agreement.

    McNemar looks only at the discordant pairs: cases one model gets right
    and the other gets wrong. With the continuity correction the statistic
    is (|b - c| - 1)^2 / (b + c), where b and c are the two discordant
    counts, and the p-value is the upper tail of a chi-square with 1 df.

    Parameters
    ----------
    model_a, model_b : fitted classifiers exposing ``predict``.
    df : pd.DataFrame
        Data including the target column.
    target : str
        Target column name.
    show : bool
        Print a human-readable report (default True).

    Returns
    -------
    dict
        statistic, p_value, n_discordant, only_a_correct, only_b_correct,
        significant (p < 0.05), and a plain-English verdict.
    """
    check_df_target(df, target)

    X = df.drop(columns=[target])
    y = df[target] if isinstance(df[target], pd.Series) else pd.Series(df[target])
    stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_a = np.asarray(model_a.predict(X_te))
        pred_b = np.asarray(model_b.predict(X_te))
    y_true = np.asarray(y_te)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # Discordant cells: b = only A right, c = only B right.
    only_a_correct = int(np.sum(correct_a & ~correct_b))
    only_b_correct = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a_correct + only_b_correct

    # Guard the zero-discordant case (identical predictions): no evidence of
    # a difference, so the statistic is 0 and the p-value is 1.
    if n_discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(only_a_correct - only_b_correct) - 1.0) ** 2 / n_discordant
        statistic = max(statistic, 0.0)  # correction can push tiny counts negative
        p_value = _chi2_1df_sf(statistic)

    significant = bool(p_value < 0.05)
    verdict = _verdict(
        "model_a", "model_b",
        float(np.mean(correct_a)), float(np.mean(correct_b)),
        significant,
    )

    result = {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "n_discordant": n_discordant,
        "only_a_correct": only_a_correct,
        "only_b_correct": only_b_correct,
        "significant": significant,
        "verdict": verdict,
    }

    if show:
        print(f"\nBreezeML McNemar Test - two classifiers on a shared holdout (target: '{target}')")
        print("-" * 68)
        print(f"  Holdout size: {len(y_true)} rows")
        print(f"  Discordant pairs: {n_discordant}  "
              f"(only A correct: {only_a_correct}, only B correct: {only_b_correct})")
        print(f"  McNemar chi-square (continuity corrected): {result['statistic']}")
        print(f"  p-value (chi-square, 1 df): {result['p_value']}")
        print("-" * 68)
        print(f"  Verdict: {verdict}\n")

    return result


# ---------------------------------------------------------------------------
# 2. Paired cross-validation t-test between two named estimators.
# ---------------------------------------------------------------------------

def paired_cv_ttest(
    model_a_name: str,
    model_b_name: str,
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    cv: int = 5,
    show: bool = True,
) -> dict:
    """Paired t-test on per-fold cross-validation scores of two estimators.

    Both estimators are scored on the SAME folds (one shared splitter with a
    fixed seed), so the fold-by-fold score differences are genuinely paired.
    The paired t-statistic is mean_diff / (std_diff / sqrt(n)) over the cv
    folds, and the two-sided p-value comes from a Student t distribution with
    cv - 1 degrees of freedom, evaluated with the regularized incomplete beta
    implemented in this module.

    Parameters
    ----------
    model_a_name, model_b_name : str
        Algorithm names resolved through classifiers._algo_factories() or
        regressors._algo_factories() depending on the task.
    df, target : training data and target column.
    task : str
        "auto", "classification", or "regression".
    cv : int
        Number of cross-validation folds (default 5).
    show : bool
        Print a human-readable report (default True).

    Returns
    -------
    dict
        mean_score_a, mean_score_b, mean_difference, t_statistic, p_value,
        dof, significant (p < 0.05), and a plain-English verdict.
    """
    check_df_target(df, target)
    if cv < 2:
        raise ValueError("cv must be at least 2 for a paired fold comparison.")

    y = df[target] if isinstance(df[target], pd.Series) else pd.Series(df[target])
    if task == "auto":
        task = _detect_task(y)
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")

    if task == "classification":
        from . import classifiers as mod
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        from . import regressors as mod
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = "r2"

    factories = mod._algo_factories()

    def _resolve(name):
        factory = factories.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown {task} algo '{name}'. Choose from: {list(factories.keys())}"
            )
        return factory

    factory_a = _resolve(model_a_name)
    factory_b = _resolve(model_b_name)

    X = df.drop(columns=[target])
    num_cols, cat_cols = _detect_types(df, target)

    def _pipe(name, factory):
        force = name == "multinomial_nb"
        pre = _build_preprocessor(num_cols, cat_cols, force_minmax=force)
        return Pipeline([("pre", pre), ("model", factory())])

    def _score(pipe):
        kwargs = {"estimator": pipe, "X": X, "y": y, "cv": splitter, "scoring": scoring, "n_jobs": -1}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return cross_val_score(**kwargs)
            except PermissionError:
                return cross_val_score(**{**kwargs, "n_jobs": 1})

    scores_a = np.asarray(_score(_pipe(model_a_name, factory_a)), dtype=float)
    scores_b = np.asarray(_score(_pipe(model_b_name, factory_b)), dtype=float)

    diffs = scores_a - scores_b
    n = len(diffs)
    dof = n - 1
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    # Guard the zero-variance case: identical per-fold differences (including
    # two identical models with all-zero differences) give no t evidence.
    if std_diff == 0.0:
        t_statistic = 0.0
        p_value = 1.0
    else:
        t_statistic = mean_diff / (std_diff / np.sqrt(n))
        p_value = _t_two_sided_p(t_statistic, dof)

    significant = bool(p_value < 0.05)
    mean_score_a = float(np.mean(scores_a))
    mean_score_b = float(np.mean(scores_b))
    verdict = _verdict(model_a_name, model_b_name, mean_score_a, mean_score_b, significant)

    result = {
        "mean_score_a": round(mean_score_a, 4),
        "mean_score_b": round(mean_score_b, 4),
        "mean_difference": round(mean_diff, 4),
        "t_statistic": round(float(t_statistic), 4),
        "p_value": round(float(p_value), 4),
        "dof": dof,
        "significant": significant,
        "verdict": verdict,
    }

    if show:
        print(f"\nBreezeML Paired CV t-test - {model_a_name} vs {model_b_name} "
              f"({cv}-fold, {task}, target: '{target}')")
        print("-" * 68)
        print(f"  Mean {scoring} A ({model_a_name}): {result['mean_score_a']}")
        print(f"  Mean {scoring} B ({model_b_name}): {result['mean_score_b']}")
        print(f"  Mean per-fold difference (A - B): {result['mean_difference']}")
        print(f"  t-statistic: {result['t_statistic']}  (dof = {dof})")
        print(f"  p-value (two-sided): {result['p_value']}")
        print("-" * 68)
        print(f"  Verdict: {verdict}\n")

    return result
