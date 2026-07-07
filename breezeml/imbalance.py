"""
Imbalanced classification, handled honestly on the 4 core dependencies.

No SMOTE clones, no synthetic-data magic - the tools that actually move
production metrics:

    breezeml.imbalance.summary(y)                          # how bad is it?
    model, report = breezeml.classify(df, "y", balanced=True)  # class weights
    thr = breezeml.imbalance.tune_threshold(model, df, "y")    # stop using 0.5
    cal = breezeml.imbalance.calibrate(model, df, "y")         # honest probabilities
    cost = breezeml.imbalance.cost_report(model, df, "y", fp_cost=1, fn_cost=25)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from ._validation import check_df_target

__all__ = ["summary", "tune_threshold", "calibrate", "cost_report", "predict_with_threshold"]


def _binary_setup(model, df, target):
    check_df_target(df, target)
    y = df[target]
    classes = sorted(y.unique(), key=str)
    if len(classes) != 2:
        raise ValueError(f"This helper is for binary targets; '{target}' has {len(classes)} classes.")
    pipeline = getattr(model, "pipeline", model)
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Model needs predict_proba for threshold work (tree ensembles, logistic, etc.).")
    X = df.drop(columns=[target])
    # positive class = minority, the usual class of interest
    positive = y.value_counts().idxmin()
    pos_index = list(pipeline.classes_).index(positive)
    return pipeline, X, y, positive, pos_index


def summary(y, show: bool = True) -> dict:
    """Describe how imbalanced a target is and what to do about it."""
    y = pd.Series(y)
    counts = y.value_counts()
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    minority = counts.idxmin()
    result = {
        "classes": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": round(ratio, 2),
        "minority_class": str(minority),
        "minority_share": round(float(counts.min() / counts.sum()), 4),
        "severity": "mild" if ratio < 3 else ("moderate" if ratio < 10 else "severe"),
    }
    if show:
        print(f"Imbalance: {result['imbalance_ratio']}:1 ({result['severity']}), "
              f"minority '{minority}' = {result['minority_share']:.1%} of rows.")
        if ratio >= 3:
            print("Recommended: balanced=True at training, tune_threshold() after, "
                  "judge by macro F1 / recall on the minority class - never accuracy.")
    return result


def tune_threshold(model, df: pd.DataFrame, target: str, metric: str = "f1", show: bool = True) -> dict:
    """Find the decision threshold that maximizes F1 (or precision/recall trade)
    for the minority class, instead of the default 0.5.

    Uses a held-out 25% split of ``df`` so the threshold is not tuned on
    training rows.
    """
    pipeline, X, y, positive, pos_index = _binary_setup(model, df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    proba = pipeline.predict_proba(X_te)[:, pos_index]
    y_bin = (y_te == positive).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_bin, proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    default_pred = (proba >= 0.5).astype(int)
    tuned_pred = (proba >= best_threshold).astype(int)
    result = {
        "positive_class": str(positive),
        "best_threshold": round(best_threshold, 4),
        "f1_at_default": round(float(f1_score(y_bin, default_pred, zero_division=0)), 4),
        "f1_at_best": round(float(f1_score(y_bin, tuned_pred, zero_division=0)), 4),
        "precision_at_best": round(float(precision[best_idx]), 4),
        "recall_at_best": round(float(recall[best_idx]), 4),
    }
    if show:
        print(f"Threshold tuning (positive='{positive}'): 0.50 -> {result['best_threshold']}")
        print(f"  minority F1: {result['f1_at_default']} -> {result['f1_at_best']} "
              f"(precision {result['precision_at_best']}, recall {result['recall_at_best']})")
    return result


def predict_with_threshold(model, X, threshold: float, positive=None):
    """Predict labels using a custom probability threshold (binary)."""
    pipeline = getattr(model, "pipeline", model)
    classes = list(pipeline.classes_)
    if positive is None:
        positive = classes[-1]
    pos_index = classes.index(positive)
    negative = classes[1 - pos_index] if len(classes) == 2 else classes[0]
    proba = pipeline.predict_proba(X)[:, pos_index]
    return np.where(proba >= threshold, positive, negative)


def calibrate(model, df: pd.DataFrame, target: str, method: str = "isotonic", show: bool = True):
    """Calibrate a model's probabilities and report Brier score before/after.

    Uncalibrated probabilities lie: a boosted tree saying "0.9" is often
    right only 70% of the time. Calibration makes probabilities mean what
    they say - essential before threshold tuning or cost decisions.

    Returns
    -------
    (calibrated_pipeline, dict)
    """
    if method not in ("isotonic", "sigmoid"):
        raise ValueError("method must be 'isotonic' or 'sigmoid'.")
    pipeline, X, y, positive, pos_index = _binary_setup(model, df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    y_te_bin = (y_te == positive).astype(int)
    before = brier_score_loss(y_te_bin, pipeline.predict_proba(X_te)[:, pos_index])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(clone(pipeline), method=method, cv=3)
        calibrated.fit(X_tr, y_tr)
    pos_index_cal = list(calibrated.classes_).index(positive)
    after = brier_score_loss(y_te_bin, calibrated.predict_proba(X_te)[:, pos_index_cal])

    report = {
        "method": method,
        "brier_before": round(float(before), 4),
        "brier_after": round(float(after), 4),
        "improved": bool(after < before),
    }
    if show:
        arrow = "improved" if report["improved"] else "did NOT improve (small data? already calibrated?)"
        print(f"Calibration ({method}): Brier {report['brier_before']} -> {report['brier_after']} ({arrow})")
    return calibrated, report


def cost_report(
    model, df: pd.DataFrame, target: str,
    fp_cost: float = 1.0, fn_cost: float = 1.0, show: bool = True,
) -> dict:
    """Pick the threshold that minimizes real-world cost, not a proxy metric.

    Example: missing a fraud case (FN) costs 25x a false alarm (FP):
    ``cost_report(model, df, "fraud", fp_cost=1, fn_cost=25)``.
    """
    pipeline, X, y, positive, pos_index = _binary_setup(model, df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    proba = pipeline.predict_proba(X_te)[:, pos_index]
    y_bin = (y_te == positive).astype(int).to_numpy()

    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        costs.append(fp * fp_cost + fn * fn_cost)
    best_idx = int(np.argmin(costs))
    default_cost = costs[49]  # threshold 0.5

    result = {
        "positive_class": str(positive),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "best_threshold": round(float(thresholds[best_idx]), 2),
        "cost_at_best": float(costs[best_idx]),
        "cost_at_default": float(default_cost),
        "savings_vs_default": float(default_cost - costs[best_idx]),
    }
    if show:
        print(f"Cost-optimal threshold: {result['best_threshold']} "
              f"(cost {result['cost_at_best']:.0f} vs {result['cost_at_default']:.0f} at 0.5; "
              f"saves {result['savings_vs_default']:.0f} on this holdout)")
    return result
