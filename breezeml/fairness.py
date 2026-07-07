"""
Fairness auditing: per-group performance and selection-rate parity.

Every BreezeML model card warns "no fairness audit was performed."
This module is how you perform one:

    result = breezeml.fairness.report(model, df, sensitive="gender")

For classification it reports, per group: size, accuracy, F1, selection
rate (share predicted positive), TPR and FPR, plus the demographic parity
ratio and the four-fifths (80%) rule verdict. For regression: per-group
MAE and mean error (bias direction).

A fairness report is evidence, not absolution: passing parity on one
attribute does not make a model fair. But it beats not looking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from ._validation import check_df_target

__all__ = ["report"]

FOUR_FIFTHS = 0.8
_MIN_GROUP = 10


def _positive_label(y: pd.Series, positive):
    if positive is not None:
        return positive
    # default: minority class in binary problems (usually the outcome of interest)
    counts = y.value_counts()
    if len(counts) == 2:
        return counts.idxmin()
    return None


def report(
    model,
    df: pd.DataFrame,
    sensitive: str,
    target: str | None = None,
    positive=None,
    show: bool = True,
) -> dict:
    """Per-group fairness report for a trained model.

    Parameters
    ----------
    model : EasyModel or sklearn estimator
        Trained model.
    df : pd.DataFrame
        Evaluation data containing the target column and the sensitive
        attribute. Use a holdout set, not training data.
    sensitive : str
        Column defining the groups (e.g. "gender", "age_band"). The column
        may be excluded from features; it only needs to exist in ``df``.
    target : str, optional
        Target column (defaults to the model's stored target).
    positive : optional
        The positive class for selection rate / TPR / FPR. Defaults to the
        minority class in binary problems.
    show : bool
        Print the report table.

    Returns
    -------
    dict
        Per-group metrics, parity measures, and plain-language notes.
    """
    target = target or getattr(model, "target", None)
    if target is None:
        raise ValueError("Pass target= (the model carries no target name).")
    check_df_target(df, target)
    if sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive}' not found in DataFrame.")

    task = getattr(model, "task", None)
    y_true = df[target]
    if task is None:
        task = "classification" if (y_true.dtype == "object" or y_true.nunique() < 20) else "regression"

    # Predict using the model's expected feature columns.
    X = df.drop(columns=[target])
    meta = getattr(model, "meta", None) or {}
    profile = meta.get("profile", {})
    feature_cols = list(profile.get("numeric_columns", [])) + list(profile.get("categorical_columns", []))
    if feature_cols:
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"DataFrame is missing model feature columns: {missing}")
        X = X[feature_cols]
    preds = pd.Series(np.asarray(model.predict(X)), index=df.index)

    groups = df[sensitive].astype(str).fillna("(missing)")
    group_names = sorted(groups.unique())

    rows = {}
    notes = []
    pos = _positive_label(y_true, positive) if task == "classification" else None

    for g in group_names:
        mask = groups == g
        n = int(mask.sum())
        if n < _MIN_GROUP:
            notes.append(f"Group '{g}' has only {n} rows; its numbers are noise, not evidence.")
        yt, yp = y_true[mask], preds[mask]
        if task == "classification":
            entry = {
                "n": n,
                "accuracy": round(float(accuracy_score(yt, yp)), 4),
                "f1": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
            }
            if pos is not None:
                predicted_pos = yp == pos
                actual_pos = yt == pos
                entry["selection_rate"] = round(float(predicted_pos.mean()), 4)
                entry["tpr"] = (
                    round(float((predicted_pos & actual_pos).sum() / actual_pos.sum()), 4)
                    if actual_pos.sum() > 0 else None
                )
                entry["fpr"] = (
                    round(float((predicted_pos & ~actual_pos).sum() / (~actual_pos).sum()), 4)
                    if (~actual_pos).sum() > 0 else None
                )
        else:
            errors = np.asarray(yt, dtype=float) - np.asarray(yp, dtype=float)
            entry = {
                "n": n,
                "mae": round(float(mean_absolute_error(yt, yp)), 4),
                "mean_error": round(float(errors.mean()), 4),
            }
        rows[g] = entry

    result = {"task": task, "sensitive": sensitive, "positive_class": pos,
              "groups": rows, "notes": notes}

    if task == "classification" and pos is not None:
        rates = {g: r["selection_rate"] for g, r in rows.items() if r.get("selection_rate") is not None}
        if len(rates) >= 2 and max(rates.values()) > 0:
            dp_ratio = min(rates.values()) / max(rates.values())
            result["demographic_parity_ratio"] = round(float(dp_ratio), 4)
            result["passes_four_fifths"] = bool(dp_ratio >= FOUR_FIFTHS)
            tprs = [r["tpr"] for r in rows.values() if r.get("tpr") is not None]
            fprs = [r["fpr"] for r in rows.values() if r.get("fpr") is not None]
            if len(tprs) >= 2:
                result["tpr_gap"] = round(float(max(tprs) - min(tprs)), 4)
            if len(fprs) >= 2:
                result["fpr_gap"] = round(float(max(fprs) - min(fprs)), 4)
            if not result["passes_four_fifths"]:
                lowest = min(rates, key=rates.get)
                notes.append(
                    f"FAILS the four-fifths rule (parity ratio {dp_ratio:.2f} < 0.80): "
                    f"group '{lowest}' is selected far less often. If this decision "
                    "affects people, do not deploy without review."
                )
    elif task == "regression":
        maes = {g: r["mae"] for g, r in rows.items()}
        if len(maes) >= 2 and min(maes.values()) > 0:
            error_ratio = max(maes.values()) / min(maes.values())
            result["error_ratio"] = round(float(error_ratio), 4)
            if error_ratio > 1.5:
                worst = max(maes, key=maes.get)
                notes.append(
                    f"Error is {error_ratio:.1f}x higher for group '{worst}'. The model "
                    "serves some groups much worse than others."
                )

    if show:
        print(f"\nBreezeML Fairness Report - sensitive attribute: '{sensitive}' ({task})")
        print("-" * 66)
        for g, r in rows.items():
            parts = ", ".join(f"{k}={v}" for k, v in r.items())
            print(f"  {g:<20} {parts}")
        for key in ("demographic_parity_ratio", "passes_four_fifths", "tpr_gap", "fpr_gap", "error_ratio"):
            if key in result:
                print(f"  {key}: {result[key]}")
        for note in notes:
            print(f"  ! {note}")
        if not notes:
            print("  No parity violations detected on this attribute (this is evidence, not absolution).")
        print("-" * 66 + "\n")

    return result
