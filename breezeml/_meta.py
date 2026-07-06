"""
Internal helpers that profile training data and capture pipeline metadata.

The resulting ``meta`` dict powers the teaching narration
(``breezeml._narrate``), model cards (``breezeml.card``), and code export
(``breezeml.export``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_data(df: pd.DataFrame, target: str) -> dict:
    """Profile a training DataFrame and its target column.

    Returns a plain dict (JSON-friendly) describing the data facts that
    drive BreezeML's automatic decisions.
    """
    X = df.drop(columns=[target])
    y = df[target]

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_counts = X.isna().sum()
    missing_total = int(missing_counts.sum())
    missing_pct = float(missing_total / max(X.size, 1) * 100.0)
    columns_with_missing = {
        col: int(count) for col, count in missing_counts.items() if count > 0
    }

    target_nunique = int(y.nunique())
    target_missing = int(y.isna().sum())

    # Class balance only meaningful for classification-like targets.
    imbalance_ratio = None
    class_counts = None
    if y.dtype == "object" or target_nunique < 20:
        counts = y.value_counts()
        class_counts = {str(k): int(v) for k, v in counts.items()}
        if len(counts) > 1 and counts.min() > 0:
            imbalance_ratio = float(counts.max() / counts.min())

    # Numeric outlier check (IQR rule) drives the "median imputer" narration.
    outlier_columns = []
    for col in numeric:
        series = X[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        if outliers > 0:
            outlier_columns.append(col)

    return {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_total": missing_total,
        "missing_pct": round(missing_pct, 2),
        "columns_with_missing": columns_with_missing,
        "target": target,
        "target_dtype": str(y.dtype),
        "target_nunique": target_nunique,
        "target_missing": target_missing,
        "class_counts": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
        "outlier_columns": outlier_columns,
    }


def build_meta(
    df: pd.DataFrame,
    target: str,
    task: str,
    estimator,
    *,
    task_reason: str = "",
    stratified: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    report: dict | None = None,
) -> dict:
    """Assemble the full metadata dict stored on a trained EasyModel."""
    from datetime import datetime, timezone

    meta = {
        "profile": profile_data(df, target),
        "task": task,
        "task_reason": task_reason,
        "estimator": type(estimator).__name__,
        "estimator_params": _safe_params(estimator),
        "stratified": stratified,
        "test_size": test_size,
        "random_state": random_state,
        "report": dict(report) if report else None,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        from . import __version__

        meta["breezeml_version"] = __version__
    except Exception:
        meta["breezeml_version"] = "unknown"
    return meta


def _safe_params(estimator) -> dict:
    """Non-default constructor params of an estimator, repr-safe."""
    try:
        defaults = type(estimator)().get_params()
        current = estimator.get_params()
    except Exception:
        return {}
    diff = {}
    for key, value in current.items():
        default = defaults.get(key, object())
        try:
            same = bool(value == default)
        except Exception:
            same = value is default
        if not same:
            diff[key] = repr(value)
    return diff
