"""
Data drift detection for deployed BreezeML models.

Models trained with the core API (v1.3+) store compact reference
distributions of their training data. ``drift.check()`` compares new data
against that reference and reports, per column:

- PSI (Population Stability Index) for numeric columns
- new / vanished categories and PSI for categorical columns
- missing-rate shifts

PSI conventions: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 drift.

    result = breezeml.drift.check(model, new_df)
    if result["drifted"]:
        print(result["summary"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["check", "psi"]

_EPS = 1e-4
PSI_WARN = 0.10
PSI_DRIFT = 0.25


def psi(ref_props, new_props) -> float:
    """Population Stability Index between two proportion vectors."""
    ref = np.clip(np.asarray(ref_props, dtype=float), _EPS, None)
    new = np.clip(np.asarray(new_props, dtype=float), _EPS, None)
    ref = ref / ref.sum()
    new = new / new.sum()
    return float(np.sum((new - ref) * np.log(new / ref)))


def _status(value: float) -> str:
    if value >= PSI_DRIFT:
        return "drift"
    if value >= PSI_WARN:
        return "warning"
    return "ok"


def _check_numeric(reference: dict, series: pd.Series) -> dict:
    edges = np.asarray(reference["bin_edges"], dtype=float)
    ref_props = np.asarray(reference["bin_props"], dtype=float)
    values = series.dropna().astype(float)
    if len(values) == 0:
        return {"psi": None, "status": "no_data"}
    # Clip so out-of-range values land in the edge bins instead of vanishing.
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    new_props = counts / max(counts.sum(), 1)
    value = round(psi(ref_props, new_props), 4)
    out_of_range = float(((values < edges[0]) | (values > edges[-1])).mean())
    return {
        "psi": value,
        "status": _status(value),
        "share_outside_training_range": round(out_of_range, 4),
    }


def _check_categorical(reference: dict, series: pd.Series) -> dict:
    values = series.dropna().astype(str)
    if len(values) == 0:
        return {"psi": None, "status": "no_data"}
    new_props_full = values.value_counts(normalize=True)
    categories = list(reference.keys())
    ref_props = [reference[c] for c in categories]
    new_props = [float(new_props_full.get(c, 0.0)) for c in categories]

    # Anything not seen in training gets pooled into one "unseen" bucket.
    unseen_share = float(sum(p for c, p in new_props_full.items() if c not in reference))
    ref_props.append(_EPS)
    new_props.append(unseen_share)

    value = round(psi(ref_props, new_props), 4)
    new_categories = sorted(c for c in new_props_full.index if c not in reference)[:10]
    return {
        "psi": value,
        "status": _status(value),
        "new_categories": new_categories,
        "unseen_category_share": round(unseen_share, 4),
    }


def check(model, new_df: pd.DataFrame, threshold: float = PSI_DRIFT) -> dict:
    """Compare new data against a model's training reference distributions.

    Parameters
    ----------
    model : EasyModel
        Model trained with the core API (v1.3+), carrying reference stats
        in ``model.meta["reference"]``.
    new_df : pd.DataFrame
        New feature data (the target column, if present, is ignored).
    threshold : float
        PSI at or above which a column counts as drifted (default 0.25).

    Returns
    -------
    dict
        Per-column results, a list of drifted columns, an overall
        ``drifted`` flag, and a human-readable ``summary``.
    """
    meta = getattr(model, "meta", None) or {}
    reference = meta.get("reference")
    if not reference:
        raise ValueError(
            "This model has no reference distributions. Drift checks need a model "
            "trained with breezeml v1.3+ core API (fit / auto / classify / regress / automl)."
        )

    data = new_df.copy()
    target = getattr(model, "target", None)
    if target and target in data.columns:
        data = data.drop(columns=[target])

    columns = {}
    for col, ref in reference["numeric"].items():
        if col not in data.columns:
            columns[col] = {"status": "missing_column"}
            continue
        columns[col] = _check_numeric(ref, data[col])
    for col, ref in reference["categorical"].items():
        if col not in data.columns:
            columns[col] = {"status": "missing_column"}
            continue
        columns[col] = _check_categorical(ref, data[col])

    # Missing-rate shifts
    for col, ref_rate in reference["missing_rates"].items():
        if col in data.columns and col in columns:
            new_rate = float(data[col].isna().mean())
            columns[col]["missing_rate_train"] = round(ref_rate, 4)
            columns[col]["missing_rate_now"] = round(new_rate, 4)
            if new_rate - ref_rate > 0.10 and columns[col].get("status") == "ok":
                columns[col]["status"] = "warning"

    drifted_cols = [c for c, r in columns.items() if r.get("psi") is not None and r["psi"] >= threshold]
    warning_cols = [c for c, r in columns.items() if r.get("status") == "warning"]
    missing_cols = [c for c, r in columns.items() if r.get("status") == "missing_column"]

    drifted = bool(drifted_cols or missing_cols)
    parts = []
    if drifted_cols:
        parts.append(f"{len(drifted_cols)} column(s) drifted (PSI >= {threshold}): {drifted_cols}")
    if missing_cols:
        parts.append(f"{len(missing_cols)} training column(s) absent from new data: {missing_cols}")
    if warning_cols:
        parts.append(f"{len(warning_cols)} column(s) show moderate shift: {warning_cols}")
    if not parts:
        parts.append("No significant drift detected.")
    summary = " ".join(parts)

    return {
        "drifted": drifted,
        "drifted_columns": drifted_cols,
        "warning_columns": warning_cols,
        "missing_columns": missing_cols,
        "columns": columns,
        "n_rows_checked": int(len(data)),
        "threshold": threshold,
        "summary": summary,
    }
