"""
Pre-training data audit: catch the mistakes that ruin models before they
reach production.

    report = breezeml.audit(df, "target")

Checks: ID-like columns, constant columns, duplicate rows, contradictory
labels, high-cardinality categoricals, heavy missingness, and the big one -
target leakage, probed by testing whether any single feature predicts the
target suspiciously well on held-out data.

Also: ``audit.contamination(train_df, test_df)`` detects rows leaking
across a train/test split.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ._validation import check_df_target

__all__ = ["audit", "contamination"]

_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key", "number", "no")
LEAK_SCORE_CLF = 0.98
LEAK_SCORE_REG = 0.98


def _is_id_like(name: str, series: pd.Series, n_rows: int) -> bool:
    unique_ratio = series.nunique() / max(n_rows, 1)
    name_lower = name.lower()
    name_hit = any(
        name_lower == hint or name_lower.endswith("_" + hint) or name_lower.endswith(hint)
        for hint in _ID_NAME_HINTS
    )
    return unique_ratio > 0.98 and (name_hit or not pd.api.types.is_float_dtype(series))


def _leakage_probe(X_col: pd.Series, y: pd.Series, task: str) -> float | None:
    """CV score of a tiny tree trained on ONE feature. Near-perfect = leak."""
    values = X_col
    frame = pd.DataFrame({"f": values, "y": y}).dropna()
    if len(frame) < 40 or frame["f"].nunique() < 2:
        return None
    feats = frame[["f"]]
    if not pd.api.types.is_numeric_dtype(feats["f"]):
        # factorize categoricals so the tree can split on them
        feats = pd.DataFrame({"f": pd.factorize(feats["f"])[0]}, index=frame.index)
    target = frame["y"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "classification":
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                scores = cross_val_score(model, feats, target, cv=3, scoring="accuracy")
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
                scores = cross_val_score(model, feats, target, cv=3, scoring="r2")
        return float(scores.mean())
    except Exception:
        return None


def audit(df: pd.DataFrame, target: str, show: bool = True) -> dict:
    """Audit a training DataFrame for the mistakes that break models.

    Parameters
    ----------
    df : pd.DataFrame
        Training data including the target column.
    target : str
        Target column name.
    show : bool
        Print a human-readable findings report (default True).

    Returns
    -------
    dict
        Findings by category, a severity per finding, and an overall
        ``ok`` flag (False when any critical finding exists).
    """
    check_df_target(df, target)
    X = df.drop(columns=[target])
    y = df[target]
    n_rows = len(df)
    task = "classification" if (y.dtype == "object" or y.nunique() < 20) else "regression"

    findings: list[dict] = []

    def add(severity: str, category: str, message: str, columns: list | None = None):
        findings.append({
            "severity": severity, "category": category,
            "message": message, "columns": columns or [],
        })

    # --- structural checks -------------------------------------------------
    id_cols = [c for c in X.columns if _is_id_like(c, X[c], n_rows)]
    if id_cols:
        add("critical", "id_columns",
            f"{len(id_cols)} column(s) look like row identifiers. Models memorize IDs "
            "instead of learning patterns. Drop them before training.", id_cols)

    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant:
        add("warning", "constant_columns",
            f"{len(constant)} column(s) have a single value and carry no signal.", constant)

    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        add("warning", "duplicate_rows",
            f"{dup_rows} exact duplicate row(s) ({dup_rows / n_rows:.1%}). Duplicates that "
            "land in both train and test inflate scores.", [])

    # contradictory labels: identical features, different target
    feature_dups = df[df.duplicated(subset=list(X.columns), keep=False)]
    if len(feature_dups) > 0:
        conflicting = feature_dups.groupby(list(X.columns), dropna=False)[target].nunique()
        n_conflicts = int((conflicting > 1).sum())
        if n_conflicts > 0:
            add("warning", "label_noise",
                f"{n_conflicts} feature-identical group(s) carry different labels. "
                "No model can resolve these; they cap your achievable accuracy.", [])

    high_card = [
        c for c in X.select_dtypes(exclude=[np.number]).columns
        if X[c].nunique() > max(50, n_rows * 0.5) and c not in id_cols
    ]
    if high_card:
        add("warning", "high_cardinality",
            f"{len(high_card)} categorical column(s) have very many levels; one-hot "
            "encoding will explode the feature space.", high_card)

    heavy_missing = [c for c in X.columns if X[c].isna().mean() > 0.5]
    if heavy_missing:
        add("warning", "heavy_missingness",
            f"{len(heavy_missing)} column(s) are more than half missing; imputation "
            "will dominate whatever signal remains.", heavy_missing)

    if task == "classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() > 0 and counts.max() / counts.min() >= 10:
            add("warning", "class_imbalance",
                f"Severe class imbalance ({counts.max() / counts.min():.0f}:1). Accuracy will "
                "mislead; see breezeml.imbalance for thresholds and weights.", [])

    # --- leakage probes ----------------------------------------------------
    leak_threshold = LEAK_SCORE_CLF if task == "classification" else LEAK_SCORE_REG
    leaks = []
    probe_cols = [c for c in X.columns if c not in id_cols and c not in constant]
    for col in probe_cols:
        score = _leakage_probe(X[col], y, task)
        if score is not None and score >= leak_threshold:
            leaks.append({"column": col, "single_feature_score": round(score, 4)})
    if leaks:
        add("critical", "target_leakage",
            f"{len(leaks)} feature(s) predict the target almost perfectly ON THEIR OWN. "
            "That usually means the target leaked into the features (a post-outcome "
            "column, an encoded label, a proxy). Investigate before trusting any model.",
            [leak["column"] for leak in leaks])

    critical = [f for f in findings if f["severity"] == "critical"]
    result = {
        "ok": len(critical) == 0,
        "n_rows": n_rows,
        "task": task,
        "findings": findings,
        "leak_details": leaks,
        "summary": (
            "No critical issues found."
            if not critical
            else f"{len(critical)} CRITICAL issue(s): " + ", ".join(f["category"] for f in critical)
        ),
    }

    if show:
        print(f"\nBreezeML Data Audit - target: '{target}' ({n_rows:,} rows, {task})")
        print("-" * 64)
        if not findings:
            print("  Clean. No issues detected.")
        for f in findings:
            marker = "!!" if f["severity"] == "critical" else "! "
            print(f"  {marker} [{f['severity'].upper()}] {f['message']}")
            if f["columns"]:
                cols_shown = ", ".join(str(c) for c in f["columns"][:8])
                more = "" if len(f["columns"]) <= 8 else f" (+{len(f['columns']) - 8} more)"
                print(f"       columns: {cols_shown}{more}")
        print("-" * 64)
        print(f"  Verdict: {result['summary']}\n")

    return result


def contamination(train_df: pd.DataFrame, test_df: pd.DataFrame, show: bool = True) -> dict:
    """Detect rows that appear in both train and test sets.

    Train/test contamination silently inflates every metric you report.
    """
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("train_df and test_df must have identical columns.")

    merged = pd.merge(
        train_df.drop_duplicates(), test_df.drop_duplicates(),
        how="inner", on=list(train_df.columns),
    )
    n_shared = len(merged)
    share_of_test = n_shared / max(len(test_df), 1)
    result = {
        "shared_rows": int(n_shared),
        "share_of_test": round(float(share_of_test), 4),
        "contaminated": bool(n_shared > 0),
    }
    if show:
        if n_shared == 0:
            print("No train/test contamination detected.")
        else:
            print(
                f"CONTAMINATION: {n_shared} row(s) appear in both train and test "
                f"({share_of_test:.1%} of the test set). Metrics on this split are inflated."
            )
    return result
