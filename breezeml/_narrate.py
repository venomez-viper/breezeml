"""
Teaching narration: plain-English explanations of BreezeML's automatic
pipeline decisions, generated from a ``meta`` dict (see ``breezeml._meta``).
"""
from __future__ import annotations


def narrate(meta: dict) -> list[str]:
    """Turn a meta dict into a list of plain-English decision explanations."""
    profile = meta.get("profile", {})
    decisions = []

    # 1. Task detection
    task = meta.get("task", "unknown")
    reason = meta.get("task_reason")
    if reason:
        decisions.append(f"Detected a {task} task: {reason}")
    else:
        decisions.append(f"Trained a {task} model on target '{profile.get('target')}'.")

    # 2. Train/test split
    test_size = meta.get("test_size", 0.2)
    holdout_pct = int(test_size * 100)
    if meta.get("stratified"):
        imbalance = profile.get("imbalance_ratio")
        why = (
            f"class imbalance is {imbalance}:1, so a plain random split could "
            "leave rare classes out of the test set"
            if imbalance and imbalance >= 1.5
            else "it keeps class proportions identical in train and test sets"
        )
        decisions.append(
            f"Used a stratified {100 - holdout_pct}/{holdout_pct} train/test split because {why}."
        )
    else:
        decisions.append(
            f"Used a random {100 - holdout_pct}/{holdout_pct} train/test split "
            f"(seed={meta.get('random_state', 42)} for reproducibility)."
        )

    # 3. Missing values / imputation
    numeric = profile.get("numeric_columns", [])
    categorical = profile.get("categorical_columns", [])
    missing_pct = profile.get("missing_pct", 0)
    outliers = profile.get("outlier_columns", [])
    if numeric:
        if missing_pct > 0:
            outlier_note = (
                f" Median (not mean) matters here: {len(outliers)} column(s) contain "
                "outliers that would drag a mean-based fill."
                if outliers
                else ""
            )
            decisions.append(
                f"Filled missing numeric values ({missing_pct}% of cells overall) with the "
                f"column median, which is robust to skew and outliers.{outlier_note}"
            )
        else:
            decisions.append(
                "No missing numeric values found; the median imputer stays in the pipeline "
                "as a safety net for future data."
            )
    if categorical:
        decisions.append(
            f"Encoded {len(categorical)} categorical column(s) with one-hot encoding "
            "(unknown categories at prediction time are ignored instead of crashing), "
            "and filled missing entries with the most frequent value."
        )

    # 4. Scaling
    if numeric:
        decisions.append(
            f"Standardized {len(numeric)} numeric column(s) to mean 0 / std 1 so that "
            "distance- and gradient-based models are not dominated by large-scale features."
        )

    # 5. Metric guidance
    if meta.get("task") == "classification":
        imbalance = profile.get("imbalance_ratio")
        if imbalance and imbalance >= 3:
            decisions.append(
                f"Classes are imbalanced ({imbalance}:1). Accuracy is misleading here; "
                "judge this model by macro F1 instead."
            )
        else:
            decisions.append(
                "Classes are reasonably balanced, so accuracy and weighted F1 are both fair metrics."
            )
    elif meta.get("task") == "regression":
        decisions.append(
            "For regression, check R2 alongside MAE/RMSE: R2 alone can look good while "
            "absolute errors remain too large for your use case."
        )

    # 6. Data-size caveat
    n_rows = profile.get("n_rows", 0)
    if 0 < n_rows < 200:
        decisions.append(
            f"Caution: only {n_rows} rows. Test metrics from a single split are noisy at this "
            "size; prefer cross-validation (cv=5) before trusting them."
        )

    return decisions


def format_narration(decisions: list[str]) -> str:
    """Format decisions as a printable block."""
    lines = ["", "BreezeML decisions explained:"]
    for i, decision in enumerate(decisions, 1):
        lines.append(f"  {i}. {decision}")
    lines.append("")
    return "\n".join(lines)
