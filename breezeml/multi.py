"""
Multi-label classification and multi-output regression.

Sometimes one row has several answers at once. A support ticket can be
tagged "billing" AND "urgent" AND "refund" all together (multi-label
classification), or a sensor reading maps to several numeric outputs like
temperature AND pressure at the same time (multi-output regression).

    from breezeml import multi

    # several binary/multiclass label columns
    model, report = multi.multi_label(df, ["billing", "urgent", "refund"])

    # several numeric target columns
    model, report = multi.multi_output(df, ["temperature", "pressure"])

For multi-label problems the ``chain=True`` option uses a ClassifierChain,
which feeds each label prediction forward as a feature for the next label
so the model can capture correlations between labels (for example, "urgent"
and "refund" tending to co-occur). The default MultiOutputClassifier trains
one independent classifier per label instead.

Everything here builds on the same ColumnTransformer preprocessing and
``build_meta`` metadata as the rest of BreezeML, so downstream features
keep working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
)
from sklearn.pipeline import Pipeline

from ._meta import build_meta
from ._preprocessing import _build_preprocessor
from .regressors import _regression_report

__all__ = ["multi_label", "multi_output"]


def _check_targets(df, targets):
    """Validate a list of target columns and return the feature columns.

    Ensures ``df`` is a DataFrame, ``targets`` is a non-empty list of unique
    columns that all exist, and that at least one feature column remains.
    Because the feature set is defined as "every column that is not a
    target", overlap between features and targets is impossible by
    construction; the only way to get overlap is a duplicated target name,
    which is checked here explicitly.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    if isinstance(targets, str):
        raise TypeError(
            "targets must be a list of column names, not a single string. "
            f"Did you mean [\"{targets}\"]?"
        )
    if targets is None or len(targets) == 0:
        raise ValueError("targets must be a non-empty list of column names.")

    seen = set()
    duplicates = [t for t in targets if t in seen or seen.add(t)]
    if duplicates:
        raise ValueError(
            f"Duplicate target columns are not allowed: {sorted(set(duplicates))}"
        )

    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(
            f"Target column(s) {missing} not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c not in set(targets)]
    if not feature_cols:
        raise ValueError(
            "No feature columns left after removing the targets. "
            "At least one non-target column is required."
        )
    return feature_cols


def _feature_types(df, feature_cols):
    """Split feature columns into numeric and categorical (like _detect_types)."""
    X = df[feature_cols]
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical


def _meta_for_multi(df, feature_cols, targets, task, estimator, task_reason, report):
    """Build BreezeML metadata for a multi-target model.

    ``build_meta`` profiles data against a single target column, so we hand
    it a view containing the features plus one representative target, then
    attach the full target list and combined report onto the result.
    """
    representative = targets[0]
    meta_df = df[feature_cols + [representative]]
    meta = build_meta(
        meta_df,
        representative,
        task,
        estimator,
        task_reason=task_reason,
        stratified=False,
        report=report,
    )
    meta["targets"] = list(targets)
    meta["n_targets"] = len(targets)
    return meta


def multi_label(df, targets, base=None, chain=False, show=True):
    """Multi-label classification: predict several label columns at once.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature columns plus the target columns.
    targets : list of str
        The binary/multiclass label columns to predict. Every remaining
        column is treated as a feature.
    base : estimator, optional
        The per-label classifier. Defaults to ``RandomForestClassifier``.
    chain : bool, default False
        If True, use a ClassifierChain (labels predicted in sequence, each
        one fed forward as a feature) to capture label correlations. If
        False, use MultiOutputClassifier (one independent model per label).
    show : bool, default True
        Print a short report.

    Returns
    -------
    (EasyModel, dict)
        The model stores the list of targets on both ``.target`` and
        ``.targets`` and has ``.task == "multi_label"``. The report has
        per-target accuracy/F1 plus overall ``subset_accuracy`` (exact match
        ratio) and ``hamming_loss``.
    """
    from .breezeml import EasyModel

    feature_cols = _check_targets(df, targets)
    numeric, categorical = _feature_types(df, feature_cols)
    pre = _build_preprocessor(numeric, categorical)

    base_estimator = base if base is not None else RandomForestClassifier(random_state=42)
    if chain:
        multi_estimator = ClassifierChain(base_estimator, random_state=42)
    else:
        multi_estimator = MultiOutputClassifier(base_estimator)

    pipe = Pipeline([("pre", pre), ("model", multi_estimator)])

    X = df[feature_cols]
    y = df[targets]
    # Stratification is not well defined across multiple targets, so use a
    # plain (unstratified) split here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    preds = np.asarray(pipe.predict(X_test))
    y_true = np.asarray(y_test)

    per_target = {}
    for i, name in enumerate(targets):
        yt = y_true[:, i]
        pt = preds[:, i]
        per_target[name] = {
            "accuracy": float(accuracy_score(yt, pt)),
            "f1": float(f1_score(yt, pt, average="weighted", zero_division=0)),
        }

    # subset_accuracy is the exact-match ratio (all labels correct on a row).
    # hamming_loss is the fraction of individual labels predicted wrong.
    # Compute robustly so both binary and multiclass label sets work.
    try:
        subset_accuracy = float(accuracy_score(y_true, preds))
    except ValueError:
        subset_accuracy = float(np.mean(np.all(y_true == preds, axis=1)))
    try:
        ham = float(hamming_loss(y_true, preds))
    except ValueError:
        ham = float(np.mean(y_true != preds))

    report = {
        "per_target": per_target,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": ham,
    }

    strategy = "ClassifierChain" if chain else "MultiOutputClassifier"
    task_reason = (
        f"{len(targets)} label columns {list(targets)} are predicted together, "
        f"so this is multi-label classification using a {strategy}"
        + (" that chains labels to capture their correlations." if chain else ".")
    )
    meta = _meta_for_multi(
        df, feature_cols, targets, "multi_label", multi_estimator, task_reason, report
    )

    model = EasyModel(pipe, list(targets), "multi_label", meta=meta)
    model.targets = list(targets)

    if show:
        print(f"\nBreezeML multi-label classification ({strategy})")
        print(f"targets: {list(targets)}")
        print(f"{'Target':<24}{'Accuracy':<12}{'F1':<12}")
        print("-" * 48)
        for name in targets:
            row = per_target[name]
            print(f"{name:<24}{row['accuracy']:<12.4f}{row['f1']:<12.4f}")
        print("-" * 48)
        print(f"subset_accuracy (exact match): {subset_accuracy:.4f}")
        print(f"hamming_loss:                  {ham:.4f}\n")

    return model, report


def multi_output(df, targets, base=None, show=True):
    """Multi-output regression: predict several numeric target columns at once.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature columns plus the numeric target columns.
    targets : list of str
        The numeric columns to predict. Every remaining column is a feature.
    base : estimator, optional
        The per-target regressor. Defaults to ``RandomForestRegressor``.
    show : bool, default True
        Print a short report.

    Returns
    -------
    (EasyModel, dict)
        The model stores the target list on ``.target`` and ``.targets`` and
        has ``.task == "multi_output"``. The report has per-target
        r2/mae/rmse plus ``average_r2`` across the targets.
    """
    from .breezeml import EasyModel

    feature_cols = _check_targets(df, targets)
    numeric, categorical = _feature_types(df, feature_cols)
    pre = _build_preprocessor(numeric, categorical)

    base_estimator = base if base is not None else RandomForestRegressor(
        n_estimators=200, random_state=42
    )
    multi_estimator = MultiOutputRegressor(base_estimator)
    pipe = Pipeline([("pre", pre), ("model", multi_estimator)])

    X = df[feature_cols]
    y = df[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    preds = np.asarray(pipe.predict(X_test))
    y_true = np.asarray(y_test, dtype=float)
    n_features = X_test.shape[1]

    per_target = {}
    r2_values = []
    for i, name in enumerate(targets):
        target_report = _regression_report(y_true[:, i], preds[:, i], n_features)
        per_target[name] = target_report
        if target_report["r2"] is not None:
            r2_values.append(target_report["r2"])

    average_r2 = float(np.mean(r2_values)) if r2_values else None

    report = {
        "per_target": per_target,
        "average_r2": average_r2,
    }

    task_reason = (
        f"{len(targets)} numeric columns {list(targets)} are predicted together, "
        "so this is multi-output regression using a MultiOutputRegressor "
        "(one regressor fit per target)."
    )
    meta = _meta_for_multi(
        df, feature_cols, targets, "multi_output", multi_estimator, task_reason, report
    )

    model = EasyModel(pipe, list(targets), "multi_output", meta=meta)
    model.targets = list(targets)

    if show:
        print("\nBreezeML multi-output regression (MultiOutputRegressor)")
        print(f"targets: {list(targets)}")
        print(f"{'Target':<24}{'R2':<12}{'MAE':<12}{'RMSE':<12}")
        print("-" * 60)
        for name in targets:
            row = per_target[name]
            r2v = f"{row['r2']:.4f}" if row["r2"] is not None else "n/a"
            maev = f"{row['mae']:.4f}" if row["mae"] is not None else "n/a"
            rmsev = f"{row['rmse']:.4f}" if row["rmse"] is not None else "n/a"
            print(f"{name:<24}{r2v:<12}{maev:<12}{rmsev:<12}")
        print("-" * 60)
        avg = f"{average_r2:.4f}" if average_r2 is not None else "n/a"
        print(f"average_r2: {avg}\n")

    return model, report
