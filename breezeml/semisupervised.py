"""
Semi-supervised learning: train when most of your labels are missing.

Labeling is expensive. If you have 200 labeled rows and 5,000 unlabeled
ones, self-training can use the unlabeled data instead of throwing it away:

    from breezeml import semisupervised

    # rows where target is NaN are treated as unlabeled
    model, report = semisupervised.self_train(df, "label")

The report always includes ``supervised_baseline`` - the score using only
the labeled rows - so you can see whether the unlabeled data actually
helped. Sometimes it does not; the report says so instead of hiding it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from ._meta import build_meta
from ._preprocessing import _build_preprocessor, _detect_types
from ._validation import check_df_target

__all__ = ["self_train"]


def self_train(
    df: pd.DataFrame,
    target: str,
    base_estimator=None,
    threshold: float = 0.75,
    show: bool = True,
):
    """Self-training on a partially labeled DataFrame.

    Rows with a missing (NaN) target are the unlabeled pool. The base
    classifier is trained on labeled rows, labels the unlabeled rows it is
    confident about (probability >= ``threshold``), retrains, and repeats.

    Returns
    -------
    (EasyModel, dict)
        Report includes labeled/unlabeled counts, how many pseudo-labels
        were adopted, holdout metrics, and the supervised-only baseline.
    """
    from .breezeml import EasyModel

    check_df_target(df, target)
    y_all = df[target]
    labeled_mask = y_all.notna()
    n_labeled, n_unlabeled = int(labeled_mask.sum()), int((~labeled_mask).sum())
    if n_unlabeled == 0:
        raise ValueError("No unlabeled rows (NaN targets) found; use breezeml.classify instead.")
    if n_labeled < 20:
        raise ValueError(f"Only {n_labeled} labeled rows; need at least 20 to start self-training.")

    labeled_df = df[labeled_mask]
    y_labeled = labeled_df[target]

    # Honest holdout carved from LABELED data only.
    train_labeled, holdout = train_test_split(
        labeled_df, test_size=0.25, random_state=42,
        stratify=y_labeled if y_labeled.nunique() > 1 else None,
    )

    num_cols, cat_cols = _detect_types(df, target)
    base = base_estimator or RandomForestClassifier(n_estimators=200, random_state=42)

    # Supervised baseline: labeled training rows only.
    baseline_pipe = Pipeline([
        ("pre", _build_preprocessor(num_cols, cat_cols)),
        ("model", base.__class__(**base.get_params())),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline_pipe.fit(train_labeled.drop(columns=[target]), train_labeled[target])
        base_pred = baseline_pipe.predict(holdout.drop(columns=[target]))
    baseline_acc = float(accuracy_score(holdout[target], base_pred))

    # Self-training: labeled training rows + all unlabeled rows (-1 marker).
    train_pool = pd.concat([train_labeled, df[~labeled_mask]])
    X_pool = train_pool.drop(columns=[target])
    classes = np.unique(train_labeled[target].astype(str))
    label_map = {c: i for i, c in enumerate(classes)}
    y_pool = np.array([
        label_map[str(v)] if pd.notna(v) else -1 for v in train_pool[target]
    ])

    pre = _build_preprocessor(num_cols, cat_cols)
    self_trainer = SelfTrainingClassifier(base, threshold=threshold)
    pipe = Pipeline([("pre", pre), ("model", self_trainer)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_pool, y_pool)
        pred_encoded = pipe.predict(holdout.drop(columns=[target]))
    inverse = {i: c for c, i in label_map.items()}
    pred = np.array([inverse[int(p)] for p in pred_encoded])
    y_hold = holdout[target].astype(str)

    self_acc = float(accuracy_score(y_hold, pred))
    trainer = pipe.named_steps["model"]
    n_pseudo = int((np.asarray(trainer.labeled_iter_) > 0).sum()) if hasattr(trainer, "labeled_iter_") else None

    report = {
        "accuracy": round(self_acc, 4),
        "f1": round(float(f1_score(y_hold, pred, average="weighted", zero_division=0)), 4),
        "supervised_baseline_accuracy": round(baseline_acc, 4),
        "gain_over_baseline": round(self_acc - baseline_acc, 4),
        "helped": bool(self_acc > baseline_acc),
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "pseudo_labels_adopted": n_pseudo,
        "confidence_threshold": threshold,
    }

    meta = build_meta(labeled_df, target, "classification", base,
                      task_reason=f"semi-supervised self-training: {n_labeled} labeled, {n_unlabeled} unlabeled rows.",
                      stratified=True, report=report)
    meta["semisupervised"] = {k: report[k] for k in
                              ("n_labeled", "n_unlabeled", "pseudo_labels_adopted", "gain_over_baseline", "helped")}
    model = EasyModel(pipe, target, "classification", meta=meta)

    if show:
        print(f"Self-training: {n_labeled} labeled + {n_unlabeled} unlabeled rows, "
              f"{n_pseudo if n_pseudo is not None else '?'} pseudo-labels adopted.")
        verdict = "helped" if report["helped"] else "did NOT help - keep the supervised baseline"
        print(f"  accuracy {report['supervised_baseline_accuracy']} -> {report['accuracy']} "
              f"({report['gain_over_baseline']:+}) - unlabeled data {verdict}.")
    return model, report
