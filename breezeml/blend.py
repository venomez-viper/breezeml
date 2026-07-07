"""
Ensembling: squeeze more accuracy out of the models you already trained.

    model, report = breezeml.blend(df, "target")            # soft-voting top 3
    model, report = breezeml.blend(df, "target", method="stack")  # stacking

Runs compare() under the hood, takes the top performers, and combines
them. Reports the blend against the best single model so you know whether
the extra complexity earned its keep - blends that don't beat the best
single model get called out, not hidden.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ._meta import build_meta
from ._preprocessing import _build_preprocessor, _detect_types
from ._validation import check_df_target

__all__ = ["blend"]


def _detect_task(y):
    return "classification" if (y.dtype == "object" or y.nunique() < 20) else "regression"


def blend(
    df: pd.DataFrame,
    target: str,
    top_k: int = 3,
    method: str = "vote",
    task: str = "auto",
    show: bool = True,
):
    """Combine the top models from compare() into one ensemble.

    Parameters
    ----------
    df, target : training data and target column.
    top_k : int
        How many leaderboard winners to combine (default 3).
    method : str
        "vote" (soft voting / averaging, fast) or "stack" (meta-learner
        on out-of-fold predictions, slower, often stronger).
    task : str
        "auto", "classification", or "regression".

    Returns
    -------
    (EasyModel, dict)
        The fitted ensemble and a report comparing it against the best
        single model on the same holdout.
    """
    from .breezeml import EasyModel, _task_reason

    check_df_target(df, target)
    if method not in ("vote", "stack"):
        raise ValueError("method must be 'vote' or 'stack'.")
    y = df[target]
    if task == "auto":
        task = _detect_task(y)

    if task == "classification":
        from . import classifiers as mod
    else:
        from . import regressors as mod

    leaderboard = mod.compare(df, target, show=False, progress=show)
    score_key = "accuracy" if task == "classification" else "r2"
    ranked = [r for r in leaderboard if r.get(score_key) is not None]
    if len(ranked) < 2:
        raise RuntimeError("Need at least 2 working models to blend.")

    name_key = "classifier" if task == "classification" else "regressor"
    factories = mod._regressor_factories() if task == "regression" else mod._classifier_factories()
    chosen = []
    for row in ranked[:top_k]:
        display = row[name_key]
        factory = factories.get(display)
        if factory is None:
            continue
        estimator = factory()
        # Voting/stacking classifiers need predict_proba for soft voting.
        if task == "classification" and not hasattr(estimator, "predict_proba"):
            continue
        chosen.append((display.lower().replace(" ", "_").replace("(", "").replace(")", ""), estimator))
    if len(chosen) < 2:
        raise RuntimeError("Fewer than 2 blendable models (soft voting needs predict_proba).")

    if task == "classification":
        ensemble = (
            VotingClassifier(estimators=chosen, voting="soft")
            if method == "vote"
            else StackingClassifier(estimators=chosen, final_estimator=LogisticRegression(max_iter=500), cv=3)
        )
    else:
        ensemble = (
            VotingRegressor(estimators=chosen)
            if method == "vote"
            else StackingRegressor(estimators=chosen, final_estimator=Ridge(), cv=3)
        )

    X = df.drop(columns=[target])
    num_cols, cat_cols = _detect_types(df, target)
    pipe = Pipeline([("pre", _build_preprocessor(num_cols, cat_cols)), ("model", ensemble)])

    stratify = y if (task == "classification" and y.nunique() > 1 and y.nunique() < len(y)) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

    if task == "classification":
        holdout = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        }
        blend_score, best_single = holdout["accuracy"], ranked[0][score_key]
    else:
        from . import regressors

        holdout = regressors._regression_report(y_te, pred, X_te.shape[1])
        blend_score, best_single = holdout["r2"], ranked[0][score_key]

    meta = build_meta(df, target, task, ensemble,
                      task_reason=_task_reason(y, task),
                      stratified=stratify is not None, report=holdout)
    meta["blend"] = {
        "method": method,
        "members": [name for name, _ in chosen],
        "best_single_model": ranked[0][name_key],
        "best_single_score": best_single,
        "blend_score": blend_score,
        "beats_best_single": bool(blend_score > best_single),
    }
    model = EasyModel(pipe, target, task, meta=meta)
    report = {**holdout, **meta["blend"], "leaderboard_considered": ranked[:top_k]}

    if show:
        verdict = ("beats" if report["beats_best_single"] else "does NOT beat")
        print(f"Blend ({method}, {len(chosen)} models): {blend_score} - "
              f"{verdict} best single model {ranked[0][name_key]} ({best_single}).")
        if not report["beats_best_single"]:
            print("Honest call: keep the single model. Simpler wins ties.")
    return model, report
