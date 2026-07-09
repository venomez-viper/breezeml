"""
Active learning: spend your labeling budget where it actually helps.

Labeling is expensive. If you have 50 labeled rows and 10,000 unlabeled
ones, you cannot afford to label them all - so which rows do you send to a
human next? Random picks waste effort on rows the model already understands.
Active learning ranks the unlabeled pool by how much a label would teach the
model, so you learn faster per label spent.

    from breezeml import active

    model, _ = breezeml.classify(labeled_df, "label")
    pick = active.query(model, unlabeled_df, k=20, strategy="uncertainty")
    # -> label the rows in pick["indices"], add them, retrain

    # Does active learning even beat random on YOUR data? Find out honestly:
    curve = active.simulate(df, "label", initial=20, budget=120, step=20)

``simulate`` always runs a random-sampling baseline alongside the active
loop and reports the area between the two curves, so you can see whether
smart querying was worth it here. Sometimes random is competitive; the
report says so instead of hiding it.

When NOT to use it
------------------
- No ``predict_proba``. Every strategy here scores rows from predicted class
  probabilities. A model that only emits hard labels (some SVMs, some
  custom estimators) cannot be used.
- Noisy data or outliers. Uncertainty sampling chases the rows the model is
  least sure about, and pure noise / mislabeled outliers look exactly like
  that. You can end up spending your whole budget labeling garbage. Audit
  first (see breezeml.audit) and consider "margin" or "entropy" over raw
  "uncertainty".
- Labeling is cheap. If a label costs a fraction of a cent, the machinery is
  not worth it - just label everything and train once.
- Tiny pools. With a few hundred unlabeled rows the overhead of iterative
  retraining rarely pays off over labeling them all.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ._preprocessing import _build_preprocessor, _detect_types
from ._validation import check_df_target

__all__ = ["query", "simulate"]

_STRATEGIES = ("uncertainty", "margin", "entropy", "random")
_RANDOM_SEED = 42


def _get_pipeline(model):
    """Unwrap an EasyModel to its fitted pipeline, or accept a raw estimator."""
    pipeline = getattr(model, "pipeline", model)
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "Active learning needs a classifier with predict_proba "
            "(random forest, logistic regression, gradient boosting, ...). "
            "This model only produces hard labels."
        )
    return pipeline


def _informativeness(proba: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    """Score each row (higher = more informative) under the chosen strategy.

    ``proba`` is the (n_rows, n_classes) predicted probability matrix.
    """
    if strategy == "uncertainty":
        # least confident: 1 - probability of the top class. Always >= 0.
        return 1.0 - proba.max(axis=1)

    if strategy == "margin":
        # smallest gap between the top-2 classes = model is torn = informative.
        # We negate the gap so that "smallest margin" becomes "largest score".
        if proba.shape[1] < 2:
            top2_gap = proba.max(axis=1)
        else:
            part = np.sort(proba, axis=1)
            top2_gap = part[:, -1] - part[:, -2]
        return -top2_gap

    if strategy == "entropy":
        # Shannon entropy of the predicted class distribution. Always >= 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(proba > 0, np.log2(proba), 0.0)
        return -(proba * logs).sum(axis=1)

    if strategy == "random":
        return rng.random(proba.shape[0])

    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose one of {list(_STRATEGIES)}."
    )


def query(
    model,
    unlabeled_df: pd.DataFrame,
    k: int = 10,
    strategy: str = "uncertainty",
    target=None,
    show: bool = True,
) -> dict:
    """Rank an unlabeled pool and return the k most informative rows to label.

    Parameters
    ----------
    model : EasyModel or fitted sklearn estimator/pipeline
        A trained classifier exposing ``predict_proba``.
    unlabeled_df : pd.DataFrame
        The unlabeled pool of feature rows. If ``target`` is given and present
        as a column it is dropped before scoring.
    k : int
        How many rows to recommend for labeling (default 10). Capped at the
        pool size.
    strategy : {"uncertainty", "margin", "entropy", "random"}
        How to score informativeness. "random" is the honest baseline.
    target : str or None
        Name of the target column to drop from ``unlabeled_df`` if present.
    show : bool
        Print the recommended indices and scores (default True).

    Returns
    -------
    dict
        ``{"indices": [...], "scores": [...], "strategy": ..., "k": ...}`` where
        ``indices`` are the top-k row labels of ``unlabeled_df`` ranked most to
        least informative, and ``scores`` are their informativeness scores.
    """
    if not isinstance(unlabeled_df, pd.DataFrame):
        raise TypeError(f"unlabeled_df must be a pandas DataFrame, got {type(unlabeled_df).__name__}")
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {list(_STRATEGIES)}.")

    pipeline = _get_pipeline(model)

    X = unlabeled_df
    if target is not None and target in X.columns:
        X = X.drop(columns=[target])
    if len(X) == 0:
        raise ValueError("unlabeled_df has no rows to query.")

    k = int(k)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    k = min(k, len(X))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = np.asarray(pipeline.predict_proba(X))

    rng = np.random.default_rng(_RANDOM_SEED)
    scores = _informativeness(proba, strategy, rng)

    # descending by score; np.argsort is ascending so reverse it.
    order = np.argsort(scores, kind="stable")[::-1][:k]
    indices = [unlabeled_df.index[int(pos)] for pos in order]
    top_scores = [float(scores[int(pos)]) for pos in order]

    result = {
        "indices": indices,
        "scores": top_scores,
        "strategy": strategy,
        "k": k,
    }

    if show:
        print(f"Active query ({strategy}): top {k} of {len(X)} unlabeled rows to label next")
        print("-" * 60)
        for rank, (idx, sc) in enumerate(zip(indices, top_scores), start=1):
            print(f"  {rank:>3}. row {idx!r:<14} score {sc:.4f}")
        print("-" * 60)

    return result


def _build_pipeline(df: pd.DataFrame, target: str, base):
    """Fresh preprocessor + a clean clone of the base estimator."""
    num_cols, cat_cols = _detect_types(df, target)
    estimator = clone(base)
    return Pipeline([
        ("pre", _build_preprocessor(num_cols, cat_cols)),
        ("model", estimator),
    ])


def _fit_score(df, target, labeled_idx, holdout, base):
    """Train on the currently labeled rows, return holdout accuracy + pipeline."""
    train = df.loc[labeled_idx]
    pipe = _build_pipeline(df, target, base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(train.drop(columns=[target]), train[target])
        preds = pipe.predict(holdout.drop(columns=[target]))
    return float(accuracy_score(holdout[target], preds)), pipe


def _run_curve(df, target, holdout, initial_labeled, budget, step, strategy, base):
    """One active-learning curve: start labeled, query, retrain, record."""
    pool_df = df.drop(index=holdout.index)
    labeled = list(initial_labeled)
    remaining = [i for i in pool_df.index if i not in set(labeled)]
    rng = np.random.default_rng(_RANDOM_SEED)

    budgets: list[int] = []
    accuracies: list[float] = []

    while True:
        acc, pipe = _fit_score(df, target, labeled, holdout, base)
        budgets.append(len(labeled))
        accuracies.append(acc)

        if len(labeled) >= budget or not remaining:
            break

        n_take = min(step, len(remaining), budget - len(labeled))
        X_remaining = pool_df.loc[remaining].drop(columns=[target])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = np.asarray(pipe.predict_proba(X_remaining))
        scores = _informativeness(proba, strategy, rng)
        chosen = np.argsort(scores, kind="stable")[::-1][:n_take]
        picked = [remaining[int(pos)] for pos in chosen]

        picked_set = set(picked)
        labeled.extend(picked)
        remaining = [i for i in remaining if i not in picked_set]

    return budgets, accuracies


def simulate(
    df: pd.DataFrame,
    target: str,
    initial: int = 20,
    budget: int = 100,
    step: int = 10,
    strategy: str = "uncertainty",
    base=None,
    show: bool = True,
) -> dict:
    """Measure whether active learning beats random sampling on THIS dataset.

    Carves an honest holdout, starts from ``initial`` randomly labeled rows,
    then repeatedly trains, queries the ``step`` most informative rows from the
    remaining pool, adds them, and retrains - recording holdout accuracy at
    each budget point up to ``budget`` total labels. The identical loop is run
    with ``strategy="random"`` from the SAME starting set as the baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Fully labeled data (the simulation hides labels to imitate a pool).
    target : str
        Target column name.
    initial : int
        Number of rows labeled before the first query (default 20).
    budget : int
        Total label budget to grow to (default 100).
    step : int
        Rows queried and labeled per round (default 10).
    strategy : {"uncertainty", "margin", "entropy", "random"}
        Active strategy to pit against the random baseline.
    base : sklearn classifier or None
        Estimator refit each round. Default RandomForestClassifier.
    show : bool
        Print the curve and verdict (default True).

    Returns
    -------
    dict
        ``budgets``, ``active_accuracy``, ``random_accuracy`` (equal-length
        lists), ``area_between_curves`` (positive = active ahead on average),
        and ``active_wins`` (bool).
    """
    check_df_target(df, target)
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {list(_STRATEGIES)}.")
    y = df[target]

    # Honest holdout, carved once and shared by both curves.
    stratify = y if y.nunique() > 1 else None
    _, holdout = train_test_split(
        df, test_size=0.25, random_state=_RANDOM_SEED, stratify=stratify,
    )
    pool_df = df.drop(index=holdout.index)
    pool_size = len(pool_df)

    # Clamp the schedule to what the pool can actually supply.
    initial = max(1, min(initial, pool_size))
    budget = min(budget, pool_size)
    if budget < initial:
        budget = initial
    if step < 1:
        step = 1

    if initial < 5:
        warnings.warn(
            "initial < 5 labeled rows makes the first model very unstable; "
            "results may be noisy.",
            stacklevel=2,
        )

    base = base or RandomForestClassifier(n_estimators=100, random_state=_RANDOM_SEED)

    # Same seeded starting set for both curves so the comparison is fair.
    rng = np.random.default_rng(_RANDOM_SEED)
    initial_labeled = list(rng.choice(np.asarray(pool_df.index), size=initial, replace=False))

    active_budgets, active_acc = _run_curve(
        df, target, holdout, initial_labeled, budget, step, strategy, base,
    )
    _, random_acc = _run_curve(
        df, target, holdout, initial_labeled, budget, step, "random", base,
    )

    # Align lengths defensively (identical schedule should already match).
    n = min(len(active_budgets), len(active_acc), len(random_acc))
    budgets = active_budgets[:n]
    active_acc = active_acc[:n]
    random_acc = random_acc[:n]

    diff = np.asarray(active_acc, dtype=float) - np.asarray(random_acc, dtype=float)
    if n >= 2:
        # trapezoidal integral of (active - random) over the budget axis;
        # hand-rolled so it works on both numpy 1.x (trapz) and 2.x (trapezoid).
        x = np.asarray(budgets, dtype=float)
        area = float(np.sum((diff[:-1] + diff[1:]) / 2.0 * np.diff(x)))
    else:
        area = float(diff.sum())

    result = {
        "budgets": budgets,
        "active_accuracy": [round(a, 4) for a in active_acc],
        "random_accuracy": [round(a, 4) for a in random_acc],
        "area_between_curves": round(area, 4),
        "active_wins": bool(area > 0),
    }

    if show:
        print(f"\nActive learning simulation - target '{target}' ({strategy} vs random)")
        print(f"  holdout: {len(holdout)} rows | pool: {pool_size} rows | "
              f"budget {initial} -> {budget} by {step}")
        print("-" * 60)
        print(f"  {'budget':>8}  {'active':>8}  {'random':>8}")
        for b, a, r in zip(budgets, active_acc, random_acc):
            print(f"  {b:>8}  {a:>8.4f}  {r:>8.4f}")
        print("-" * 60)
        if result["active_wins"]:
            verdict = f"active learning WON (area +{result['area_between_curves']})"
        else:
            verdict = (f"active learning did NOT beat random here "
                       f"(area {result['area_between_curves']}) - random is competitive")
        print(f"  Verdict: {verdict}\n")

    return result
