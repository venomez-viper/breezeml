"""
BreezeAutoML: budget-aware automated model selection and tuning.

One call searches across every built-in model, then spends the remaining
time budget tuning the most promising candidates:

    model, report = breezeml.automl(df, "target", time_budget=60)

Runs entirely on the 4 core dependencies. If Optuna is installed
(``pip install breezeml[automl]``), pass ``backend="optuna"`` for TPE
search instead of the native random search.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ._meta import build_meta
from ._narrate import format_narration, narrate
from ._preprocessing import _build_preprocessor, _detect_types
from ._progress import ProgressBar
from ._validation import check_df_target

__all__ = ["automl"]

# Fraction of the budget spent screening all models before tuning the best.
_SCREEN_FRACTION = 0.35
_TOP_K = 3


def _candidates(task):
    """(name, factory, param_grid) triples for the given task."""
    if task == "classification":
        from . import classifiers as mod
    else:
        from . import regressors as mod
    factories = mod._algo_factories()
    grids = mod._PARAM_GRIDS
    return [(name, factory, grids.get(name)) for name, factory in factories.items()]


def _detect_task(y):
    return "classification" if (y.dtype == "object" or y.nunique() < 20) else "regression"


def _scoring(task, metric):
    if metric:
        return metric
    return "accuracy" if task == "classification" else "r2"


def _make_pipe(df, target, factory, algo_name):
    num_cols, cat_cols = _detect_types(df, target)
    pre = _build_preprocessor(num_cols, cat_cols, force_minmax=(algo_name == "multinomial_nb"))
    return Pipeline([("pre", pre), ("model", factory())])


def _screen(df, target, task, scoring, deadline, cv, progress=False):
    """Quick cross-validated score for every candidate until the deadline."""
    X = df.drop(columns=[target])
    y = df[target]
    candidates = _candidates(task)
    bar = ProgressBar(len(candidates), desc="Screening models", enabled=progress)
    results = []
    for name, factory, grid in candidates:
        if time.monotonic() > deadline:
            bar.close(f"Screening stopped at budget: {len(results)}/{len(candidates)} models")
            break
        bar.update(name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = _make_pipe(df, target, factory, name)
                start = time.monotonic()
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                cost = time.monotonic() - start
            results.append({
                "model": name,
                "score": round(float(scores.mean()), 4),
                "score_std": round(float(scores.std()), 4),
                "fit_seconds": round(cost, 2),
                "tunable": grid is not None,
            })
        except Exception as exc:
            results.append({"model": name, "score": None, "error": str(exc)[:120]})
    else:
        bar.close()
    results.sort(key=lambda r: r["score"] if r["score"] is not None else float("-inf"), reverse=True)
    return results


def _tune_native(df, target, task, name, factory, grid, scoring, budget_seconds, cv):
    """RandomizedSearchCV sized to fit the remaining budget."""
    X = df.drop(columns=[target])
    y = df[target]
    pipe = _make_pipe(df, target, factory, name)

    # Estimate how many configs fit: one config costs roughly cv single fits.
    probe_start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    per_fit = max(time.monotonic() - probe_start, 0.05)
    n_iter = max(1, min(30, int(budget_seconds / (per_fit * cv))))

    grid_size = 1
    for values in grid.values():
        grid_size *= len(values)
    n_iter = min(n_iter, grid_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_kwargs = {
            "estimator": pipe,
            "param_distributions": grid,
            "n_iter": n_iter,
            "cv": cv,
            "scoring": scoring,
            "random_state": 42,
            "n_jobs": -1,
        }
        search = RandomizedSearchCV(**search_kwargs)
        try:
            search.fit(X, y)
        except PermissionError:
            search = RandomizedSearchCV(**{**search_kwargs, "n_jobs": 1})
            search.fit(X, y)

    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, float(search.best_score_), params, n_iter


def _tune_optuna(df, target, task, name, factory, grid, scoring, budget_seconds, cv):
    """Optuna TPE search over the same grid (requires the [automl] extra)."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "backend='optuna' requires Optuna. Install with: pip install breezeml[automl]"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = df.drop(columns=[target])
    y = df[target]

    def objective(trial):
        params = {key: trial.suggest_categorical(key, values) for key, values in grid.items()}
        pipe = _make_pipe(df, target, factory, name)
        pipe.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, timeout=budget_seconds, show_progress_bar=False)

    best_pipe = _make_pipe(df, target, factory, name)
    best_pipe.set_params(**study.best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipe.fit(X, y)
    params = {k.replace("model__", ""): v for k, v in study.best_params.items()}
    return best_pipe, float(study.best_value), params, len(study.trials)


def automl(
    df: pd.DataFrame,
    target: str,
    time_budget: int = 60,
    task: str = "auto",
    metric: str | None = None,
    cv: int = 3,
    backend: str = "native",
    explain_decisions: bool = False,
    show: bool = True,
):
    """Automated model selection and tuning within a time budget.

    Screens every built-in model with cross-validation, then spends the
    remaining budget tuning the top candidates. Returns the best model
    refit on all data, plus a full search report.

    Parameters
    ----------
    df : pd.DataFrame
        Training data including the target column.
    target : str
        Target column name.
    time_budget : int
        Soft budget in seconds (default 60). The search stops starting new
        work once exceeded; a running fit is allowed to finish.
    task : str
        "auto" (default), "classification", or "regression".
    metric : str, optional
        sklearn scoring string (default: accuracy / r2).
    cv : int
        Cross-validation folds (default 3).
    backend : str
        "native" (RandomizedSearchCV, zero extra deps) or "optuna"
        (TPE search, requires ``pip install breezeml[automl]``).
    explain_decisions : bool
        Print the plain-English narration after the search.
    show : bool
        Print the screening leaderboard and progress.

    Returns
    -------
    (EasyModel, dict)
        The best model and a report with the leaderboard, tuning history,
        and final holdout metrics.
    """
    from .breezeml import EasyModel, _task_reason

    check_df_target(df, target)
    if backend not in ("native", "optuna"):
        raise ValueError(f"Unknown backend '{backend}'. Choose 'native' or 'optuna'.")

    y = df[target]
    if task == "auto":
        task = _detect_task(y)
    scoring = _scoring(task, metric)

    start = time.monotonic()
    screen_deadline = start + time_budget * _SCREEN_FRACTION

    if show:
        print(f"BreezeAutoML: task={task}, metric={scoring}, budget={time_budget}s, backend={backend}")
        print(f"Stage 1/2: screening models ({int(time_budget * _SCREEN_FRACTION)}s max)...")

    leaderboard = _screen(df, target, task, scoring, screen_deadline, cv, progress=show)
    screened = [r for r in leaderboard if r.get("score") is not None]
    if not screened:
        raise RuntimeError("AutoML screening failed for every model; check your data.")

    if show:
        for i, row in enumerate(leaderboard[:8], 1):
            score = row["score"] if row["score"] is not None else "FAILED"
            print(f"  {i}. {row['model']:<20} {score}")

    # Stage 2: tune the top tunable candidates with the remaining budget.
    top = [r for r in screened if r.get("tunable")][:_TOP_K]
    tune_fn = _tune_optuna if backend == "optuna" else _tune_native
    tuning_history = []
    best_name, best_pipe, best_score, best_params = None, None, float("-inf"), {}

    candidates_by_name = {name: (factory, grid) for name, factory, grid in _candidates(task)}
    for rank, row in enumerate(top):
        remaining = time_budget - (time.monotonic() - start)
        if remaining <= 2:
            break
        slice_budget = remaining / (len(top) - rank)
        factory, grid = candidates_by_name[row["model"]]
        if show:
            print(f"Stage 2/2: tuning {row['model']} ({int(slice_budget)}s slice)...")
        try:
            pipe, score, params, n_configs = tune_fn(
                df, target, task, row["model"], factory, grid, scoring, slice_budget, cv
            )
        except ImportError:
            raise
        except Exception as exc:
            tuning_history.append({"model": row["model"], "error": str(exc)[:120]})
            continue
        tuning_history.append({
            "model": row["model"], "score": round(score, 4),
            "params": params, "configs_tried": n_configs,
        })
        if score > best_score:
            best_name, best_pipe, best_score, best_params = row["model"], pipe, score, params

    # Fall back to the screening winner if tuning never completed.
    if best_pipe is None:
        winner = screened[0]
        factory, _ = candidates_by_name[winner["model"]]
        best_pipe = _make_pipe(df, target, factory, winner["model"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_pipe.fit(df.drop(columns=[target]), y)
        best_name, best_score = winner["model"], winner["score"]

    # Honest final holdout evaluation with the winning config.
    X = df.drop(columns=[target])
    stratify = y if (task == "classification" and y.nunique() > 1 and y.nunique() < len(y)) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    from sklearn.base import clone

    holdout_pipe = clone(best_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holdout_pipe.fit(X_tr, y_tr)
        pred = holdout_pipe.predict(X_te)
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        holdout = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        }
    else:
        from . import regressors

        holdout = regressors._regression_report(y_te, pred, X_te.shape[1])

    elapsed = round(time.monotonic() - start, 1)
    estimator = best_pipe.named_steps["model"]
    meta = build_meta(
        df, target, task, estimator,
        task_reason=_task_reason(y, task),
        stratified=stratify is not None,
        report=holdout,
    )
    meta["automl"] = {
        "backend": backend,
        "time_budget": time_budget,
        "time_used": elapsed,
        "metric": scoring,
        "models_screened": len(leaderboard),
        "best_model": best_name,
        "best_cv_score": round(best_score, 4),
        "best_params": best_params,
    }

    model = EasyModel(best_pipe, target, task, meta=meta)
    report = {
        "best_model": best_name,
        "best_cv_score": round(best_score, 4),
        "best_params": best_params,
        "holdout": holdout,
        "leaderboard": leaderboard,
        "tuning_history": tuning_history,
        "time_used_seconds": elapsed,
        "backend": backend,
    }

    if show:
        print(f"Done in {elapsed}s. Best: {best_name} (cv {scoring}={round(best_score, 4)})")
    if explain_decisions:
        print(format_narration(narrate(meta)))

    return model, report
