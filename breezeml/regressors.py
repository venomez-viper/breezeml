"""
BreezeML regressors.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ._preprocessing import _build_preprocessor, _detect_types
from ._validation import check_df_target


def _round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def _safe_mape(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = np.abs(y_true_arr) > 1e-12
    if not np.any(non_zero):
        return None
    return float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100.0)


def _adjusted_r2(r2_value, n_samples, n_features):
    if n_samples <= n_features + 1:
        return None
    return 1.0 - ((1.0 - r2_value) * (n_samples - 1) / (n_samples - n_features - 1))


def _regression_report(y_true, y_pred, n_features):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_samples = len(y_true_arr)
    r2_value = float(r2_score(y_true_arr, y_pred_arr))
    rmse_value = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    return {
        "r2": _round_or_none(r2_value),
        "mae": _round_or_none(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": _round_or_none(rmse_value),
        "adjusted_r2": _round_or_none(_adjusted_r2(r2_value, n_samples, n_features)),
        "mape": _round_or_none(_safe_mape(y_true_arr, y_pred_arr)),
    }


def _as_series(y):
    return y if isinstance(y, pd.Series) else pd.Series(y)


def _cross_validate_regression(pipe, X_data, y_data, cv):
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    }
    kwargs = {"estimator": pipe, "X": X_data, "y": y_data, "cv": cv, "scoring": scoring, "n_jobs": -1}
    try:
        scores = cross_validate(**kwargs)
    except PermissionError:
        scores = cross_validate(**{**kwargs, "n_jobs": 1})

    r2_value = float(scores["test_r2"].mean())
    n_samples = len(y_data)
    n_features = X_data.shape[1]
    return {
        "r2": _round_or_none(r2_value),
        "r2_std": _round_or_none(scores["test_r2"].std()),
        "mae": _round_or_none(-scores["test_mae"].mean()),
        "mae_std": _round_or_none(scores["test_mae"].std()),
        "rmse": _round_or_none(-scores["test_rmse"].mean()),
        "rmse_std": _round_or_none(scores["test_rmse"].std()),
        "adjusted_r2": _round_or_none(_adjusted_r2(r2_value, n_samples, n_features)),
        "mape": _round_or_none((-scores["test_mape"].mean()) * 100.0),
        "mape_std": _round_or_none(scores["test_mape"].std() * 100.0),
    }


def _train(model, df: pd.DataFrame = None, target: str = None, X=None, y=None, X_test=None, y_test=None, cv=None):
    """Train a regressor and return (pipeline, report)."""
    if X is not None and y is not None:
        y_series = _as_series(y)
        X_data = X
        pipe = Pipeline([("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X, y_series
            X_te = X_test
            y_te = _as_series(y_test)
            pipe.fit(X_tr, y_tr)
        elif cv is not None:
            report = _cross_validate_regression(pipe, X_data, y_series, cv)
            pipe.fit(X_data, y_series)
            return pipe, report
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)
            pipe.fit(X_tr, y_tr)
    else:
        check_df_target(df, target)
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X_df, y_df
            X_te = X_test
            y_te = _as_series(y_test)
            pipe.fit(X_tr, y_tr)
        elif cv is not None:
            report = _cross_validate_regression(pipe, X_df, y_df, cv)
            pipe.fit(X_df, y_df)
            return pipe, report
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
            pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    return pipe, _regression_report(y_te, pred, X_te.shape[1])


def _load_xgb_regressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install XGBoost support with: pip install breezeml[boost]") from exc
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbosity=0,
    )


def _load_lgbm_regressor(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install LightGBM support with: pip install breezeml[boost]") from exc
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        verbose=-1,
    )


def linear(df: pd.DataFrame = None, target: str = None, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(LinearRegression(), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def ridge(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(Ridge(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def lasso(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, max_iter: int = 5000, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(Lasso(alpha=alpha, max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def elastic_net(
    df: pd.DataFrame = None,
    target: str = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def svr(
    df: pd.DataFrame = None,
    target: str = None,
    kernel: str = "rbf",
    C: float = 1.0,
    epsilon: float = 0.1,
    gamma: str | float = "scale",
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def decision_tree(
    df: pd.DataFrame = None,
    target: str = None,
    random_state: int = 42,
    max_depth: int | None = None,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(DecisionTreeRegressor(random_state=random_state, max_depth=max_depth), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def random_forest(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(RandomForestRegressor(n_estimators=n_estimators, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def gradient_boosting(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def knn(df: pd.DataFrame = None, target: str = None, n_neighbors: int = 5, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(KNeighborsRegressor(n_neighbors=n_neighbors), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def mlp(
    df: pd.DataFrame = None,
    target: str = None,
    hidden_layer_sizes=(100,),
    max_iter: int = 500,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def xgboost(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        _load_xgb_regressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def lightgbm(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        _load_lgbm_regressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def _base_regressor_factories():
    return {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(),
        "Lasso": lambda: Lasso(max_iter=5000),
        "Elastic Net": lambda: ElasticNet(max_iter=5000),
        "SVR": lambda: SVR(),
        "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
        "KNN": lambda: KNeighborsRegressor(),
        "MLP (Neural Net)": lambda: MLPRegressor(max_iter=500, random_state=42),
    }


def _regressor_factories():
    factories = _base_regressor_factories()
    try:
        _load_xgb_regressor()
        factories["XGBoost"] = lambda: _load_xgb_regressor()
    except ImportError:
        pass
    try:
        _load_lgbm_regressor()
        factories["LightGBM"] = lambda: _load_lgbm_regressor()
    except ImportError:
        pass
    return factories


def compare(df: pd.DataFrame = None, target: str = None, show: bool = True, *, X=None, y=None, cv=None):
    if X is None or y is None:
        check_df_target(df, target)

    def _run_one(name, factory_func):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, report = _train(factory_func(), df=df, target=target, X=X, y=y, cv=cv)
            return {"regressor": name, **report}
        except Exception as exc:
            print(f"Warning: {name} failed with error: {exc}")
            return {"regressor": name, "r2": None, "mae": None, "rmse": None}

    tasks = [delayed(_run_one)(name, factory) for name, factory in _regressor_factories().items()]
    try:
        results = Parallel(n_jobs=-1)(tasks)
    except PermissionError:
        results = [_run_one(name, factory) for name, factory in _regressor_factories().items()]
    results.sort(key=lambda row: row["r2"] if row["r2"] is not None else float("-inf"), reverse=True)

    if show:
        print(f"\nBreezeML Regressor Leaderboard - target: '{target}'")
        if cv is None:
            print(f"{'Rank':<6}{'Regressor':<22}{'R2':<10}{'MAE':<10}{'RMSE':<10}")
            print("-" * 58)
            for i, row in enumerate(results, 1):
                r2_value = f"{row['r2']:.4f}" if row["r2"] is not None else "FAILED"
                mae_value = f"{row['mae']:.4f}" if row["mae"] is not None else "FAILED"
                rmse_value = f"{row['rmse']:.4f}" if row["rmse"] is not None else "FAILED"
                print(f"{i:<6}{row['regressor']:<22}{r2_value:<10}{mae_value:<10}{rmse_value:<10}")
        else:
            print(f"{'Rank':<6}{'Regressor':<22}{'R2':<22}{'MAE':<22}{'RMSE':<22}")
            print("-" * 92)
            for i, row in enumerate(results, 1):
                if row["r2"] is None:
                    r2_value = "FAILED"
                    mae_value = "FAILED"
                    rmse_value = "FAILED"
                else:
                    r2_value = f"{row['r2']:.4f} +/- {row['r2_std']:.4f}"
                    mae_value = f"{row['mae']:.4f} +/- {row['mae_std']:.4f}"
                    rmse_value = f"{row['rmse']:.4f} +/- {row['rmse_std']:.4f}"
                print(f"{i:<6}{row['regressor']:<22}{r2_value:<22}{mae_value:<22}{rmse_value:<22}")
        print()

    return results


def _algo_factories():
    factories = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(),
        "lasso": lambda: Lasso(max_iter=5000),
        "elastic_net": lambda: ElasticNet(max_iter=5000),
        "svr": lambda: SVR(),
        "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
        "random_forest": lambda: RandomForestRegressor(random_state=42),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
        "knn": lambda: KNeighborsRegressor(),
        "mlp": lambda: MLPRegressor(random_state=42, max_iter=500),
    }
    try:
        _load_xgb_regressor()
        factories["xgboost"] = lambda: _load_xgb_regressor()
    except ImportError:
        pass
    try:
        _load_lgbm_regressor()
        factories["lightgbm"] = lambda: _load_lgbm_regressor()
    except ImportError:
        pass
    return factories


def detailed_report(df: pd.DataFrame = None, target: str = None, model=None, algo: str = "random_forest", *, X=None, y=None):
    if X is None or y is None:
        check_df_target(df, target)

    algo_factories = _algo_factories()
    if X is not None and y is not None:
        y_series = _as_series(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)
        if model is None:
            factory = algo_factories.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")
            pipe = Pipeline([("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model
    else:
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
        if model is None:
            factory = algo_factories.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")
            pipe = Pipeline([("pre", pre), ("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model

    pred = pipe.predict(X_te)
    residuals = np.asarray(y_te) - np.asarray(pred)
    prediction_vs_actual = list(zip(np.asarray(y_te).tolist(), np.asarray(pred).tolist()))
    result = _regression_report(y_te, pred, X_te.shape[1])
    result.update(
        {
            "explained_variance": _round_or_none(explained_variance_score(y_te, pred)),
            "residuals": residuals,
            "prediction_vs_actual": prediction_vs_actual,
            "model": pipe,
        }
    )
    return result


_PARAM_GRIDS = {
    "linear": {"model__fit_intercept": [True, False]},
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
    "lasso": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "elastic_net": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "svr": {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["rbf", "linear"], "model__epsilon": [0.01, 0.1, 0.2], "model__gamma": ["scale", "auto"]},
    "decision_tree": {"model__max_depth": [3, 5, 10, 20, None], "model__min_samples_split": [2, 5, 10]},
    "random_forest": {"model__n_estimators": [100, 200, 500], "model__max_depth": [5, 10, 20, None], "model__min_samples_split": [2, 5, 10]},
    "gradient_boosting": {"model__n_estimators": [100, 200, 500], "model__learning_rate": [0.01, 0.1, 0.2], "model__max_depth": [2, 3, 5]},
    "knn": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean", "manhattan"]},
    "mlp": {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)], "model__learning_rate_init": [0.001, 0.01], "model__max_iter": [300, 500]},
    "xgboost": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [3, 5, 7, 10],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [100, 200, 500],
        "model__num_leaves": [31, 63, 127],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
}


def quick_tune(df: pd.DataFrame = None, target: str = None, algo: str = "random_forest", n_iter: int = 20, cv: int = 3, *, X=None, y=None):
    if X is None or y is None:
        check_df_target(df, target)

    algo_factories = _algo_factories()
    factory = algo_factories.get(algo)
    param_grid = _PARAM_GRIDS.get(algo)
    if factory is None or param_grid is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")

    if X is not None and y is not None:
        y_series = _as_series(y)
        pipe = Pipeline([("model", factory())])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)
    else:
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", factory())])
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    actual_iter = min(n_iter, grid_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_kwargs = {
            "estimator": pipe,
            "param_distributions": param_grid,
            "n_iter": actual_iter,
            "cv": cv,
            "scoring": "r2",
            "random_state": 42,
            "n_jobs": -1,
            "error_score": "raise",
        }
        search = RandomizedSearchCV(**search_kwargs)
        try:
            search.fit(X_tr, y_tr)
        except PermissionError:
            search = RandomizedSearchCV(**{**search_kwargs, "n_jobs": 1})
            search.fit(X_tr, y_tr)

    best_pipe = search.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    pred = best_pipe.predict(X_te)
    report = _regression_report(y_te, pred, X_te.shape[1])

    print(f"Best params for {algo}: {best_params}")
    print(f"   R2: {report['r2']}  |  MAE: {report['mae']}  |  RMSE: {report['rmse']}")

    return best_pipe, best_params, report
