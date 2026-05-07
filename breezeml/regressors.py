"""
BreezeML Regressors
Easy wrappers for popular regression algorithms with sensible preprocessing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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


def _regression_report(y_true, y_pred, n_features):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_samples = len(y_true_arr)

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mse))

    adjusted_r2 = None
    if n_samples > n_features + 1:
        adjusted_r2 = 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1))

    return {
        "r2": _round_or_none(r2),
        "mae": _round_or_none(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": _round_or_none(rmse),
        "adjusted_r2": _round_or_none(adjusted_r2),
        "mape": _round_or_none(_safe_mape(y_true_arr, y_pred_arr)),
    }


def _as_series(y):
    return y if isinstance(y, pd.Series) else pd.Series(y)


def _train(model, df: pd.DataFrame = None, target: str = None, X=None, y=None, X_test=None, y_test=None):
    """Train a regressor and return (pipeline, report)."""
    if X is not None and y is not None:
        y_series = _as_series(y)
        if X_test is not None and y_test is not None:
            X_tr, y_tr = X, y_series
            X_te = X_test
            y_te = _as_series(y_test)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)
        pipe = Pipeline([("model", model)])
        pipe.fit(X_tr, y_tr)
    else:
        check_df_target(df, target)
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X_df, y_df
            X_te = X_test
            y_te = _as_series(y_test)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    n_features = X_te.shape[1]
    return pipe, _regression_report(y_te, pred, n_features)


def linear(df: pd.DataFrame = None, target: str = None, *, X=None, y=None, X_test=None, y_test=None):
    """Linear Regression baseline."""
    return _train(LinearRegression(), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def ridge(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None):
    """Ridge Regression."""
    return _train(Ridge(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def lasso(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, max_iter: int = 5000, *, X=None, y=None, X_test=None, y_test=None):
    """Lasso Regression."""
    return _train(Lasso(alpha=alpha, max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


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
):
    """Elastic Net Regression."""
    return _train(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
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
):
    """Support Vector Regression."""
    return _train(SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


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
):
    """Decision Tree Regressor."""
    return _train(
        DecisionTreeRegressor(random_state=random_state, max_depth=max_depth),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


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
):
    """Random Forest Regressor."""
    return _train(
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


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
):
    """Gradient Boosting Regressor."""
    return _train(
        GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


def knn(df: pd.DataFrame = None, target: str = None, n_neighbors: int = 5, *, X=None, y=None, X_test=None, y_test=None):
    """K-Nearest Neighbors Regressor."""
    return _train(KNeighborsRegressor(n_neighbors=n_neighbors), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


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
):
    """Multi-Layer Perceptron Regressor."""
    return _train(
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


_REGRESSORS = {
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


def compare(df: pd.DataFrame = None, target: str = None, show: bool = True, *, X=None, y=None):
    """Run every regressor and return results sorted by R2."""
    if X is None or y is None:
        check_df_target(df, target)

    def _run_one(name, factory_func):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, report = _train(factory_func(), df=df, target=target, X=X, y=y)
            return {"regressor": name, **report}
        except Exception as exc:
            print(f"Warning: {name} failed with error: {exc}")
            return {"regressor": name, "r2": None, "mae": None, "rmse": None}

    try:
        results = Parallel(n_jobs=-1)(delayed(_run_one)(name, factory) for name, factory in _REGRESSORS.items())
    except PermissionError:
        results = [_run_one(name, factory) for name, factory in _REGRESSORS.items()]
    results.sort(key=lambda r: r["r2"] if r["r2"] is not None else float("-inf"), reverse=True)

    if show:
        print(f"\nBreezeML Regressor Leaderboard - target: '{target}'")
        print(f"{'Rank':<6}{'Regressor':<22}{'R2':<10}{'MAE':<10}{'RMSE':<10}")
        print("-" * 58)
        for i, result in enumerate(results, 1):
            r2_value = f"{result['r2']:.4f}" if result["r2"] is not None else "FAILED"
            mae_value = f"{result['mae']:.4f}" if result["mae"] is not None else "FAILED"
            rmse_value = f"{result['rmse']:.4f}" if result["rmse"] is not None else "FAILED"
            print(f"{i:<6}{result['regressor']:<22}{r2_value:<10}{mae_value:<10}{rmse_value:<10}")
        print()

    return results


_ALGO_FACTORIES = {
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


def detailed_report(df: pd.DataFrame = None, target: str = None, model=None, algo: str = "random_forest", *, X=None, y=None):
    """Return a detailed regression report."""
    if X is None or y is None:
        check_df_target(df, target)

    if X is not None and y is not None:
        y_series = _as_series(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)

        if model is None:
            factory = _ALGO_FACTORIES.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_FACTORIES.keys())}")
            pipe = Pipeline([("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model
    else:
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

        if model is None:
            factory = _ALGO_FACTORIES.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_FACTORIES.keys())}")
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
    "linear": {},
    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 50.0],
    },
    "lasso": {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
    "elastic_net": {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "svr": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
        "model__epsilon": [0.01, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 500],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [2, 3, 5],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "mlp": {
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "model__learning_rate_init": [0.001, 0.01],
        "model__max_iter": [300, 500],
    },
}


def quick_tune(df: pd.DataFrame = None, target: str = None, algo: str = "random_forest", n_iter: int = 20, cv: int = 3, *, X=None, y=None):
    """Auto-tune a regressor's hyperparameters in one line."""
    if X is None or y is None:
        check_df_target(df, target)

    factory = _ALGO_FACTORIES.get(algo)
    param_grid = _PARAM_GRIDS.get(algo)
    if factory is None or param_grid is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_FACTORIES.keys())}")

    if X is not None and y is not None:
        y_series = _as_series(y)
        pipe = Pipeline([("model", factory())])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42)
    else:
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", factory())])
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    actual_iter = min(n_iter, grid_size) if param_grid else 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_kwargs = {
            "estimator": pipe,
            "param_distributions": param_grid if param_grid else {"model__fit_intercept": [True, False]},
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
