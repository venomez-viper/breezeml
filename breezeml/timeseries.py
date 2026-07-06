"""
BreezeML time series: forecasting for analysts, on the 4 core dependencies.

Turns a univariate series into a supervised learning problem (lag and
rolling-window features), validates with walk-forward cross-validation,
and always reports whether a model actually beats the naive
last-value baseline - the honesty check most forecasting demos skip.

    from breezeml import timeseries

    results = timeseries.compare(df, "sales", date_col="date")
    model, forecast, report = timeseries.forecast(df, "sales", horizon=14, date_col="date")
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor

from ._progress import ProgressBar
from ._validation import check_df_target

__all__ = ["make_features", "compare", "forecast"]


def _default_lags(n: int) -> tuple:
    lags = [lag for lag in (1, 2, 3, 7, 14, 28) if lag < n // 3]
    return tuple(lags) if lags else (1,)


def _prepare_series(df: pd.DataFrame, target: str, date_col: str | None) -> pd.Series:
    check_df_target(df, target)
    data = df.copy()
    if date_col is not None:
        if date_col not in data.columns:
            raise ValueError(f"date_col '{date_col}' not found in DataFrame.")
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col)
        series = data.set_index(date_col)[target]
    else:
        series = data[target]
    series = series.astype(float)
    if series.isna().any():
        raise ValueError(
            f"Target '{target}' contains missing values; fill or drop them first "
            "(forecasting needs a continuous series)."
        )
    return series


def make_features(
    df: pd.DataFrame,
    target: str,
    date_col: str | None = None,
    lags: tuple | None = None,
    windows: tuple = (7,),
) -> pd.DataFrame:
    """Build a supervised feature table from a univariate series.

    Adds lag columns (``lag_1``, ``lag_7``, ...), rolling means over past
    values (``roll_mean_7``, ...; shifted so no leakage), and calendar
    features when ``date_col`` is given. Rows without full history are
    dropped.
    """
    series = _prepare_series(df, target, date_col)
    lags = lags or _default_lags(len(series))

    out = pd.DataFrame({target: series})
    for lag in lags:
        out[f"lag_{lag}"] = series.shift(lag)
    for window in windows:
        if window >= len(series):
            continue
        out[f"roll_mean_{window}"] = series.shift(1).rolling(window).mean()
        out[f"roll_std_{window}"] = series.shift(1).rolling(window).std()

    if date_col is not None:
        idx = out.index
        out["dayofweek"] = idx.dayofweek
        out["month"] = idx.month
        out["dayofyear"] = idx.dayofyear

    return out.dropna()


def _model_factories():
    factories = {
        "naive (last value)": None,  # handled specially
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
        "knn": lambda: KNeighborsRegressor(n_neighbors=5),
    }
    try:
        from xgboost import XGBRegressor

        factories["xgboost"] = lambda: XGBRegressor(
            n_estimators=200, random_state=42, verbosity=0
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        factories["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=200, random_state=42, verbose=-1
        )
    except ImportError:
        pass
    return factories


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    non_zero = np.abs(y_true) > 1e-12
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        if non_zero.any()
        else None
    )
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2) if mape is not None else None}


def compare(
    df: pd.DataFrame,
    target: str,
    date_col: str | None = None,
    lags: tuple | None = None,
    windows: tuple = (7,),
    cv: int = 3,
    show: bool = True,
    progress: bool | None = None,
) -> list:
    """Walk-forward leaderboard of forecasting models vs the naive baseline.

    Every model is evaluated with sklearn's ``TimeSeriesSplit`` (train on
    the past, test on the future - never shuffled). The naive last-value
    baseline is always included: a model that cannot beat it is not
    forecasting, it is decorating.
    """
    table = make_features(df, target, date_col=date_col, lags=lags, windows=windows)
    X = table.drop(columns=[target])
    y = table[target]
    if len(table) < (cv + 1) * 5:
        raise ValueError(
            f"Series too short after feature building ({len(table)} usable rows) "
            f"for {cv}-fold walk-forward validation."
        )

    if progress is None:
        progress = show
    factories = _model_factories()
    bar = ProgressBar(len(factories), desc="Walk-forward validation", enabled=progress)
    splitter = TimeSeriesSplit(n_splits=cv)
    results = []
    for name, factory in factories.items():
        bar.update(name)
        fold_true, fold_pred = [], []
        try:
            for train_idx, test_idx in splitter.split(X):
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                if factory is None:  # naive: predict the last known value forward
                    pred = np.full(len(test_idx), y_tr.iloc[-1])
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = factory()
                        model.fit(X.iloc[train_idx], y_tr)
                        pred = model.predict(X.iloc[test_idx])
                fold_true.extend(y_te.tolist())
                fold_pred.extend(np.asarray(pred).tolist())
            row = {"model": name, **_metrics(fold_true, fold_pred)}
        except Exception as exc:
            row = {"model": name, "mae": None, "error": str(exc)[:120]}
        results.append(row)
    bar.close()

    naive_mae = next(r["mae"] for r in results if r["model"].startswith("naive"))
    for row in results:
        if row.get("mae") is not None and naive_mae:
            row["skill_vs_naive"] = round(1 - row["mae"] / naive_mae, 4)
            row["beats_naive"] = bool(row["mae"] < naive_mae)

    results.sort(key=lambda r: r["mae"] if r.get("mae") is not None else float("inf"))

    if show:
        print(f"\nBreezeML Forecast Leaderboard - target: '{target}' (walk-forward, {cv} folds)")
        print(f"{'Rank':<6}{'Model':<22}{'MAE':<12}{'RMSE':<12}{'Beats naive?':<12}")
        print("-" * 64)
        for i, row in enumerate(results, 1):
            if row.get("mae") is None:
                print(f"{i:<6}{row['model']:<22}{'FAILED':<12}")
                continue
            beats = "baseline" if row["model"].startswith("naive") else ("yes" if row["beats_naive"] else "NO")
            print(f"{i:<6}{row['model']:<22}{row['mae']:<12}{row['rmse']:<12}{beats:<12}")
        if results and results[0]["model"].startswith("naive"):
            print("The naive baseline won. Your series may be a random walk - no amount")
            print("of gradient boosting outsmarts a coin. Consider more history or exogenous data.")
        print()

    return results


class ForecastModel:
    """A fitted forecaster that can roll predictions forward recursively."""

    def __init__(self, estimator, target, lags, windows, date_col, history, freq):
        self.estimator = estimator
        self.target = target
        self.lags = lags
        self.windows = windows
        self.date_col = date_col
        self.history = history  # pd.Series of the full training series
        self.freq = freq
        self.task = "forecast"

    def _row_features(self, series: pd.Series, timestamp) -> dict:
        feats = {}
        values = series.values
        for lag in self.lags:
            feats[f"lag_{lag}"] = values[-lag]
        for window in self.windows:
            if window >= len(values) + 1:
                continue
            past = values[-window:]
            feats[f"roll_mean_{window}"] = float(np.mean(past))
            feats[f"roll_std_{window}"] = float(np.std(past, ddof=1)) if window > 1 else 0.0
        if self.date_col is not None and timestamp is not None:
            feats["dayofweek"] = timestamp.dayofweek
            feats["month"] = timestamp.month
            feats["dayofyear"] = timestamp.dayofyear
        return feats

    def forecast(self, horizon: int) -> pd.Series:
        """Recursive multi-step forecast: each prediction feeds the next."""
        series = self.history.copy()
        timestamps = []
        preds = []
        for step in range(horizon):
            if self.date_col is not None:
                next_ts = series.index[-1] + self.freq
            else:
                next_ts = series.index[-1] + 1
            feats = self._row_features(series, next_ts if self.date_col else None)
            X_next = pd.DataFrame([feats])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = float(self.estimator.predict(X_next)[0])
            preds.append(value)
            timestamps.append(next_ts)
            series = pd.concat([series, pd.Series([value], index=[next_ts])])
        return pd.Series(preds, index=timestamps, name=f"{self.target}_forecast")


def forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = 7,
    date_col: str | None = None,
    algo: str = "gradient_boosting",
    lags: tuple | None = None,
    windows: tuple = (7,),
    cv: int = 3,
):
    """Train a forecaster and predict ``horizon`` steps ahead.

    Returns
    -------
    (ForecastModel, pd.Series, dict)
        The fitted model, the forecast series, and a report with
        walk-forward metrics including ``beats_naive``.
    """
    factories = _model_factories()
    factory = factories.get(algo)
    if factory is None:
        valid = [k for k in factories if factories[k] is not None]
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {valid}")

    series = _prepare_series(df, target, date_col)
    lags = lags or _default_lags(len(series))
    table = make_features(df, target, date_col=date_col, lags=lags, windows=windows)
    X = table.drop(columns=[target])
    y = table[target]

    # Walk-forward validation for the chosen algo + naive baseline.
    splitter = TimeSeriesSplit(n_splits=cv)
    fold_true, fold_pred, naive_pred = [], [], []
    for train_idx, test_idx in splitter.split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = factory()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
        fold_true.extend(y.iloc[test_idx].tolist())
        fold_pred.extend(np.asarray(pred).tolist())
        naive_pred.extend([y.iloc[train_idx].iloc[-1]] * len(test_idx))

    metrics = _metrics(fold_true, fold_pred)
    naive = _metrics(fold_true, naive_pred)
    report = {
        **metrics,
        "naive_mae": naive["mae"],
        "skill_vs_naive": round(1 - metrics["mae"] / naive["mae"], 4) if naive["mae"] else None,
        "beats_naive": bool(metrics["mae"] < naive["mae"]) if naive["mae"] else None,
        "algo": algo,
        "lags": list(lags),
        "cv_folds": cv,
        "n_observations": int(len(series)),
    }

    # Refit on everything, then roll forward.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = factory()
        final.fit(X, y)

    freq = None
    if date_col is not None:
        freq = pd.infer_freq(series.index)
        freq = pd.tseries.frequencies.to_offset(freq) if freq else (series.index[-1] - series.index[-2])

    model = ForecastModel(final, target, lags, windows, date_col, series, freq)
    predictions = model.forecast(horizon)

    if report["beats_naive"] is False:
        print(
            f"Warning: {algo} did not beat the naive last-value baseline "
            f"(MAE {metrics['mae']} vs {naive['mae']}). Treat this forecast with suspicion."
        )

    return model, predictions, report
