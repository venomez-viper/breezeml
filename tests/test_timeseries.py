"""Tests for breezeml.timeseries (v1.2)."""
import numpy as np
import pandas as pd
import pytest

from breezeml import timeseries


def _synthetic(n=200, seed=42, dates=True):
    """Weekly-seasonal series with trend and noise: learnable, not trivial."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 100 + 0.5 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n)
    if dates:
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "sales": values,
        })
    return pd.DataFrame({"sales": values})


# ----------------------------------------------------------- features

def test_make_features_lags_and_calendar():
    df = _synthetic()
    table = timeseries.make_features(df, "sales", date_col="date")
    assert "lag_1" in table.columns
    assert "lag_7" in table.columns
    assert "roll_mean_7" in table.columns
    assert "dayofweek" in table.columns
    assert not table.isna().any().any()


def test_make_features_no_leakage():
    """roll_mean_7 at time t must not include the value at time t."""
    df = _synthetic(n=50)
    table = timeseries.make_features(df, "sales", date_col="date", lags=(1,), windows=(7,))
    idx = table.index[10]
    pos = df.set_index("date")["sales"]
    expected = pos.shift(1).rolling(7).mean().loc[idx]
    assert abs(table.loc[idx, "roll_mean_7"] - expected) < 1e-9


def test_missing_values_rejected():
    df = _synthetic(n=60)
    df.loc[5, "sales"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        timeseries.make_features(df, "sales", date_col="date")


# ------------------------------------------------------------ compare

def test_compare_includes_naive_and_ranks(capsys):
    df = _synthetic()
    results = timeseries.compare(df, "sales", date_col="date", show=True)
    names = [r["model"] for r in results]
    assert any(n.startswith("naive") for n in names)
    maes = [r["mae"] for r in results if r.get("mae") is not None]
    assert maes == sorted(maes)
    out = capsys.readouterr().out
    assert "Forecast Leaderboard" in out


def test_compare_models_beat_naive_on_seasonal_data():
    df = _synthetic()
    results = timeseries.compare(df, "sales", date_col="date", show=False)
    best = results[0]
    assert not best["model"].startswith("naive"), "a real model should win on seasonal data"
    assert best["beats_naive"] is True
    assert best["skill_vs_naive"] > 0.2


def test_compare_too_short_series():
    df = _synthetic(n=20)
    with pytest.raises(ValueError, match="too short"):
        timeseries.compare(df, "sales", date_col="date", show=False)


# ----------------------------------------------------------- forecast

def test_forecast_horizon_and_index():
    df = _synthetic()
    model, preds, report = timeseries.forecast(df, "sales", horizon=14, date_col="date")
    assert len(preds) == 14
    assert preds.index[0] == pd.Timestamp("2025-01-01") + pd.Timedelta(days=200)
    assert report["beats_naive"] is True
    assert report["algo"] == "gradient_boosting"


def test_forecast_without_dates():
    df = _synthetic(dates=False)
    model, preds, report = timeseries.forecast(df, "sales", horizon=5)
    assert len(preds) == 5
    assert "mae" in report


def test_forecast_values_sane():
    """Forecasts should stay in the neighborhood of the series."""
    df = _synthetic()
    _, preds, _ = timeseries.forecast(df, "sales", horizon=14, date_col="date")
    assert preds.between(100, 350).all(), f"wild forecast values: {preds.values}"


def test_forecast_unknown_algo():
    df = _synthetic()
    with pytest.raises(ValueError, match="Unknown algo"):
        timeseries.forecast(df, "sales", algo="prophet", date_col="date")
