# Time series: forecasting with a built-in honesty check

`breezeml.timeseries` turns a univariate series into a supervised learning
problem and validates the way forecasts must be validated: walk-forward,
never shuffled, always against the naive baseline.

```python
from breezeml import timeseries

# Leaderboard: which model actually forecasts your series?
results = timeseries.compare(df, "sales", date_col="date")

# Train + predict 14 steps ahead
model, forecast, report = timeseries.forecast(df, "sales", horizon=14, date_col="date")
print(forecast)
print(report["beats_naive"], report["skill_vs_naive"])
```

## The naive baseline is always in the room

Every leaderboard includes **naive (last value)**: tomorrow = today. A
model that cannot beat it is not forecasting, it is decorating. `forecast()`
prints a warning when your chosen model loses to naive, and the report
carries `beats_naive` and `skill_vs_naive` (1.0 = perfect, 0 = no better
than naive, negative = worse).

## How it works

1. `make_features()` builds lag columns (`lag_1`, `lag_7`, ...), rolling
   means/stds computed on *past values only* (shifted, leakage-free), and
   calendar features (day of week, month) when `date_col` is given.
2. Validation uses sklearn's `TimeSeriesSplit`: train on the past, test on
   the future, repeated across folds.
3. `forecast()` refits on the full series and rolls forward recursively:
   each prediction becomes an input for the next step.

Models: linear, ridge, random forest, gradient boosting, KNN, plus XGBoost
and LightGBM when installed. All on the 4 core dependencies.

## When NOT to use it

- Multi-seasonal or holiday-heavy series (retail with promotions): a
  dedicated library (statsforecast, prophet) will do better.
- Very short series (< 60 points): there is not enough history for lag
  features plus walk-forward validation to mean anything.
- Multivariate forecasting with exogenous drivers: not supported in v1.2.
