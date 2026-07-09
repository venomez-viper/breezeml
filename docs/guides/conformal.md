# Conformal prediction: honest uncertainty on every prediction

Every point prediction a model makes is a guess with no error bars. Split
(inductive) conformal prediction fixes that. Given a model that is ALREADY
trained and a calibration set the model has never seen, it turns each guess
into a prediction interval (regression) or a prediction set (classification)
with a guaranteed marginal coverage of at least `1 - alpha` (alpha=0.1 gives
90% coverage: a 90% interval contains the truth about 90% of the time),
without any assumption about the shape of the data or the errors. This is the
honesty crown jewel of BreezeML: wrap any trained model and every prediction
starts carrying mathematically honest uncertainty.

The two entry points are `conformal_regressor()` and `conformal_classifier()`
(convenience wrappers around the `ConformalRegressor` and `ConformalClassifier`
classes).

## `conformal_regressor()`: prediction intervals around a regressor

```python
import breezeml
from breezeml import datasets, regressors, conformal
from sklearn.model_selection import train_test_split

df = datasets.diabetes()
train_df, rest = train_test_split(df, test_size=0.4, random_state=0)
calib_df, test_df = train_test_split(rest, test_size=0.5, random_state=0)

# 1. Train on TRAIN only.
model, _ = regressors.random_forest(train_df, "target")

# 2. Calibrate on held-out data the model has never seen.
cp = conformal.conformal_regressor(model, calib_df, "target", alpha=0.1)

# 3. Every prediction now carries an interval.
bands = cp.predict_interval(test_df.drop(columns=["target"]))
print(bands.head())          # lower / point / upper, one row per input row

# 4. Check that the promise held.
cp.coverage_report(test_df, "target")   # empirical coverage should land near 90%
```

`predict_interval(X)` returns a DataFrame with `lower`, `point`, and `upper`
columns. `point` is the model's own prediction; the band is `point +/- q`,
where `q` is the conformal quantile of the calibration residuals. The
nonconformity score for regression is the absolute residual `|y - yhat|`, and
`q` is the k-th smallest score with the finite-sample correction
`k = ceil((n + 1) * (1 - alpha))` over `n` calibration rows. That small bump
over the plain empirical quantile is what upgrades the guarantee from
"asymptotic" to "holds for your exact calibration size".

Pass `normalize=True` to `conformal_regressor()` for locally adaptive
(normalized) intervals: a small random forest estimates the per-row residual
magnitude so bands widen where the model is less certain and tighten where it
is confident. The calibration data is split in half so the difficulty model and
the quantile use disjoint rows, keeping the guarantee intact; with fewer than 8
calibration rows it warns and falls back to plain residuals.

You can also read a different coverage level off the same calibration without
refitting: `cp.predict_interval(X, alpha=0.05)` gives 95% intervals.

## `conformal_classifier()`: prediction sets around a classifier

```python
from breezeml import datasets, conformal
from sklearn.model_selection import train_test_split

df = datasets.breast_cancer()
train_df, rest = train_test_split(df, test_size=0.4, random_state=0)
calib_df, test_df = train_test_split(rest, test_size=0.5, random_state=0)

model, _ = breezeml.classify(train_df, "label")
cp = conformal.conformal_classifier(model, calib_df, "label", alpha=0.1)

sets = cp.predict_set(test_df.drop(columns=["label"]))
print(sets[:5])                         # a list of labels per row
cp.coverage_report(test_df, "label")    # true label lands in the set ~ 90% of the time
```

`predict_set(X)` returns a list (one entry per row) of the labels that survive
the conformal threshold: every class whose predicted probability is at least
`1 - q`. The score is the LAC (least-ambiguous / score) method,
`1 - predicted_probability_of_the_true_class`. An easy row gets a set of size
one; an ambiguous row gets a larger set that is honest about the model being
torn. Sets can occasionally be empty in edge cases, and the marginal coverage
guarantee still holds. The classifier must expose `predict_proba` (tree
ensembles, logistic regression, calibrated classifiers); a clear `TypeError`
is raised otherwise.

## `coverage_report()`: check the promise empirically

```python
result = cp.coverage_report(test_df, "target")
print(result["target_coverage"], result["empirical_coverage"], result["well_calibrated"])
```

Both classes expose `coverage_report(df, target)`. For regression it returns
`target_coverage`, `empirical_coverage`, `mean_interval_width`, `alpha`,
`n_eval`, `n_calibration`, `normalized`, and `well_calibrated`. For
classification it swaps `mean_interval_width` for `mean_set_size` and adds
`n_classes`. `well_calibrated` is True when empirical coverage is within 0.05
of the target. When it is not, the report prints a warning and raises a
Python warning: the calibration set was probably too small, or the test data
is not exchangeable with it (drift).

## When NOT to use it

- **Coverage is marginal, not conditional.** Across many rows about `1 - alpha`
  are covered, but coverage is NOT guaranteed inside any particular subgroup (a
  rare class, a tail region, one hospital). A model that is badly wrong on a
  minority slice can still pass the marginal test by being extra tight
  elsewhere. Do not read these intervals as a per-group promise.
- **The guarantee needs exchangeability.** Under distribution drift, time
  ordering, or a train/calibration/test population mismatch the assumption
  breaks and coverage is no longer guaranteed. Recalibrate on fresh data.
- **Small calibration sets make the quantile noisy.** With very few rows the
  finite-sample correction can even demand an infinite quantile (an interval or
  set covering everything); the coverage report warns when `n` is too small to
  trust.
- **It quantifies uncertainty, it does not reduce it.** Conformal prediction
  wraps whatever model you give it; a weak model yields honest but wide bands.
  Improve the model to tighten them.
- **The calibration data must be held out from training.** Calibrate on the
  model's own training rows and the scores are optimistically small, the
  intervals too narrow, and real coverage falls below `1 - alpha`. Always pass
  a DataFrame the model has never seen.
