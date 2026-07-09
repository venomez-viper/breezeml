"""
BreezeML conformal prediction: honest, distribution-free uncertainty.

Every point prediction a model makes is a guess with no error bars. Split
(inductive) conformal prediction fixes that: given an ALREADY-TRAINED model
and a calibration set that the model has NEVER seen, it turns each guess
into a prediction interval (regression) or a prediction set (classification)
that is guaranteed to contain the truth with probability at least 1 - alpha,
WITHOUT any assumption about the shape of the data or the errors.

    from breezeml import conformal, regressors

    model, _ = regressors.random_forest(train_df, "y")     # train on TRAIN
    cp = conformal.conformal_regressor(model, calib_df, "y", alpha=0.1)
    bands = cp.predict_interval(new_X)      # lower / point / upper per row
    cp.coverage_report(test_df, "y")        # empirical coverage ~ 90%

How it works
------------
1. Predict on the calibration rows and score each one by how "wrong" the
   model was (the nonconformity score). For regression the score is the
   absolute residual ``|y - yhat|``; for classification it is
   ``1 - predicted_probability_of_the_true_class`` (the LAC / score method).
2. Take the conformal quantile of those scores with the finite-sample
   correction ``ceil((n + 1) * (1 - alpha)) / n`` (the k-th smallest score,
   k = ceil((n + 1)(1 - alpha))). This tiny bump over the plain empirical
   quantile is what upgrades the guarantee from "asymptotic" to "holds for
   your exact calibration size n".
3. At prediction time, widen every point prediction by that quantile:
   an interval ``yhat +/- q`` for regression, or the set of every label
   whose score passes the threshold for classification.

The calibration data MUST be held out from training
----------------------------------------------------
The whole guarantee rests on the calibration rows being EXCHANGEABLE with
the future test rows, i.e. drawn the same way and never used to fit the
model. If you calibrate on the model's own training data the scores are
optimistically small, the interval is too narrow, and real coverage falls
below 1 - alpha. Pass ``.fit`` a dataframe that the model has not seen.

When NOT to use it
------------------
- Coverage is MARGINAL, not conditional. Across many rows about 1 - alpha
  are covered, but coverage is NOT guaranteed inside any particular
  subgroup (a rare class, a tail region, one hospital). A model that is
  badly wrong on a minority slice can still pass the marginal test by being
  extra tight elsewhere. Do not read these intervals as a per-group promise.
- The guarantee needs exchangeability. Under distribution drift, time
  ordering, or a train/calibration/test population mismatch the assumption
  breaks and coverage is no longer guaranteed. Recalibrate on fresh data.
- Small calibration sets make the quantile noisy. With very few rows the
  correction can even demand an infinite quantile (an interval covering
  everything); the coverage report warns when n is too small to trust.
- It quantifies uncertainty, it does not reduce it. Conformal prediction
  wraps whatever model you give it; a weak model yields honest but wide
  bands. Improve the model to tighten them.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ._preprocessing import _build_preprocessor
from ._validation import check_df_target

__all__ = [
    "ConformalRegressor",
    "ConformalClassifier",
    "conformal_regressor",
    "conformal_classifier",
]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _unwrap(model):
    """Return the underlying sklearn estimator/pipeline.

    BreezeML EasyModel objects hold their fitted pipeline on ``.pipeline``;
    a bare sklearn estimator or pipeline is returned unchanged.
    """
    return getattr(model, "pipeline", model)


def _check_alpha(alpha) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
    return alpha


def _conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores.

    Returns the k-th smallest score where ``k = ceil((n + 1) * (1 - alpha))``.
    When k exceeds n (calibration set too small for this alpha) there is no
    finite score high enough to guarantee coverage, so the quantile is
    +inf, which produces an interval/set that covers everything.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = scores.size
    if n == 0:
        raise ValueError("Cannot compute a conformal quantile from 0 scores.")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def _to_native(value):
    """Convert a numpy scalar label to a plain Python type for clean output."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
class ConformalRegressor:
    """Distribution-free prediction intervals around a trained regressor.

    Wrap an already-trained BreezeML regressor, EasyModel, or sklearn
    pipeline and calibrate it on held-out data to get intervals with a
    guaranteed marginal coverage of at least ``1 - alpha``.
    """

    def __init__(self):
        self.pipeline = None
        self.target = None
        self.alpha = None
        self.normalize = False
        self.q = None
        self.scores_ = None
        self.n_calibration_ = 0
        self.difficulty_ = None
        self._sigma_floor = 1e-6
        self.fitted_ = False

    # -- fitting ----------------------------------------------------------- #
    def fit(self, model, df: pd.DataFrame, target: str, alpha: float = 0.1,
            normalize: bool = False):
        """Calibrate on ``df`` (which the model must NOT have been trained on).

        Parameters
        ----------
        model : trained BreezeML regressor / EasyModel / sklearn pipeline
            Only its ``predict`` is used; it is never refit.
        df : pd.DataFrame
            Calibration data INCLUDING the target column. Must be rows the
            model has never seen, or the intervals will be too narrow.
        target : str
            Target column name.
        alpha : float
            Miscoverage rate; coverage is guaranteed at ``>= 1 - alpha``
            (alpha=0.1 -> 90% intervals). Default 0.1.
        normalize : bool
            When True, scale residuals by a per-row difficulty estimate so
            intervals are wider where the model is less certain (locally
            adaptive / normalized conformal). A small random forest predicts
            the residual magnitude; the calibration data is split in half so
            the difficulty model and the quantile use disjoint rows, keeping
            the coverage guarantee intact. Falls back to plain residuals when
            there are too few rows to split.
        """
        check_df_target(df, target)
        self.alpha = _check_alpha(alpha)
        self.pipeline = _unwrap(model)
        if not hasattr(self.pipeline, "predict"):
            raise TypeError(
                "ConformalRegressor needs a trained model with a .predict "
                "method (a BreezeML regressor, EasyModel, or sklearn pipeline)."
            )
        self.target = target
        X = df.drop(columns=[target])
        y = df[target].to_numpy(dtype=float)

        use_normalize = bool(normalize) and len(X) >= 8
        if bool(normalize) and not use_normalize:
            warnings.warn(
                "normalize=True needs at least 8 calibration rows to split "
                "safely; falling back to plain (unnormalized) residuals.",
                stacklevel=2,
            )
        self.normalize = use_normalize

        if use_normalize:
            idx = np.arange(len(X))
            fit_idx, cal_idx = train_test_split(idx, test_size=0.5, random_state=42)
            X_fit, y_fit = X.iloc[fit_idx], y[fit_idx]
            resid_fit = np.abs(y_fit - np.asarray(self.pipeline.predict(X_fit), dtype=float))
            self._fit_difficulty(X_fit, resid_fit)
            X_cal, y_cal = X.iloc[cal_idx], y[cal_idx]
            resid_cal = np.abs(y_cal - np.asarray(self.pipeline.predict(X_cal), dtype=float))
            scores = resid_cal / self._sigma(X_cal)
            self.n_calibration_ = len(cal_idx)
        else:
            resid = np.abs(y - np.asarray(self.pipeline.predict(X), dtype=float))
            scores = resid
            self.n_calibration_ = len(X)

        self.scores_ = np.sort(scores)
        self.q = _conformal_quantile(scores, self.alpha)
        self.fitted_ = True
        return self

    def _fit_difficulty(self, X: pd.DataFrame, resid: np.ndarray):
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
        pre = _build_preprocessor(numeric, categorical)
        pipe = Pipeline([
            ("pre", pre),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, resid)
        self.difficulty_ = pipe
        # floor keeps sigma strictly positive so scores never blow up
        self._sigma_floor = max(float(np.mean(resid)) * 0.1, 1e-6)

    def _sigma(self, X) -> np.ndarray:
        raw = np.asarray(self.difficulty_.predict(X), dtype=float)
        return np.maximum(raw, self._sigma_floor)

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Call .fit(model, df, target) before predicting.")

    # -- prediction -------------------------------------------------------- #
    def predict_interval(self, X, alpha: float | None = None) -> pd.DataFrame:
        """Prediction intervals for the rows in ``X``.

        Returns a DataFrame with ``lower``, ``point`` and ``upper`` columns
        (one row per input row). ``point`` is the model's own prediction;
        the band is ``point +/- q`` (scaled by local difficulty when the
        regressor was fitted with ``normalize=True``). Pass ``alpha`` to get
        a different coverage level from the same calibration without refit.
        """
        self._require_fitted()
        if alpha is None:
            q = self.q
        else:
            q = _conformal_quantile(self.scores_, _check_alpha(alpha))
        point = np.asarray(self.pipeline.predict(X), dtype=float)
        if self.normalize:
            half = q * self._sigma(X)
        else:
            half = np.full(point.shape, q, dtype=float)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(
            {"lower": point - half, "point": point, "upper": point + half},
            index=index,
        )

    # -- evaluation -------------------------------------------------------- #
    def coverage_report(self, df: pd.DataFrame, target: str | None = None,
                        show: bool = True) -> dict:
        """Empirical coverage and mean interval width on a labelled test set.

        Coverage should land near ``1 - alpha``. A large gap usually means
        the calibration set was too small or the test data is not
        exchangeable with it (drift); the report warns when that happens.
        """
        self._require_fitted()
        target = target or self.target
        check_df_target(df, target)
        X = df.drop(columns=[target])
        y = df[target].to_numpy(dtype=float)

        bands = self.predict_interval(X)
        lower = bands["lower"].to_numpy()
        upper = bands["upper"].to_numpy()
        covered = (y >= lower) & (y <= upper)
        coverage = float(np.mean(covered)) if len(y) else 0.0
        width = float(np.mean(upper - lower)) if len(y) else 0.0
        target_cov = 1.0 - self.alpha
        off = abs(coverage - target_cov)

        result = {
            "target_coverage": round(target_cov, 4),
            "empirical_coverage": round(coverage, 4),
            "mean_interval_width": round(width, 4) if math.isfinite(width) else float("inf"),
            "alpha": self.alpha,
            "n_eval": int(len(y)),
            "n_calibration": int(self.n_calibration_),
            "normalized": self.normalize,
            "well_calibrated": bool(off <= 0.05),
        }
        if show:
            print(f"\nBreezeML Conformal Coverage - target '{target}' (regression)")
            print("-" * 64)
            print(f"  target coverage : {target_cov:.1%}  (alpha = {self.alpha})")
            print(f"  empirical       : {coverage:.1%} on {len(y):,} rows")
            print(f"  mean width      : {result['mean_interval_width']}")
            print(f"  calibrated on   : {self.n_calibration_:,} held-out rows")
            if not result["well_calibrated"]:
                print(
                    "  WARNING: empirical coverage is far from target. The "
                    "calibration set may be too small, or the test data may "
                    "not be exchangeable with it (drift)."
                )
            print("-" * 64)
        if not result["well_calibrated"]:
            warnings.warn(
                f"Empirical coverage {coverage:.3f} is far from the target "
                f"{target_cov:.3f}; calibration set may be too small or data "
                "has drifted.",
                stacklevel=2,
            )
        return result


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
class ConformalClassifier:
    """Distribution-free prediction sets around a trained classifier.

    Uses the LAC / score method: the nonconformity score of a row is
    ``1 - predicted_probability_of_the_true_class``. Each prediction set
    contains every label whose score passes the conformal threshold, so the
    true label lands in the set at least ``1 - alpha`` of the time.
    """

    def __init__(self):
        self.pipeline = None
        self.target = None
        self.alpha = None
        self.q = None
        self.scores_ = None
        self.classes_ = None
        self.n_calibration_ = 0
        self.fitted_ = False

    def fit(self, model, df: pd.DataFrame, target: str, alpha: float = 0.1):
        """Calibrate on ``df`` (which the model must NOT have been trained on).

        The model must expose ``predict_proba`` (tree ensembles, logistic
        regression, calibrated classifiers, ...); a clear TypeError is raised
        otherwise.
        """
        check_df_target(df, target)
        self.alpha = _check_alpha(alpha)
        self.pipeline = _unwrap(model)
        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError(
                "ConformalClassifier needs a model with predict_proba "
                "(tree ensembles, logistic regression, calibrated models). "
                "The given model does not expose predicted probabilities."
            )
        self.target = target
        X = df.drop(columns=[target])
        y = df[target].to_numpy()

        proba = np.asarray(self.pipeline.predict_proba(X), dtype=float)
        classes = list(self.pipeline.classes_)
        self.classes_ = np.asarray(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        unknown = set(np.unique(y).tolist()) - set(class_to_idx)
        if unknown:
            raise ValueError(
                f"Calibration labels {sorted(unknown)} were never seen by the "
                "model; calibrate with the same label set the model was "
                "trained on."
            )

        true_idx = np.array([class_to_idx[v] for v in y])
        true_proba = proba[np.arange(len(y)), true_idx]
        scores = 1.0 - true_proba

        self.scores_ = np.sort(scores)
        self.q = _conformal_quantile(scores, self.alpha)
        self.n_calibration_ = len(y)
        self.fitted_ = True
        return self

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Call .fit(model, df, target) before predicting.")

    def predict_set(self, X, alpha: float | None = None) -> list:
        """Prediction sets for the rows in ``X``.

        Returns a list (one entry per row) of lists of labels: every class
        whose score ``1 - proba`` is at most the conformal quantile ``q``,
        i.e. every class with ``proba >= 1 - q``. Sets can occasionally be
        empty in edge cases; the marginal coverage guarantee still holds.
        Pass ``alpha`` for a different coverage level without refitting.
        """
        self._require_fitted()
        if alpha is None:
            q = self.q
        else:
            q = _conformal_quantile(self.scores_, _check_alpha(alpha))
        proba = np.asarray(self.pipeline.predict_proba(X), dtype=float)
        sets = []
        for row in proba:
            keep = [
                _to_native(self.classes_[j])
                for j in range(len(self.classes_))
                if (1.0 - row[j]) <= q
            ]
            sets.append(keep)
        return sets

    def coverage_report(self, df: pd.DataFrame, target: str | None = None,
                        show: bool = True) -> dict:
        """Empirical coverage and mean set size on a labelled test set.

        Coverage should land near ``1 - alpha``. A large gap usually means
        the calibration set was too small or the test data is not
        exchangeable with it (drift); the report warns when that happens.
        """
        self._require_fitted()
        target = target or self.target
        check_df_target(df, target)
        X = df.drop(columns=[target])
        y = df[target].to_numpy()

        sets = self.predict_set(X)
        covered = [
            _to_native(y_i) in set_i for y_i, set_i in zip(y, sets)
        ]
        coverage = float(np.mean(covered)) if len(y) else 0.0
        mean_size = float(np.mean([len(s) for s in sets])) if sets else 0.0
        target_cov = 1.0 - self.alpha
        off = abs(coverage - target_cov)

        result = {
            "target_coverage": round(target_cov, 4),
            "empirical_coverage": round(coverage, 4),
            "mean_set_size": round(mean_size, 4),
            "alpha": self.alpha,
            "n_eval": int(len(y)),
            "n_calibration": int(self.n_calibration_),
            "n_classes": int(len(self.classes_)),
            "well_calibrated": bool(off <= 0.05),
        }
        if show:
            print(f"\nBreezeML Conformal Coverage - target '{target}' (classification)")
            print("-" * 64)
            print(f"  target coverage : {target_cov:.1%}  (alpha = {self.alpha})")
            print(f"  empirical       : {coverage:.1%} on {len(y):,} rows")
            print(f"  mean set size   : {result['mean_set_size']} of {len(self.classes_)} classes")
            print(f"  calibrated on   : {self.n_calibration_:,} held-out rows")
            if not result["well_calibrated"]:
                print(
                    "  WARNING: empirical coverage is far from target. The "
                    "calibration set may be too small, or the test data may "
                    "not be exchangeable with it (drift)."
                )
            print("-" * 64)
        if not result["well_calibrated"]:
            warnings.warn(
                f"Empirical coverage {coverage:.3f} is far from the target "
                f"{target_cov:.3f}; calibration set may be too small or data "
                "has drifted.",
                stacklevel=2,
            )
        return result


# --------------------------------------------------------------------------- #
# convenience functions
# --------------------------------------------------------------------------- #
def conformal_regressor(model, df: pd.DataFrame, target: str, alpha: float = 0.1,
                        normalize: bool = False) -> ConformalRegressor:
    """Fit and return a :class:`ConformalRegressor` in one call.

    ``model`` must already be trained; ``df`` must be calibration data the
    model has NOT seen. See :meth:`ConformalRegressor.fit`.
    """
    cp = ConformalRegressor()
    cp.fit(model, df, target, alpha=alpha, normalize=normalize)
    return cp


def conformal_classifier(model, df: pd.DataFrame, target: str,
                         alpha: float = 0.1) -> ConformalClassifier:
    """Fit and return a :class:`ConformalClassifier` in one call.

    ``model`` must already be trained and expose ``predict_proba``; ``df``
    must be calibration data the model has NOT seen. See
    :meth:`ConformalClassifier.fit`.
    """
    cp = ConformalClassifier()
    cp.fit(model, df, target, alpha=alpha)
    return cp
