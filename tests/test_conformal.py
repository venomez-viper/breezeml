"""Tests for breezeml.conformal (v1.9): distribution-free prediction
intervals and sets with a marginal coverage guarantee."""
import numpy as np
import pytest

import breezeml
from breezeml import datasets


def _splits(df, a, b):
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


# ---------------------------------------------------------- regression

def test_conformal_regression_coverage():
    df = datasets.diabetes()
    train, calib, test = _splits(df, 250, 350)
    model, _ = breezeml.regressors.random_forest(train, "target")
    cr = breezeml.conformal.conformal_regressor(model, calib, "target", alpha=0.1)
    rep = cr.coverage_report(test, "target", show=False)
    # marginal coverage should sit near 1-alpha (finite-sample slack)
    assert abs(rep["empirical_coverage"] - 0.9) <= 0.1
    assert rep["mean_interval_width"] > 0


def test_conformal_interval_ordering():
    df = datasets.diabetes()
    train, calib, _ = _splits(df, 250, 350)
    model, _ = breezeml.regressors.random_forest(train, "target")
    cr = breezeml.conformal.conformal_regressor(model, calib, "target", alpha=0.1)
    iv = cr.predict_interval(df.drop(columns=["target"]).head(20), alpha=None)
    lo = np.asarray(iv["lower"])
    pt = np.asarray(iv["point"])
    hi = np.asarray(iv["upper"])
    assert np.all(lo <= pt) and np.all(pt <= hi)


def test_conformal_regression_alpha_monotonic():
    df = datasets.diabetes()
    train, calib, _ = _splits(df, 250, 350)
    model, _ = breezeml.regressors.random_forest(train, "target")
    cr = breezeml.conformal.conformal_regressor(model, calib, "target", alpha=0.1)
    X = df.drop(columns=["target"]).head(50)
    wide = cr.predict_interval(X, alpha=0.05)
    narrow = cr.predict_interval(X, alpha=0.2)
    w_wide = (np.asarray(wide["upper"]) - np.asarray(wide["lower"])).mean()
    w_narrow = (np.asarray(narrow["upper"]) - np.asarray(narrow["lower"])).mean()
    assert w_wide > w_narrow


# ------------------------------------------------------ classification

def test_conformal_classification_coverage():
    df = datasets.breast_cancer()
    train, calib, test = _splits(df, 300, 450)
    model, _ = breezeml.classifiers.random_forest(train, "label")
    cc = breezeml.conformal.conformal_classifier(model, calib, "label", alpha=0.1)
    rep = cc.coverage_report(test, "label", show=False)
    # marginal coverage is a finite-sample guarantee in expectation over
    # calibration draws; a single small eval split can dip below the target
    assert rep["empirical_coverage"] >= 0.80
    assert rep["mean_set_size"] > 0


def test_conformal_predict_set_shape():
    df = datasets.breast_cancer()
    train, calib, _ = _splits(df, 300, 450)
    model, _ = breezeml.classifiers.random_forest(train, "label")
    cc = breezeml.conformal.conformal_classifier(model, calib, "label", alpha=0.1)
    sets = cc.predict_set(df.drop(columns=["label"]).head(5), alpha=None)
    assert isinstance(sets, list) and len(sets) == 5
    assert all(isinstance(s, list) for s in sets)


def test_conformal_classification_alpha_monotonic():
    df = datasets.breast_cancer()
    train, calib, _ = _splits(df, 300, 450)
    model, _ = breezeml.classifiers.random_forest(train, "label")
    cc = breezeml.conformal.conformal_classifier(model, calib, "label", alpha=0.1)
    X = df.drop(columns=["label"]).head(100)
    strict = cc.predict_set(X, alpha=0.05)
    loose = cc.predict_set(X, alpha=0.3)
    assert np.mean([len(s) for s in strict]) >= np.mean([len(s) for s in loose])


def test_conformal_classifier_requires_proba():
    df = datasets.wine()
    train, calib, _ = _splits(df, 100, 140)
    # LinearSVC has no predict_proba
    model, _ = breezeml.classifiers.linear_svm(train, "class")
    with pytest.raises(TypeError):
        breezeml.conformal.conformal_classifier(model, calib, "class", alpha=0.1)
