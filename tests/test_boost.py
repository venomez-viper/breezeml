import pytest

from breezeml import classifiers, datasets, regressors


def test_xgboost_classifier():
    pytest.importorskip("xgboost")
    df = datasets.iris()
    model, report = classifiers.xgboost(df, "species")
    assert hasattr(model, "predict")
    assert "accuracy" in report


def test_lightgbm_classifier():
    pytest.importorskip("lightgbm")
    df = datasets.iris()
    model, report = classifiers.lightgbm(df, "species")
    assert hasattr(model, "predict")
    assert "accuracy" in report


def test_xgboost_regressor():
    pytest.importorskip("xgboost")
    df = datasets.diabetes()
    model, report = regressors.xgboost(df, "target")
    assert hasattr(model, "predict")
    assert "r2" in report


def test_lightgbm_regressor():
    pytest.importorskip("lightgbm")
    df = datasets.diabetes()
    model, report = regressors.lightgbm(df, "target")
    assert hasattr(model, "predict")
    assert "r2" in report
