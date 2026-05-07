from breezeml import datasets, regressors


def test_linear_regressor():
    df = datasets.diabetes()
    model, report = regressors.linear(df, "target")
    assert hasattr(model, "predict")
    assert "r2" in report
    assert "mae" in report
    assert "rmse" in report
    assert "adjusted_r2" in report
    assert "mape" in report


def test_compare_regressors():
    df = datasets.diabetes()
    results = regressors.compare(df, "target", show=False)
    assert len(results) > 0
    assert "regressor" in results[0]
    assert "r2" in results[0]


def test_regressor_detailed_report():
    df = datasets.diabetes()
    info = regressors.detailed_report(df, "target", algo="decision_tree")
    assert "explained_variance" in info
    assert "residuals" in info
    assert "prediction_vs_actual" in info


def test_regressor_quick_tune():
    df = datasets.diabetes()
    model, params, report = regressors.quick_tune(df, "target", algo="decision_tree", n_iter=2, cv=2)
    assert hasattr(model, "predict")
    assert isinstance(params, dict)
    assert "r2" in report
