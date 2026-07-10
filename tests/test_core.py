from breezeml import datasets, fit, predict, auto, report

def test_classify_and_predict():
    df = datasets.iris()
    model = fit(df, "species")
    assert model.task == "classification"
    
    preds = predict(model, df.drop(columns=["species"]))
    assert len(preds) == len(df)

def test_regress_and_predict():
    df = datasets.diabetes()
    model = fit(df, "target")
    assert model.task == "regression"
    
    preds = predict(model, df.drop(columns=["target"]))
    assert len(preds) == len(df)

def test_auto_classification():
    df = datasets.iris()
    model, rep = auto(df, "species")
    assert model.task == "classification"
    assert "accuracy" in rep
    assert "f1" in rep
    assert 0 <= rep["accuracy"] <= 1

def test_auto_regression():
    df = datasets.diabetes()
    model, rep = auto(df, "target")
    assert model.task == "regression"
    assert "r2" in rep
    assert "rmse" in rep

def test_report():
    df = datasets.iris()
    model = fit(df, "species")
    # 2.0: report() returns a Report (honest scorecard), not a metrics dict
    rep = report(model, df, show=False)
    assert rep.verdict in {"SHIP", "WARN", "STOP"}
    assert "accuracy" in rep.sections["performance"]["metrics"]
    # a plain metrics dict is available via evaluate()
    assert "accuracy" in model.evaluate(df)
