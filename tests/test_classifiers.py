from breezeml import datasets, classifiers

def test_logistic():
    df = datasets.iris()
    model, report = classifiers.logistic(df, "species")
    assert "accuracy" in report
    assert "f1" in report
    assert "macro_f1" in report

def test_compare():
    df = datasets.iris()
    results = classifiers.compare(df, "species", show=False)
    assert len(results) > 0
    assert "classifier" in results[0]
    assert "accuracy" in results[0]

def test_detailed_report():
    df = datasets.iris()
    info = classifiers.detailed_report(df, "species", algo="decision_tree")
    assert "accuracy" in info
    assert "confusion_matrix" in info
    assert "roc_auc" in info

def test_quick_tune():
    df = datasets.iris()
    model, params, report = classifiers.quick_tune(df, "species", algo="decision_tree", n_iter=2, cv=2)
    assert isinstance(params, dict)
    assert "accuracy" in report
