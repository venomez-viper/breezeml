import pytest
pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

from breezeml import datasets, fit, plot


def test_pr_curve_binary():
    df = datasets.breast_cancer()
    model = fit(df, "label")
    X_test = df.drop(columns=["label"])
    y_test = df["label"]
    plot.pr_curve(model, X_test, y_test)


def test_pr_curve_multiclass_raises():
    import io
    import contextlib

    df = datasets.iris()
    model = fit(df, "species")
    X_test = df.drop(columns=["species"])
    y_test = df["species"]
    raised = False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            plot.pr_curve(model, X_test, y_test)
    except ValueError as e:
        raised = "binary" in str(e).lower()
    assert raised, "expected ValueError for multiclass data"
