"""2.0 unified Model surface: one coherent object from fit()."""
import warnings

import breezeml
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def test_model_alias():
    assert breezeml.Model is breezeml.EasyModel


def test_coherent_surface():
    df = breezeml.datasets.iris()
    m = breezeml.fit(df, "species")
    for method in ("predict", "evaluate", "report", "explain", "card",
                   "export", "deploy", "predict_interval", "predict_set"):
        assert hasattr(m, method), f"Model missing {method}()"


def test_predict_set_classification():
    df = breezeml.datasets.iris()
    tr, cal = train_test_split(df, test_size=0.3, random_state=0)
    m = breezeml.fit(tr, "species")
    sets = m.predict_set(cal.drop(columns=["species"]).head(5), cal, alpha=0.1)
    assert len(sets) == 5
    assert all(len(s) >= 1 for s in sets)


def test_predict_interval_regression():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True).frame.rename(columns={"target": "y"})
    tr, cal = train_test_split(d, test_size=0.3, random_state=0)
    m = breezeml.fit(tr, "y")
    iv = m.predict_interval(cal.drop(columns=["y"]).head(3), cal, alpha=0.1)
    assert list(iv.columns) == ["lower", "point", "upper"]
    assert (iv["lower"] <= iv["upper"]).all()


def test_wrong_task_raises():
    import pytest
    df = breezeml.datasets.iris()
    m = breezeml.fit(df, "species")
    with pytest.raises(ValueError):
        m.predict_interval(df.drop(columns=["species"]).head(2), df)
