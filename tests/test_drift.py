"""Tests for breezeml.drift (v1.3)."""
import numpy as np
import pandas as pd
import pytest

import breezeml
from breezeml import datasets, drift


@pytest.fixture(scope="module")
def iris_model():
    df = datasets.iris()
    model, _ = breezeml.auto(df, "species")
    return model, df


def test_reference_stats_stored(iris_model):
    model, _ = iris_model
    ref = model.meta["reference"]
    assert ref["numeric"], "numeric reference distributions missing"
    assert ref["n_rows"] == 150
    for col, stats in ref["numeric"].items():
        assert len(stats["bin_edges"]) == len(stats["bin_props"]) + 1
        assert abs(sum(stats["bin_props"]) - 1.0) < 1e-6


def test_no_drift_on_same_data(iris_model):
    model, df = iris_model
    result = drift.check(model, df)
    assert result["drifted"] is False
    assert result["drifted_columns"] == []
    assert "No significant drift" in result["summary"]


def test_drift_detected_on_shifted_data(iris_model):
    model, df = iris_model
    shifted = df.drop(columns=["species"]).copy()
    shifted["sepal length (cm)"] = shifted["sepal length (cm)"] + 10  # massive shift
    result = drift.check(model, shifted)
    assert result["drifted"] is True
    assert "sepal length (cm)" in result["drifted_columns"]
    col = result["columns"]["sepal length (cm)"]
    assert col["psi"] >= 0.25
    assert col["share_outside_training_range"] > 0.9


def test_missing_column_flagged(iris_model):
    model, df = iris_model
    partial = df.drop(columns=["species", "petal width (cm)"])
    result = drift.check(model, partial)
    assert result["drifted"] is True
    assert "petal width (cm)" in result["missing_columns"]


def test_categorical_drift_new_categories():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "color": rng.choice(["red", "blue"], 300),
        "size": rng.normal(10, 2, 300),
        "label": rng.choice(["a", "b"], 300),
    })
    model, _ = breezeml.auto(df, "label")
    new = df.drop(columns=["label"]).copy()
    new["color"] = "green"  # never seen in training
    result = drift.check(model, new)
    col = result["columns"]["color"]
    assert col["unseen_category_share"] == 1.0
    assert "green" in col["new_categories"]
    assert result["drifted"] is True


def test_check_drift_method(iris_model):
    model, df = iris_model
    result = model.check_drift(df)
    assert result["drifted"] is False


def test_old_model_without_reference():
    from breezeml.breezeml import EasyModel

    bare = EasyModel(None, "y", "classification", meta={"profile": {}})
    with pytest.raises(ValueError, match="reference"):
        drift.check(bare, pd.DataFrame({"x": [1]}))


def test_deploy_exports_reference_and_drift_endpoint(tmp_path, iris_model):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import os

    model, df = iris_model
    out = tmp_path / "api"
    breezeml.deploy(model, str(out))
    assert (out / "reference.json").exists()

    cwd = os.getcwd()
    os.chdir(out)
    try:
        namespace = {}
        exec(compile((out / "app.py").read_text(encoding="utf-8"), "app.py", "exec"), namespace)
        client = TestClient(namespace["app"])

        records = df.drop(columns=["species"]).to_dict(orient="records")
        # Below buffer minimum
        client.post("/predict", json=records[:10])
        early = client.get("/drift").json()
        assert early["status"] == "insufficient_data"

        # Enough data, same distribution -> no drift
        client.post("/predict", json=records)
        report = client.get("/drift").json()
        assert report["drifted"] is False

        # Shifted data -> drift
        shifted = [{k: (v + 10 if isinstance(v, float) else v) for k, v in r.items()} for r in records]
        client.post("/predict", json=shifted)
        report = client.get("/drift").json()
        assert report["drifted"] is True
    finally:
        os.chdir(cwd)
