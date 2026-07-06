"""Tests for v0.5.0: deploy() FastAPI + Docker generation."""
import ast
import os

import joblib
import pytest

import breezeml
from breezeml import datasets


@pytest.fixture(scope="module")
def iris_model():
    df = datasets.iris()
    model, _ = breezeml.auto(df, "species")
    return model


def test_deploy_writes_all_files(tmp_path, iris_model):
    out = tmp_path / "api"
    breezeml.deploy(iris_model, str(out), name="iris-api")
    for filename in ["model.joblib", "app.py", "requirements.txt", "Dockerfile", "README.md"]:
        assert (out / filename).exists(), f"missing {filename}"


def test_deploy_app_is_valid_python(tmp_path, iris_model):
    out = tmp_path / "api"
    breezeml.deploy(iris_model, str(out))
    source = (out / "app.py").read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if broken
    assert "FastAPI" in source
    assert '"species"' in source or "'species'" in source


def test_deploy_saves_raw_pipeline_no_breezeml_needed(tmp_path, iris_model):
    out = tmp_path / "api"
    breezeml.deploy(iris_model, str(out))
    pipeline = joblib.load(out / "model.joblib")
    # Must be a plain sklearn Pipeline, not an EasyModel
    assert type(pipeline).__module__.startswith("sklearn")
    df = datasets.iris().drop(columns=["species"])
    preds = pipeline.predict(df.head())
    assert len(preds) == 5


def test_deploy_app_serves_predictions(tmp_path, iris_model):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    out = tmp_path / "api"
    breezeml.deploy(iris_model, str(out))

    cwd = os.getcwd()
    os.chdir(out)
    try:
        namespace = {}
        source = (out / "app.py").read_text(encoding="utf-8")
        exec(compile(source, "app.py", "exec"), namespace)
        client = TestClient(namespace["app"])

        health = client.get("/health")
        assert health.status_code == 200

        df = datasets.iris().drop(columns=["species"]).head(2)
        records = df.to_dict(orient="records")
        resp = client.post("/predict", json=records)
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["predictions"]) == 2

        bad = client.post("/predict", json=[{"wrong": 1}])
        assert bad.status_code == 422
    finally:
        os.chdir(cwd)


def test_deploy_method_on_easymodel(tmp_path, iris_model):
    out = tmp_path / "via_method"
    iris_model.deploy(str(out))
    assert (out / "app.py").exists()


def test_onnx_requires_extra_or_works(iris_model):
    # iris via core API is numeric-only, so this either works (skl2onnx
    # installed) or raises the helpful ImportError.
    try:
        import skl2onnx  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="breezeml\\[onnx\\]"):
            from breezeml.deploy import to_onnx
            to_onnx(iris_model, "unused.onnx")
