"""Tests for v0.4.0: export(), model cards, and teaching narration."""
import subprocess
import sys

import pytest

import breezeml
from breezeml import datasets
from breezeml._meta import profile_data
from breezeml._narrate import narrate


@pytest.fixture(scope="module")
def iris_model():
    df = datasets.iris()
    model, report = breezeml.auto(df, "species")
    return model, report, df


@pytest.fixture(scope="module")
def diabetes_model():
    df = datasets.diabetes()
    model, report = breezeml.auto(df, "target")
    return model, report, df


# ---------------------------------------------------------------- meta

def test_profile_data_basics():
    df = datasets.iris()
    profile = profile_data(df, "species")
    assert profile["n_rows"] == len(df)
    assert profile["n_features"] == df.shape[1] - 1
    assert profile["target"] == "species"
    assert profile["target_nunique"] == 3
    assert profile["class_counts"] is not None


def test_meta_attached_to_model(iris_model):
    model, _, _ = iris_model
    assert model.meta is not None
    assert model.meta["task"] == "classification"
    assert model.meta["estimator"] == "RandomForestClassifier"
    assert model.meta["report"]["accuracy"] > 0.5
    assert "task_reason" in model.meta


def test_meta_regression(diabetes_model):
    model, _, _ = diabetes_model
    assert model.meta["task"] == "regression"
    assert "r2" in model.meta["report"]


# ------------------------------------------------------------- narrate

def test_narration_produces_decisions(iris_model):
    model, _, _ = iris_model
    decisions = narrate(model.meta)
    assert len(decisions) >= 4
    joined = " ".join(decisions)
    assert "classification" in joined
    assert "split" in joined


def test_explain_decisions_prints(iris_model, capsys):
    model, _, _ = iris_model
    model.explain_decisions()
    out = capsys.readouterr().out
    assert "BreezeML decisions explained" in out


def test_explain_decisions_flag(capsys):
    df = datasets.iris()
    breezeml.auto(df, "species", explain_decisions=True)
    out = capsys.readouterr().out
    assert "BreezeML decisions explained" in out


# -------------------------------------------------------------- export

def test_export_code_no_breezeml_import(iris_model):
    model, _, _ = iris_model
    from breezeml.export import export_code

    code = export_code(model)
    assert "import breezeml" not in code
    assert "from breezeml" not in code
    assert "RandomForestClassifier" in code
    assert 'TARGET = "species"' in code
    assert "ColumnTransformer" in code


def test_export_writes_and_script_is_valid_python(tmp_path, iris_model):
    model, _, df = iris_model
    csv_path = tmp_path / "iris.csv"
    df.to_csv(csv_path, index=False)
    script_path = tmp_path / "train.py"
    breezeml.export(model, str(script_path), data_path=str(csv_path).replace("\\", "/"))

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "accuracy:" in result.stdout
    assert (tmp_path / "model.joblib").exists()


def test_export_regression(tmp_path, diabetes_model):
    model, _, _ = diabetes_model
    from breezeml.export import export_code

    code = export_code(model)
    assert "r2_score" in code
    assert "import breezeml" not in code


# ---------------------------------------------------------------- card

def test_card_markdown(iris_model):
    model, _, _ = iris_model
    text = breezeml.card(model)
    assert text.startswith("# Model Card:")
    assert "## Training Data" in text
    assert "## Evaluation" in text
    assert "## Pipeline Decisions" in text
    assert "## Limitations" in text


def test_card_writes_file(tmp_path, iris_model):
    model, _, _ = iris_model
    path = tmp_path / "card.md"
    breezeml.card(model, str(path))
    assert path.exists()
    assert "Model Card" in path.read_text(encoding="utf-8")


def test_card_requires_meta():
    from breezeml.breezeml import EasyModel

    bare = EasyModel(None, "y", "classification")
    with pytest.raises(ValueError):
        breezeml.card(bare)
