"""Tests for BreezeAutoML (v1.1)."""
import time

import pytest

import breezeml
from breezeml import datasets


@pytest.fixture(scope="module")
def iris_result():
    df = datasets.iris()
    start = time.monotonic()
    model, report = breezeml.automl(df, "species", time_budget=45, show=False)
    elapsed = time.monotonic() - start
    return model, report, elapsed


def test_automl_classification(iris_result):
    model, report, _ = iris_result
    assert model.task == "classification"
    assert report["best_model"] is not None
    assert report["holdout"]["accuracy"] > 0.8
    assert len(report["leaderboard"]) >= 5


def test_automl_budget_soft_respected(iris_result):
    _, report, elapsed = iris_result
    # Soft budget: running fits may finish, but should not blow far past it.
    assert elapsed < 45 * 3, f"took {elapsed}s against a 45s budget"
    assert report["time_used_seconds"] > 0


def test_automl_meta_and_narration(iris_result):
    model, _, _ = iris_result
    assert model.meta["automl"]["best_model"]
    assert model.meta["automl"]["backend"] == "native"
    # Full v1.0 feature set works on automl output
    card = breezeml.card(model)
    assert "Model Card" in card
    from breezeml.export import export_code

    code = export_code(model)
    assert "import breezeml" not in code


def test_automl_regression():
    df = datasets.diabetes().sample(150, random_state=42)
    model, report = breezeml.automl(df, "target", time_budget=25, show=False)
    assert model.task == "regression"
    assert "r2" in report["holdout"]


def test_automl_tuning_history(iris_result):
    _, report, _ = iris_result
    completed = [t for t in report["tuning_history"] if "score" in t]
    assert completed, "expected at least one tuned model"
    assert completed[0]["params"]


def test_automl_invalid_backend():
    df = datasets.iris()
    with pytest.raises(ValueError, match="backend"):
        breezeml.automl(df, "species", backend="magic")


def test_automl_optuna_backend_or_clear_error():
    df = datasets.iris().sample(60, random_state=42)
    try:
        import optuna  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="breezeml\\[automl\\]"):
            breezeml.automl(df, "species", time_budget=10, backend="optuna", show=False)
    else:
        model, report = breezeml.automl(df, "species", time_budget=15, backend="optuna", show=False)
        assert report["backend"] == "optuna"
        assert report["best_model"] is not None
