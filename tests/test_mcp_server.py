"""Tests for the BreezeML MCP server tool logic (no mcp package required)."""
import json

import pytest

from breezeml import datasets
from breezeml import mcp_server


@pytest.fixture(scope="module")
def iris_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "iris.csv"
    datasets.iris().to_csv(path, index=False)
    return str(path)


def test_inspect_data(iris_csv):
    out = json.loads(mcp_server.tool_inspect_data(iris_csv, "species"))
    assert out["n_rows"] == 150
    assert out["target"] == "species"


def test_train_returns_model_id_and_decisions(iris_csv):
    out = json.loads(mcp_server.tool_train(iris_csv, "species"))
    assert out["model_id"].startswith("model_")
    assert out["report"]["accuracy"] > 0.5
    assert len(out["decisions"]) >= 4


def test_predict_roundtrip(iris_csv):
    trained = json.loads(mcp_server.tool_train(iris_csv, "species"))
    df = datasets.iris().drop(columns=["species"]).head(3)
    records = json.dumps(df.to_dict(orient="records"))
    out = json.loads(mcp_server.tool_predict(trained["model_id"], records))
    assert len(out["predictions"]) == 3


def test_explain_and_card(iris_csv):
    trained = json.loads(mcp_server.tool_train(iris_csv, "species"))
    explained = json.loads(mcp_server.tool_explain(trained["model_id"]))
    assert explained["decisions"]
    card_md = mcp_server.tool_model_card(trained["model_id"])
    assert card_md.startswith("# Model Card:")


def test_export_and_deploy(tmp_path, iris_csv):
    trained = json.loads(mcp_server.tool_train(iris_csv, "species"))
    script = tmp_path / "train.py"
    msg = mcp_server.tool_export(trained["model_id"], str(script))
    assert script.exists() and "written" in msg

    out_dir = tmp_path / "api"
    msg = mcp_server.tool_deploy(trained["model_id"], str(out_dir))
    assert (out_dir / "app.py").exists()


def test_unknown_model_id():
    with pytest.raises(ValueError, match="Unknown model_id"):
        mcp_server.tool_explain("model_nope")


def test_compare_leaderboard(iris_csv):
    out = json.loads(mcp_server.tool_compare(iris_csv, "species"))
    assert out["task"] == "classification"
    assert len(out["leaderboard"]) >= 10


def test_build_server_requires_mcp_or_builds():
    try:
        import mcp  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="breezeml\\[mcp\\]"):
            mcp_server.build_server()
    else:
        server = mcp_server.build_server()
        assert server is not None
