"""
BreezeML MCP server: lets AI agents (Claude, GPT, etc.) train, compare,
explain, export, and deploy ML models through the Model Context Protocol.

Install:
    pip install breezeml[mcp]

Run (stdio transport):
    breezeml-mcp

Claude Code registration:
    claude mcp add breezeml -- breezeml-mcp

All tool logic lives in plain ``tool_*`` functions so it is fully testable
without the mcp package; ``build_server()`` wraps them for MCP.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ._meta import profile_data
from ._narrate import narrate

# ------------------------------------------------------------------ state

_MODELS: dict[str, object] = {}
_COUNTER = {"n": 0}


def _register(model) -> str:
    _COUNTER["n"] += 1
    model_id = f"model_{_COUNTER['n']}"
    _MODELS[model_id] = model
    return model_id


def _get(model_id: str):
    model = _MODELS.get(model_id)
    if model is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Trained models this session: {list(_MODELS)}"
        )
    return model


def _load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ------------------------------------------------------------------ tools


def tool_inspect_data(csv_path: str, target: str) -> str:
    """Profile a CSV: rows, features, missing values, class balance."""
    df = _load_df(csv_path)
    profile = profile_data(df, target)
    return json.dumps(profile, indent=2)


def tool_audit(csv_path: str, target: str) -> str:
    """Audit a CSV for data-quality problems and target leakage before
    training: ID columns, constants, duplicates, label noise, and features
    that predict the target suspiciously well on their own."""
    from .audit import audit

    df = _load_df(csv_path)
    result = audit(df, target, show=False)
    return json.dumps(result, indent=2, default=str)


def tool_train(csv_path: str, target: str, task: str = "auto") -> str:
    """Train a model on a CSV. Returns model_id, metrics, and an explanation
    of every pipeline decision."""
    from .breezeml import auto

    df = _load_df(csv_path)
    model, report = auto(df, target, task=task)
    model_id = _register(model)
    return json.dumps(
        {
            "model_id": model_id,
            "task": model.task,
            "estimator": model.meta["estimator"],
            "report": report,
            "decisions": narrate(model.meta),
        },
        indent=2,
    )


def tool_compare(csv_path: str, target: str, task: str = "auto") -> str:
    """Benchmark all built-in models on a CSV and return a ranked leaderboard."""
    from . import classifiers, regressors

    df = _load_df(csv_path)
    y = df[target]
    is_classification = task == "classification" or (
        task == "auto" and (y.dtype == "object" or y.nunique() < 20)
    )
    if is_classification:
        results = classifiers.compare(df, target, show=False)
    else:
        results = regressors.compare(df, target, show=False)
    return json.dumps({"task": "classification" if is_classification else "regression",
                       "leaderboard": results}, indent=2, default=str)


def tool_predict(model_id: str, records_json: str) -> str:
    """Run predictions with a trained model. records_json is a JSON list of
    feature dicts."""
    model = _get(model_id)
    records = json.loads(records_json)
    if not isinstance(records, list) or not records:
        raise ValueError("records_json must be a non-empty JSON list of objects.")
    df = pd.DataFrame(records)
    preds = model.predict(df)
    return json.dumps({"predictions": [p.item() if hasattr(p, "item") else p for p in preds]})


def tool_explain(model_id: str) -> str:
    """Plain-English explanation of a trained model's pipeline decisions."""
    model = _get(model_id)
    return json.dumps({"decisions": narrate(model.meta)}, indent=2)


def tool_report(model_id: str, csv_path: str, target: str = "", sensitive: str = "") -> str:
    """Run the full honesty gauntlet on a trained model and return a single
    SHIP / WARN / STOP verdict: cross-validated performance vs a naive baseline,
    data audit (leakage/quality), class-imbalance severity, and an optional
    fairness check. Agents SHOULD call this and confirm a SHIP verdict before
    calling deploy() or export()."""
    from .report import report as _report

    model = _get(model_id)
    df = _load_df(csv_path)
    rep = _report(model, df, target=target or None, sensitive=sensitive or None, show=False)
    return json.dumps(rep.to_dict(), indent=2, default=str)


def tool_model_card(model_id: str, path: str = "") -> str:
    """Generate a markdown model card. Optionally write it to a file."""
    from .card import card

    model = _get(model_id)
    return card(model, path or None)


def tool_export(model_id: str, path: str = "train.py", data_path: str = "YOUR_DATA.csv") -> str:
    """Export a trained model as a standalone scikit-learn script with zero
    breezeml imports."""
    from .export import export

    model = _get(model_id)
    written = export(model, path, data_path=data_path)
    return f"Standalone sklearn training script written to {written}"


def tool_deploy(model_id: str, out_dir: str = "deployment", name: str = "breezeml-model") -> str:
    """Write a complete FastAPI + Docker serving directory for a trained model."""
    from .deploy import deploy

    model = _get(model_id)
    written = deploy(model, out_dir, name)
    return (
        f"Serving app written to {written}/ "
        f"(run: cd {written} && pip install -r requirements.txt && uvicorn app:app)"
    )


def tool_save(model_id: str, path: str) -> str:
    """Persist a trained model to a .joblib file."""
    model = _get(model_id)
    model.save(path)
    return f"Model saved to {path}"


# ------------------------------------------------------------------ server


def build_server():
    """Construct the FastMCP server (requires the [mcp] extra)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The MCP server requires the mcp package. Install with: pip install breezeml[mcp]"
        ) from exc

    server = FastMCP(
        "breezeml",
        instructions=(
            "BreezeML: train, compare, explain, export, and deploy classical ML "
            "models on CSV data. Typical flow: inspect_data -> compare -> train -> "
            "model_card -> deploy."
        ),
    )
    server.tool(name="inspect_data")(tool_inspect_data)
    server.tool(name="audit")(tool_audit)
    server.tool(name="train")(tool_train)
    server.tool(name="compare")(tool_compare)
    server.tool(name="predict")(tool_predict)
    server.tool(name="explain")(tool_explain)
    server.tool(name="report")(tool_report)
    server.tool(name="model_card")(tool_model_card)
    server.tool(name="export")(tool_export)
    server.tool(name="deploy")(tool_deploy)
    server.tool(name="save")(tool_save)
    return server


def main():
    """Entry point for the ``breezeml-mcp`` console script (stdio transport)."""
    build_server().run()


if __name__ == "__main__":
    main()
