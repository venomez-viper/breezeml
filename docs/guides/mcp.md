# MCP server: the ML library AI agents can use

BreezeML ships a [Model Context Protocol](https://modelcontextprotocol.io)
server. Any MCP-capable agent (Claude Code, Claude Desktop, and a growing
list of others) can train, compare, explain, export, and deploy models on
your CSV files, with BreezeML's statistical guardrails underneath.

## Setup

```bash
pip install breezeml[mcp]
```

Register with Claude Code:

```bash
claude mcp add breezeml -- breezeml-mcp
```

Or in any MCP client config:

```json
{
  "mcpServers": {
    "breezeml": {
      "command": "breezeml-mcp"
    }
  }
}
```

## Tools exposed

| Tool | What it does |
|---|---|
| `inspect_data` | Profile a CSV: rows, feature types, missing values, class balance |
| `compare` | Benchmark all built-in models, return a ranked leaderboard |
| `train` | Train the auto-selected pipeline; returns metrics + plain-English decisions |
| `predict` | Run predictions with a trained model |
| `explain` | Explain every pipeline decision for a trained model |
| `model_card` | Generate a markdown model card |
| `export` | Write a standalone sklearn training script (zero breezeml imports) |
| `deploy` | Write a FastAPI + Docker serving directory |
| `save` | Persist a model to .joblib |

## Example session

Ask your agent:

> "Train a model on sales.csv predicting churn, show me why it made its
> preprocessing choices, and deploy it as an API."

The agent chains `inspect_data` → `compare` → `train` → `explain` →
`deploy`, and every intermediate result is a JSON report you can audit.

## Why this matters

Agents writing raw sklearn from scratch re-invent (and frequently botch)
train/test hygiene, imputation, and metric choice. The MCP server gives
agents the same sound defaults human users get: stratified splits, leakage-
safe pipelines, honest metrics.
