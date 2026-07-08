# BreezeML

A beginner-friendly, production-aware ML workflow layer for students,
analysts, and AI agents.

BreezeML helps you train, compare, explain, export, and deploy
scikit-learn models without drowning in boilerplate, while keeping the
workflow statistically sound. Version `1.0.0` makes four promises no other
low-code ML library makes together:

1. **4 dependencies. Always.** Core installs with only scikit-learn, pandas,
   numpy, joblib - enforced by a CI test that fails if anyone adds a fifth.
2. **Zero lock-in.** [`export()`](guides/export.md) writes a standalone
   sklearn script reproducing your exact pipeline. Graduate anytime.
3. **It teaches you.** [`explain_decisions=True`](guides/model-cards.md)
   narrates every pipeline choice in plain English; `card()` writes honest
   model cards with auto-detected caveats.
4. **AI agents can use it.** [`breezeml-mcp`](guides/mcp.md) is a built-in
   Model Context Protocol server for Claude and other agents.

Plus everything from the other releases: 22 classifiers, 22 regressors,
9 clusterers, comparison leaderboards, hyperparameter search,
cross-validation, feature engineering, optional XGBoost/LightGBM, plotting
and SHAP explainability. Version 1.7 adds the
[honest-ML toolkit](guides/honest-ml.md): data audits with target-leakage
detection, fairness reports, an imbalance toolkit, and blending that admits
when it loses. Version 1.8 adds four more answers: statistical
[significance testing](guides/significance.md) (is this difference real?),
[multi-label and multi-output](guides/multi-output.md) prediction (many
targets at once), [recommenders](guides/recommenders.md) (what should this
user see next?), and [survival analysis](guides/survival.md) (when will the
event happen?).

## Quickstart

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model, report = breezeml.auto(df, "species", explain_decisions=True)

breezeml.card(model, "MODEL_CARD.md")   # honest model card
breezeml.export(model, "train.py")      # pure-sklearn script, zero lock-in
breezeml.deploy(model, "api/")          # FastAPI app + Dockerfile
```

## Install

```bash
pip install breezeml
```

Optional extras:

```bash
pip install "breezeml[boost]"    # XGBoost + LightGBM
pip install "breezeml[deploy]"   # fastapi + uvicorn for deploy()
pip install "breezeml[mcp]"      # MCP server for AI agents
pip install "breezeml[onnx]"     # ONNX export
pip install "breezeml[all]"
```

## Where to go next

- [Honest ML: audit, fairness, imbalance, blend](guides/honest-ml.md)
- [Toolkits: track, anomaly, semi-supervised, native explain, CLI](guides/toolkits.md)
- [Significance: is this difference real?](guides/significance.md)
- [Multi-output: predict many targets at once](guides/multi-output.md)
- [Recommenders: what should this user see next?](guides/recommenders.md)
- [Survival: when will the event happen?](guides/survival.md)
- [Export: graduate from BreezeML anytime](guides/export.md)
- [Model cards & teaching narration](guides/model-cards.md)
- [Deploy: DataFrame to API in one line](guides/deploy.md)
- [MCP server for AI agents](guides/mcp.md)
- [Benchmarks vs PyCaret and LazyPredict](benchmarks.md)
- Read the [examples](examples.md) and browse the API reference
- Review the [CHANGELOG](https://github.com/venomez-viper/breezeml/blob/main/CHANGELOG.md)
