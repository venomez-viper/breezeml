# BreezeML v1.0.0 - The Legendary Release

Four promises no other low-code ML library makes together:

## 1. Zero lock-in: `export()`

```python
breezeml.export(model, "train.py")
```

A standalone scikit-learn script reproducing your exact pipeline - same
imputers, scaler, encoder, hyperparameters, seed, and split. **No breezeml
import anywhere.** The moment you outgrow BreezeML, your code walks out the
door with you.

## 2. It teaches you: narration + model cards

```python
model, report = breezeml.auto(df, "churn", explain_decisions=True)
# 1. Detected a classification task: target 'churn' has only 2 distinct values...
# 2. Used a stratified 80/20 split because class imbalance is 4.3:1...
# 3. Filled missing numeric values (2.1% of cells) with the column median...

breezeml.card(model, "MODEL_CARD.md")   # honest model card with auto-detected caveats
```

Narration is generated from measured facts about *your* data - imbalance
ratios, missing percentages, IQR outlier checks - not canned text.

## 3. One-line deployment: `deploy()`

```python
breezeml.deploy(model, "api/")
# api/: FastAPI app + Dockerfile + requirements + raw sklearn pipeline
```

The serving app validates inputs, exposes Swagger docs, and never imports
breezeml.

## 4. AI agents can use it: `breezeml-mcp`

```bash
pip install breezeml[mcp]
claude mcp add breezeml -- breezeml-mcp
```

The first low-code ML library with a built-in Model Context Protocol
server. Agents get `inspect_data`, `compare`, `train`, `predict`,
`explain`, `model_card`, `export`, and `deploy` - with stratified splits,
leakage-safe pipelines, and honest metrics underneath.

## The dependency contract

Core BreezeML installs with **scikit-learn, pandas, numpy, joblib. Nothing
else. Ever.** A CI test fails the build if anyone adds a fifth hard
dependency. No dependency hell.

## Also in this release

- ONNX export (`[onnx]` extra) for numeric pipelines
- `benchmarks/run_benchmarks.py` - honest, reproducible comparison vs PyCaret and LazyPredict
- New docs guides: export, model cards, deploy, MCP
- Training metadata (`model.meta`) on all core-API models

Full details in [CHANGELOG.md](CHANGELOG.md).
