<div align="center">

<img src="https://raw.githubusercontent.com/venomez-viper/breezeml/main/assets/breezeml_banner_1778130992461.png" alt="BreezeML Banner" width="800"/>

# BreezeML

**Machine learning without the boilerplate.**

*Train, evaluate, compare, and save models in a few lines.*

<br/>

[![PyPI version](https://badge.fury.io/py/breezeml.svg)](https://pypi.org/project/breezeml/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/breezeml?color=blue&label=PyPI%20Downloads)](https://pypi.org/project/breezeml/)
![CI Status](https://github.com/venomez-viper/breezeml/actions/workflows/ci.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/venomez-viper/breezeml?color=green)](https://github.com/venomez-viper/breezeml/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![4 Dependencies](https://img.shields.io/badge/core%20deps-4%2C%20always-brightgreen)](tests/test_dependency_contract.py)
[![MCP Server](https://img.shields.io/badge/MCP-agent%20ready-blueviolet)](docs/guides/mcp.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)
[![scikit-learn](https://img.shields.io/badge/built%20on-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

<br/>

[**Getting Started**](#installation) · [**API Reference**](#api-reference) · [**Examples**](#examples) · [**Contributing**](CONTRIBUTING.md) · [**Changelog**](CHANGELOG.md)

</div>

---

## Overview

BreezeML is a high-level machine learning library built on top of **scikit-learn**, designed to remove boilerplate while keeping the underlying workflow statistically sound. It handles preprocessing, train/test splits, model comparison, tuning, evaluation, deployment, and persistence behind a compact API that stays readable for both beginners and working practitioners.

```python
from breezeml import datasets, fit, predict

df = datasets.iris()
model = fit(df, "species")
preds = predict(model, df.drop(columns=["species"]))
```

That is the core idea: fewer moving parts, fewer repetitive preprocessing steps, and sensible defaults.

---

## Why BreezeML (v1.0)

Four promises no other low-code ML library makes together:

1. **4 dependencies. Always.** Core installs with only scikit-learn, pandas, numpy, joblib, [CI-enforced](tests/test_dependency_contract.py). No dependency hell, ever.
2. **Zero lock-in.** `export()` hands you a standalone sklearn script reproducing your exact pipeline, with no breezeml import. Graduate anytime.
3. **It teaches you.** `explain_decisions=True` narrates every pipeline choice in plain English, and `card()` writes an honest model card with auto-detected caveats.
4. **AI agents can use it.** `breezeml-mcp` is a built-in [MCP server](docs/guides/mcp.md): Claude and other agents train, compare, explain, and deploy models with sound statistical defaults.

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model, report = breezeml.auto(df, "species", explain_decisions=True)  # learn while training

breezeml.card(model, "MODEL_CARD.md")     # honest model card
breezeml.export(model, "train.py")        # pure-sklearn script, zero lock-in
breezeml.deploy(model, "api/")            # FastAPI app + Dockerfile, ready to run
```

---

## Key Features

| Feature | Description |
|---|---|
| **Zero lock-in export** *(v1.0)* | `export()` writes a standalone sklearn training script with no breezeml imports |
| **Model cards** *(v1.0)* | `card()` generates honest markdown model cards with auto-detected caveats |
| **Teaching narration** *(v1.0)* | `explain_decisions=True` explains every pipeline choice in plain English |
| **One-line deployment** *(v1.0)* | `deploy()` writes a FastAPI app + Dockerfile serving the raw sklearn pipeline |
| **MCP server for AI agents** *(v1.0)* | `breezeml-mcp` lets Claude & other agents train/compare/explain/deploy models |
| **Dependency contract** *(v1.0)* | Core needs only sklearn, pandas, numpy, joblib, enforced by CI, forever |
| **Auto task detection** | Automatically selects classification or regression based on the target column |
| **12 classifiers** | From Logistic Regression to Neural Nets, available in one function call |
| **10 regressors** *(v0.3.0)* | From Linear Regression to Gradient Boosting and MLP, available in one function call |
| **Classifier leaderboard** | `classifiers.compare()` ranks all built-in classifiers by accuracy and F1 |
| **Regressor leaderboard** *(v0.3.0)* | `regressors.compare()` ranks all built-in regressors by R2, MAE, and RMSE |
| **Cross-validation support** *(v0.3.0)* | Most classifier and regressor training helpers now accept `cv=` and return mean/std metrics |
| **Feature engineering toolkit** *(v0.3.0)* | `breezeml.features` adds selection, importance, PCA, and polynomial expansion helpers |
| **Optional boosting backends** *(v0.3.0)* | XGBoost and LightGBM plug into the compare and tuning flows when installed |
| **Hyperparameter tuning** | `quick_tune()` wrappers run `RandomizedSearchCV` with curated parameter grids |
| **Detailed reports** | Classification and regression helpers expose richer diagnostics in one call |
| **Built-in datasets** | Iris, Wine, Breast Cancer, Diabetes, California Housing, and Penguins are available immediately |
| **Model persistence** | `save()` / `load()` use `joblib` under the hood |
| **Text embeddings** *(v0.2.9)* | `breezeml.text.embed()` converts raw text columns to dense semantic vectors |
| **Explainability** *(v0.2.9)* | `breezeml.explain` gives SHAP-based feature importance plots |
| **Plotting helpers** *(v0.2.9)* | `breezeml.plot` includes confusion matrix and ROC curve helpers |
| **Strict validation** *(v0.2.8)* | Public APIs validate dataframes and target columns up front |

---

## Architecture

```text
breezeml/
|-- breezeml.py        # Core API: fit, predict, auto, from_csv, save, load
|-- classifiers.py     # 12 classifiers + compare, detailed_report, quick_tune
|-- regressors.py      # 10 regressors + compare, detailed_report, quick_tune
|-- clustering.py      # kmeans, agglomerative, dbscan
|-- features.py        # feature selection, importances, PCA, polynomial expansion
|-- text.py            # semantic text embeddings
|-- explain.py         # SHAP explainability
|-- plot.py            # plotting helpers
`-- __init__.py        # public API surface
```

**Internal pipeline**

```text
Raw DataFrame
    |
    v
ColumnTransformer
  |- Numeric     -> Median imputer + scaler
  `- Categorical -> Mode imputer + one-hot encoder
    |
    v
sklearn estimator
    |
    v
EasyModel wrapper
```

---

## Installation

**Stable release**

```bash
pip install breezeml
```

**Latest from source**

```bash
git clone https://github.com/venomez-viper/breezeml.git
cd breezeml
pip install -e .
```

**Requirements:** Python >= 3.9, scikit-learn, pandas, numpy, joblib

Optional extras:

```bash
pip install "breezeml[nlp]"
pip install "breezeml[explain]"
pip install "breezeml[plot]"
pip install "breezeml[boost]"
pip install "breezeml[datasets]"
pip install "breezeml[deploy]"   # fastapi + uvicorn for deploy()
pip install "breezeml[onnx]"     # ONNX export
pip install "breezeml[mcp]"      # MCP server for AI agents
pip install "breezeml[all]"
```

---

## Quickstart

### Classification in 3 lines

```python
from breezeml import datasets, fit, predict

df = datasets.iris()
model = fit(df, "species")
print(predict(model, df.drop(columns=["species"]))[:5])
```

### Auto mode for regression

```python
from breezeml import auto, datasets

df = datasets.diabetes()
model, report = auto(df, "target")
print(report)
```

### Dedicated regression workflow *(new in v0.3.0)*

```python
from breezeml import datasets, regressors

df = datasets.diabetes()
model, report = regressors.gradient_boosting(df, "target")
print(report)
```

### Cross-validation in one line *(new in v0.3.0)*

```python
from breezeml import classifiers, datasets

df = datasets.iris()
model, report = classifiers.logistic(df, "species", cv=5)
print(report)
```

### Load your own CSV

```python
from breezeml import from_csv

model, report = from_csv("sales_data.csv", target="revenue")
print(report)
```

---

## API Reference

### Core Functions

#### `fit(df, target, task="auto") -> EasyModel`

Train a model. Task type is inferred automatically unless you override it.

```python
model = fit(df, "target_column", task="classification")
```

#### `predict(model, X) -> np.ndarray`

Run inference on new data.

```python
predictions = predict(model, new_df)
```

#### `auto(df, target, task="auto") -> (EasyModel, dict)`

Same as `fit`, but returns an evaluation report alongside the trained model.

```python
model, report = auto(df, "target_column", task="regression")
```

#### `from_csv(path, target) -> (EasyModel, dict)`

Load a CSV, train a model, and return its evaluation report.

```python
model, report = from_csv("data.csv", target="label")
```

#### `save(model, path)` / `load(path)`

Persist and restore any trained `EasyModel`.

```python
save(model, "my_model.joblib")
model = load("my_model.joblib")
```

#### `export(model, path, data_path="YOUR_DATA.csv")` *(v1.0)*

Write a standalone scikit-learn training script that reproduces the exact
pipeline (imputers, scaler, encoder, estimator, seed, split) with **zero
breezeml imports**. Also available as `model.export(path)`.

```python
breezeml.export(model, "train.py", data_path="iris.csv")
```

#### `card(model, path=None)` *(v1.0)*

Generate an honest markdown model card: data profile, metrics, every
pipeline decision explained, and auto-detected caveats (small data, class
imbalance, heavy imputation). Also available as `model.card(path)`.

```python
print(breezeml.card(model))
breezeml.card(model, "MODEL_CARD.md")
```

#### `deploy(model, out_dir, name)` *(v1.0)*

Write a complete serving directory: FastAPI app (`/predict`, `/health`,
`/docs`), Dockerfile, requirements, and the raw sklearn pipeline. The app
never imports breezeml. Also available as `model.deploy(out_dir)`.

```python
breezeml.deploy(model, "api/", name="iris-classifier")
# cd api && pip install -r requirements.txt && uvicorn app:app
```

#### Teaching narration *(v1.0)*

```python
model, report = breezeml.auto(df, "species", explain_decisions=True)
# BreezeML decisions explained:
#   1. Detected a classification task: target 'species' has only 3 distinct values...
#   2. Used a stratified 80/20 train/test split because...
model.explain_decisions()  # again, anytime
```

---

### MCP Server for AI Agents *(v1.0)*

BreezeML ships a [Model Context Protocol](https://modelcontextprotocol.io) server so AI agents (Claude Code, Claude Desktop, and other MCP clients) can train, compare, explain, export, and deploy models with statistically sound defaults.

```bash
pip install breezeml[mcp]
claude mcp add breezeml -- breezeml-mcp
```

Tools: `inspect_data`, `compare`, `train`, `predict`, `explain`, `model_card`, `export`, `deploy`, `save`. See the [MCP guide](docs/guides/mcp.md).

---

### `classifiers` Module

All classifier functions share the same signature:

```python
model, report = classifiers.<name>(df, target)
```

The standard report includes:

```python
{"accuracy": float, "f1": float, "macro_f1": float}
```

#### Available Classifiers

| Function | Algorithm | Notes |
|---|---|---|
| `classifiers.logistic` | Logistic Regression | Linear baseline |
| `classifiers.svm` | SVM (RBF kernel) | Robust for small to medium datasets |
| `classifiers.linear_svm` | Linear SVM | Scales well to large sparse feature spaces |
| `classifiers.gaussian_nb` | Gaussian Naive Bayes | Fast for numeric features |
| `classifiers.multinomial_nb` | Multinomial Naive Bayes | Good for counts and TF-IDF |
| `classifiers.decision_tree` | Decision Tree | Fully interpretable |
| `classifiers.random_forest` | Random Forest | Strong general-purpose baseline |
| `classifiers.knn` | K-Nearest Neighbors | Non-parametric |
| `classifiers.gradient_boosting` | Gradient Boosting | High tabular accuracy |
| `classifiers.adaboost` | AdaBoost | Ensemble boosting |
| `classifiers.extra_trees` | Extra Trees | Faster random-forest-style ensemble |
| `classifiers.mlp` | Neural Network (MLP) | Deep learning baseline |

#### `classifiers.compare(df, target)`

Benchmark every built-in classifier and receive a ranked leaderboard.

```python
from breezeml import classifiers, datasets

df = datasets.iris()
results = classifiers.compare(df, "species")
```

#### `classifiers.detailed_report(df, target)`

Returns confusion matrix, precision, recall, ROC-AUC, and the full classification report.

```python
info = classifiers.detailed_report(df, "species", algo="decision_tree")
print(info["accuracy"])
print(info["confusion_matrix"])
print(info["roc_auc"])
```

#### `classifiers.quick_tune(df, target, algo)`

Runs `RandomizedSearchCV` with curated search spaces for the selected classifier.

```python
model, params, report = classifiers.quick_tune(
    df, "species", algo="random_forest"
)
print(params)
print(report)
```

Supported algorithms: `logistic`, `svm`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting`, `adaboost`, `extra_trees`, `mlp`, plus optional `xgboost` and `lightgbm`

Aliases:

- `classifiers.logistic_regression`
- `classifiers.naive_bayes`

---

### `regressors` Module *(new in v0.3.0)*

All regressor functions share the same signature:

```python
model, report = regressors.<name>(df, target)
```

The standard regression report includes:

```python
{
    "r2": float,
    "mae": float,
    "rmse": float,
    "adjusted_r2": float,
    "mape": float,
}
```

#### Available Regressors

| Function | Algorithm | Notes |
|---|---|---|
| `regressors.linear` | Linear Regression | Simple baseline |
| `regressors.ridge` | Ridge Regression | L2 regularization |
| `regressors.lasso` | Lasso Regression | L1 regularization |
| `regressors.elastic_net` | Elastic Net | Hybrid L1 + L2 |
| `regressors.svr` | Support Vector Regression | Nonlinear baseline |
| `regressors.decision_tree` | Decision Tree Regressor | Interpretable |
| `regressors.random_forest` | Random Forest Regressor | Strong tabular baseline |
| `regressors.gradient_boosting` | Gradient Boosting Regressor | Often the strongest built-in option |
| `regressors.knn` | K-Nearest Neighbors Regressor | Non-parametric |
| `regressors.mlp` | Neural Network (MLP) Regressor | Deep learning baseline |

#### `regressors.compare(df, target)`

Benchmark every built-in regressor and rank them by R2.

```python
from breezeml import regressors, datasets

df = datasets.diabetes()
results = regressors.compare(df, "target")
```

#### `regressors.detailed_report(df, target)`

Returns richer diagnostics such as explained variance, residuals, and prediction-vs-actual pairs.

```python
from breezeml import regressors, datasets

df = datasets.diabetes()
info = regressors.detailed_report(df, "target", algo="random_forest")
print(info["r2"])
print(info["explained_variance"])
print(info["residuals"][:5])
```

#### `regressors.quick_tune(df, target, algo)`

Runs `RandomizedSearchCV` with curated search spaces for the selected regressor.

```python
from breezeml import regressors, datasets

df = datasets.diabetes()
model, params, report = regressors.quick_tune(
    df, "target", algo="decision_tree", n_iter=10, cv=3
)
print(params)
print(report)
```

Supported algorithms: `linear`, `ridge`, `lasso`, `elastic_net`, `svr`, `decision_tree`, `random_forest`, `gradient_boosting`, `knn`, `mlp`, plus optional `xgboost` and `lightgbm`

---

### `features` Module *(new in v0.3.0)*

Use `breezeml.features` to reduce noisy feature spaces, inspect model importances, and engineer stronger tabular inputs.

#### `features.select(df, target, method="mutual_info", k=10)`

```python
from breezeml import datasets, features

df = datasets.iris()
selected = features.select(df, "species", method="mutual_info", k=3)
print(selected.head())
```

#### `features.importance(model, df, target=None)`

```python
from breezeml import datasets, features, regressors

df = datasets.diabetes()
model, _ = regressors.random_forest(df, "target")
print(features.importance(model, df, target="target"))
```

#### `features.pca(df, n_components=0.95)` and `features.polynomial(df, degree=2, columns=None)`

```python
from breezeml import datasets, features

df = datasets.iris().drop(columns=["species"])
pca_df = features.pca(df, n_components=2)
poly_df = features.polynomial(df, degree=2, columns=df.columns[:2].tolist())
```

---

### Cascade Classification *(v0.2.6)*

A cascade chains multiple BreezeML classifiers into a hierarchical pipeline where each level narrows the prediction space. This is useful when a target has a natural hierarchy, such as sector -> group -> leaf code.

```python
from breezeml import classifiers
import joblib

m1, r1 = classifiers.linear_svm(X=X_train, y=y_sector, X_test=X_test, y_test=y_sector_test)
m2, r2 = classifiers.linear_svm(X=X_train, y=y_group, X_test=X_test, y_test=y_group_test)
m3, r3 = classifiers.linear_svm(X=X_train, y=y_code, X_test=X_test, y_test=y_code_test)

joblib.dump({"sector": m1, "group": m2, "code": m3}, "cascade_model.joblib")
```

---

### NLP and Semantic Embeddings *(v0.2.9)*

Convert raw text columns into dense semantic vectors with `sentence-transformers`.

```python
from breezeml.text import embed

df_dense = embed(df, text_columns=["review"])
model = fit(df_dense, target="sentiment")
```

---

### Explainability and Plotting *(v0.2.9)*

#### `explain.explain(model, df)`

Generate a SHAP summary plot for a trained model.

```python
from breezeml.explain import explain

explain(model, X_test)
```

#### `plot.confusion_matrix(model, X_test, y_test)` and `plot.roc_curve(model, X_test, y_test)`

Instant Matplotlib visualizations without the usual boilerplate.

```python
from breezeml.plot import confusion_matrix, roc_curve

confusion_matrix(model, X_test, y_test, cmap="Blues")
roc_curve(model, X_test, y_test)
```

#### `plot.compare_chart`, `plot.learning_curve`, and `plot.feature_importance` *(v0.3.0)*

```python
from breezeml import datasets, classifiers, plot

df = datasets.iris()
results = classifiers.compare(df, "species", show=False)
plot.compare_chart(results, metric="accuracy")
```

---

### `clustering` Module

```python
from breezeml import clustering, datasets

df = datasets.wine()
res = clustering.kmeans(df.drop(columns=["class"]), n_clusters=3)
print(res["silhouette"])
print(res["labels"][:10])
```

| Function | Algorithm |
|---|---|
| `clustering.kmeans(df, n_clusters)` | K-Means |
| `clustering.agglomerative(df, n_clusters)` | Agglomerative Hierarchical |
| `clustering.dbscan(df, eps, min_samples)` | DBSCAN |

---

### Built-in Datasets

| Function | Source | Target Column | Task |
|---|---|---|---|
| `datasets.iris()` | sklearn | `species` | Classification |
| `datasets.wine()` | sklearn | `class` | Classification |
| `datasets.breast_cancer()` | sklearn | `label` | Classification |
| `datasets.diabetes()` | sklearn | `target` | Regression |
| `datasets.california_housing()` | sklearn | `MedHouseVal` | Regression |
| `datasets.penguins()` | seaborn | `species` | Classification |
| `datasets.from_url(url)` | CSV URL | user-defined | Mixed |

---

## Examples

All examples live in [`/examples`](examples/). You can also open the Colab quickstart notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)

| File | Description |
|---|---|
| `breezeml_quickstart.ipynb` | Interactive notebook walkthrough |
| `test_classification.py` | Basic classification smoke test |
| `test_classifiers.py` | All 12 classifiers end-to-end |
| `test_clustering.py` | Clustering algorithms |
| `test_boost.py` | Optional XGBoost and LightGBM coverage |
| `test_features.py` | Feature engineering helpers |
| `test_regression.py` | Core regression pipeline |
| `test_regressors.py` | Regressor leaderboard, detailed report, and tuning coverage |
| `test_save_load.py` | Model persistence |
| `test_v020_features.py` | Broader feature coverage from earlier releases |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: breezeml` | Library not installed | `pip install breezeml` |
| `ValueError: columns do not match` | Feature mismatch at inference | Ensure prediction data uses the same columns as training |
| `ConvergenceWarning` | Linear or neural models did not converge | Increase `max_iter` or normalize features |
| `Version conflict` | Outdated dependencies | `pip install --upgrade scikit-learn pandas numpy` |

---

## Roadmap

- [x] Core `fit` / `predict` / `auto` API
- [x] 12 classifiers with unified interface
- [x] 10 regressors with leaderboard, detailed reports, and tuning *(v0.3.0)*
- [x] Classifier leaderboard (`compare`)
- [x] Regressor leaderboard (`regressors.compare`) *(v0.3.0)*
- [x] Cross-validation support across classifiers and regressors *(v0.3.0)*
- [x] Hyperparameter auto-tuning (`quick_tune`)
- [x] Regression hyperparameter tuning (`regressors.quick_tune`) *(v0.3.0)*
- [x] Detailed evaluation reports (confusion matrix, ROC-AUC)
- [x] Detailed regression reports (`adjusted_r2`, `mape`, residuals) *(v0.3.0)*
- [x] Feature engineering helpers (`select`, `importance`, `pca`, `polynomial`) *(v0.3.0)*
- [x] Optional XGBoost and LightGBM integration *(v0.3.0)*
- [x] Clustering (K-Means, DBSCAN, Agglomerative)
- [x] Cascade classification - hierarchical multi-level pipelines *(v0.2.6)*
- [x] External test set support (`X_test` / `y_test`) on all classifiers *(v0.2.6)*
- [x] Macro F1 in all report dicts *(v0.2.6)*
- [x] Native semantic text embeddings (`breezeml.text`) *(v0.2.9)*
- [x] `explain()` - SHAP-based feature importance *(v0.2.9)*
- [x] Native plotting (`plot_confusion_matrix`, `plot_roc`) *(v0.2.9)*
- [x] `export()` - standalone sklearn scripts, zero lock-in *(v1.0)*
- [x] `card()` - auto-generated honest model cards *(v1.0)*
- [x] Teaching narration (`explain_decisions=True`) *(v1.0)*
- [x] `deploy()` - one-line FastAPI + Docker serving *(v1.0)*
- [x] MCP server for AI agents (`breezeml-mcp`) *(v1.0)*
- [x] CI-enforced dependency contract - 4 core deps, always *(v1.0)*
- [ ] Additional datasets (Titanic, MNIST subset)
- [ ] `BreezeAutoML` - full AutoML via Optuna integration
- [ ] Time-series helpers (`breezeml.timeseries`)
- [ ] ONNX export for categorical pipelines

---

## Benchmarks

Measured 2026-07-05, Windows 11, Python 3.11, same machine and venv. Reproduce with `python benchmarks/run_benchmarks.py`.

| | BreezeML | PyCaret | LazyPredict |
|---|---|---|---|
| Fresh install | **2m 36s / 274 MB** | 6m 21s / 952 MB | (shared venv) |
| Cold import | **3.1s** | 6.9s | 7.2s |
| Wine leaderboard | 9.0s | 19.3s | **3.7s** |
| Best accuracy | **1.000** | 0.984 | **1.000** |
| User LOC | **3** | 5 | 8 |
| Zero lock-in export | **yes** | no | no |

Full methodology and honest caveats: [docs/benchmarks.md](docs/benchmarks.md).

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
git clone https://github.com/venomez-viper/breezeml.git
cd breezeml
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

All pull requests should:

- Pass the existing CI suite
- Include tests for new functionality
- Follow the existing docstring style

---

## License

MIT © 2025 **Akash Anipakalu Giridhar**

See [LICENSE](LICENSE) for full terms.

---

<div align="center">

Maintained by [Akash Anipakalu Giridhar](https://github.com/venomez-viper)

</div>
