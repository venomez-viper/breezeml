<div align="center">

<img src="https://raw.githubusercontent.com/venomez-viper/breezeml/main/docs/assets/banner.png" alt="BreezeML Banner" width="800"/>

# BreezeML

**Machine learning without the boilerplate.**

*Train, evaluate, and save a model in a single Python expression.*

<br/>

[![PyPI version](https://badge.fury.io/py/breezeml.svg)](https://pypi.org/project/breezeml/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/breezeml?color=blue&label=PyPI%20Downloads)](https://pypi.org/project/breezeml/)
![CI Status](https://github.com/venomez-viper/breezeml/actions/workflows/ci.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/venomez-viper/breezeml?color=green)](https://github.com/venomez-viper/breezeml/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)
[![scikit-learn](https://img.shields.io/badge/built%20on-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

<br/>

[**Getting Started**](#-installation) · [**API Reference**](#-api-reference) · [**Examples**](#-examples) · [**Contributing**](CONTRIBUTING.md) · [**Changelog**](CHANGELOG.md)

</div>

---

## Overview

BreezeML is a high-level machine learning library built on top of **scikit-learn**, designed to eliminate boilerplate while preserving full statistical rigor. Whether you are a student exploring ML for the first time or a practitioner who needs a fast prototyping layer, BreezeML handles preprocessing, model selection, hyperparameter search, and evaluation — all behind a clean, expressive API.

```python
from breezeml import datasets, fit, predict

df    = datasets.iris()
model = fit(df, "species")
preds = predict(model, df.drop(columns=["species"]))
```

That's it. No manual train/test splits. No encoder boilerplate. No metric aggregation.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Auto task detection** | Automatically selects classification or regression based on the target column |
| **12 classifiers** | From Logistic Regression to Neural Nets, available in one function call |
| **Classifier leaderboard** | `classifiers.compare()` benchmarks all 12 models and ranks them by accuracy and F1 |
| **Auto hyperparameter tuning** | `quick_tune()` runs `RandomizedSearchCV` with curated parameter grids |
| **Detailed evaluation reports** | Confusion matrix, precision, recall, ROC-AUC, and full classification report |
| **3 clustering algorithms** | K-Means, Agglomerative, DBSCAN — all one-liners |
| **Built-in benchmark datasets** | Iris, Wine, Breast Cancer, Diabetes — ready in one line |
| **Seamless CSV ingestion** | `from_csv("data.csv", target="price")` handles loading, preprocessing, and training |
| **Model persistence** | `save()` / `load()` powered by `joblib` |
| **Fully type-hinted** | Clean, IDE-friendly API surface |

---

## 📐 Architecture

```
breezeml/
├── breezeml.py        # Core: fit, predict, auto, from_csv, save, load
├── classifiers.py     # 12 classifiers + compare, detailed_report, quick_tune
├── clustering.py      # kmeans, agglomerative, dbscan
└── __init__.py        # Public API surface
```

**Internal Pipeline (all algorithms)**

```
Raw DataFrame
     │
     ▼
┌─────────────────────────────────────────┐
│  ColumnTransformer (Auto-detected)      │
│  ├── Numeric  → Median Imputer + Scaler │
│  └── Categorical → Mode Imputer + OHE   │
└────────────────┬────────────────────────┘
                 │
                 ▼
        sklearn Estimator
                 │
                 ▼
          EasyModel wrapper
     (pipeline + task + target)
```

---

## 📦 Installation

**Stable release (recommended)**
```bash
pip install breezeml
```

**Latest from source**
```bash
git clone https://github.com/venomez-viper/breezeml.git
cd breezeml
pip install -e .
```

**Requirements:** Python ≥ 3.8, scikit-learn, pandas, numpy, joblib

---

## 🚀 Quickstart

### Classification in 3 lines
```python
from breezeml import datasets, fit, predict

df    = datasets.iris()
model = fit(df, "species")
print(predict(model, df.drop(columns=["species"]))[:5])
# [0, 0, 0, 0, 0]
```

### Auto mode (classification or regression — chosen for you)
```python
from breezeml import auto, datasets

df = datasets.diabetes()
model, report = auto(df, "target")
print(report)
# {'r2': 0.4526, 'mae': 44.23, 'rmse': 57.81}
```

### Load your own CSV
```python
from breezeml import from_csv

model, report = from_csv("sales_data.csv", target="revenue")
print(report)
```

---

## 📖 API Reference

### Core Functions

#### `fit(df, target)` → `EasyModel`
Train a model. Task (classification vs regression) is inferred automatically from the target column.

```python
model = fit(df, "target_column")
```

#### `predict(model, X)` → `np.ndarray`
Run inference on new data.

```python
predictions = predict(model, new_df)
```

#### `auto(df, target)` → `(EasyModel, dict)`
Same as `fit`, but also returns an evaluation report.

```python
model, report = auto(df, "target_column")
```

#### `from_csv(path, target)` → `(EasyModel, dict)`
Load a CSV, train, and evaluate in one call.

```python
model, report = from_csv("data.csv", target="label")
```

#### `save(model, path)` / `load(path)`
Persist and restore any trained `EasyModel`.

```python
save(model, "my_model.joblib")
model = load("my_model.joblib")
```

---

### `classifiers` Module

All classifier functions share the same signature:

```python
model, report = classifiers.<name>(df, target)
# report = {'accuracy': float, 'f1': float}
```

#### Available Classifiers

| Function | Algorithm | Notes |
|---|---|---|
| `classifiers.logistic` | Logistic Regression | Linear baseline |
| `classifiers.svm` | SVM (RBF kernel) | Robust for small–medium datasets |
| `classifiers.linear_svm` | Linear SVM | Scales to large datasets |
| `classifiers.gaussian_nb` | Gaussian Naïve Bayes | Fast; good for continuous features |
| `classifiers.multinomial_nb` | Multinomial Naïve Bayes | Best for text/count features |
| `classifiers.decision_tree` | Decision Tree | Fully interpretable |
| `classifiers.random_forest` | Random Forest | Strong general-purpose baseline |
| `classifiers.knn` | K-Nearest Neighbors | Non-parametric |
| `classifiers.gradient_boosting` | Gradient Boosting | High accuracy on tabular data |
| `classifiers.adaboost` | AdaBoost | Ensemble boosting |
| `classifiers.extra_trees` | Extra Trees | Faster than Random Forest |
| `classifiers.mlp` | Neural Network (MLP) | Deep learning baseline |

#### `classifiers.compare(df, target)` — Leaderboard

Benchmark every classifier and receive a ranked comparison table.

```python
from breezeml import classifiers, datasets

df      = datasets.iris()
results = classifiers.compare(df, "species")
```

```
🏆 BreezeML Classifier Leaderboard — target: 'species'
Rank  Classifier               Accuracy    F1
──────────────────────────────────────────────────
1     Random Forest             1.0000    1.0000
2     Extra Trees               1.0000    1.0000
3     Gradient Boosting         0.9667    0.9667
4     K-Nearest Neighbors       0.9667    0.9667
...
```

#### `classifiers.detailed_report(df, target)` — Full Evaluation

Returns confusion matrix, per-class precision/recall, and ROC-AUC.

```python
info = classifiers.detailed_report(df, "species")

print(info["accuracy"])          # 0.9667
print(info["precision"])         # 0.9683
print(info["recall"])            # 0.9667
print(info["roc_auc"])           # 0.9958
print(info["confusion_matrix"])  # [[10, 0, 0], [0, 9, 1], ...]
```

#### `classifiers.quick_tune(df, target, algo)` — Hyperparameter Search

Runs `RandomizedSearchCV` with a curated search space for the chosen algorithm. Returns the best model, best parameters, and evaluation report.

```python
model, params, report = classifiers.quick_tune(
    df, "species", algo="random_forest"
)
print(params)   # {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
print(report)   # {'accuracy': 1.0, 'f1': 1.0}
```

Supported algorithms: `logistic`, `svm`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting`, `adaboost`, `extra_trees`, `mlp`

---

### `clustering` Module

```python
from breezeml import clustering, datasets

df  = datasets.wine()
res = clustering.kmeans(df.drop(columns=["class"]), n_clusters=3)

print(res["silhouette"])   # 0.2841
print(res["labels"][:10]) # [0, 0, 0, 2, 0, 0, 1, 0, 0, 0]
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

---

## 🧪 Examples

All examples are in [`/examples`](examples/). Run them directly or open the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)

| File | Description |
|---|---|
| `breezeml_quickstart.ipynb` | Interactive notebook walkthrough |
| `test_classification.py` | Basic classification smoke test |
| `test_classifiers.py` | All 12 classifiers end-to-end |
| `test_clustering.py` | Clustering algorithms |
| `test_regression.py` | Regression pipeline |
| `test_save_load.py` | Model persistence |
| `test_v020_features.py` | Full v0.2.0 feature coverage |

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: breezeml` | Library not installed | `pip install breezeml` |
| `ValueError: columns do not match` | Feature mismatch at inference | Ensure prediction data has the same column names as training data |
| `ConvergenceWarning` | Logistic Regression not converged | Increase `max_iter` or normalize features |
| `Version conflict` | Outdated dependencies | `pip install --upgrade scikit-learn pandas numpy` |

---

## 🗺️ Roadmap

- [x] Core `fit` / `predict` / `auto` API
- [x] 12 classifiers with unified interface
- [x] Classifier leaderboard (`compare`)
- [x] Hyperparameter auto-tuning (`quick_tune`)
- [x] Detailed evaluation reports (confusion matrix, ROC-AUC)
- [x] Clustering (K-Means, DBSCAN, Agglomerative)
- [ ] `explain()` — SHAP-based feature importance
- [ ] Native plotting (`plot_confusion_matrix`, `plot_roc`)
- [ ] Additional datasets (Titanic, MNIST subset)
- [ ] `Pipeline.export()` — export trained pipeline as Python script
- [ ] `BreezeAutoML` — full AutoML via Optuna integration

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
git clone https://github.com/venomez-viper/breezeml.git
cd breezeml
pip install -e ".[dev]"
pytest examples/
```

All PRs must:
- Pass the existing CI suite
- Include tests for new functionality
- Follow the existing docstring style

---

## 📜 License

MIT © 2025 **Akash Anipakalu Giridhar**

See [LICENSE](LICENSE) for full terms.

---

<div align="center">

Maintained by [Akash Anipakalu Giridhar](https://github.com/venomez-viper)

</div>
