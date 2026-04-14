# 🌬️ BreezeML 🔥✨
**If you can load a CSV, you can train a model.**  
Beginner-friendly machine learning on top of scikit-learn — zero boilerplate.

[![PyPI version](https://badge.fury.io/py/breezeml.svg)](https://pypi.org/project/breezeml/)
![BreezeML CI](https://github.com/venomez-viper/breezeml/actions/workflows/ci.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/venomez-viper/breezeml)](https://github.com/venomez-viper/breezeml/releases)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)

---

## 🧩 What is BreezeML?
- **One line to train** a model (`fit`) and **one line to predict** (`predict`).  
- **Auto mode** chooses classification vs regression.  
- **12 classifiers** in one line — from Logistic Regression to Neural Nets.  
- **Compare all classifiers** instantly with a ranked leaderboard.  
- **Auto-tune** hyperparameters with `quick_tune()`.  
- **Detailed reports** with confusion matrix, precision, recall, ROC-AUC.  
- **Built-in datasets**: Iris, Wine, Breast Cancer, Diabetes.  
- **Save/Load** models easily.  
- **Kid-friendly API** with sensible defaults.

---

## 📦 Install
```bash
pip install breezeml
```

---

## ⏱️ 60-Second Quickstart
```python
from breezeml import datasets, fit, predict, creator

print(creator())  # Easter Egg 🔥✨
df = datasets.iris()
model = fit(df, "species")
print(predict(model, df.drop(columns=["species"]))[:5])
```

---

## 🔮 Auto-Magic Mode
```python
from breezeml import auto, datasets

df = datasets.diabetes()
model, report = auto(df, "target")
print(report)  # r2, mae, rmse
```

---

## 🧮 Classifiers (v0.2.0)
Use popular classifiers in one line:
```python
from breezeml import classifiers, datasets
df = datasets.iris()
model, report = classifiers.knn(df, "species")
print(report)  # {'accuracy': ..., 'f1': ...}
```
**Available classifiers:**
| Function | Algorithm |
|---|---|
| `classifiers.logistic` | Logistic Regression |
| `classifiers.svm` | Support Vector Machine (RBF) |
| `classifiers.linear_svm` | Linear SVM |
| `classifiers.gaussian_nb` | Gaussian Naïve Bayes |
| `classifiers.multinomial_nb` | Multinomial Naïve Bayes |
| `classifiers.decision_tree` | Decision Tree |
| `classifiers.random_forest` | Random Forest |
| `classifiers.knn` | K-Nearest Neighbors |
| `classifiers.gradient_boosting` | Gradient Boosting |
| `classifiers.adaboost` | AdaBoost |
| `classifiers.extra_trees` | Extra Trees |
| `classifiers.mlp` | Multi-Layer Perceptron (Neural Net) |

---

## 🏆 Compare All Classifiers (v0.2.0)
Find the best model for your data in one line:
```python
from breezeml import classifiers, datasets

df = datasets.iris()
results = classifiers.compare(df, "species")
# Prints a ranked leaderboard with accuracy & F1 for all 11 classifiers!
```
Output:
```
🏆 BreezeML Classifier Leaderboard — target: 'species'
Rank  Classifier               Accuracy    F1
───────────────────────────────────────────────────────
1     Random Forest             1.0000      1.0000
2     Gradient Boosting         0.9667      0.9667
...
```

---

## 📊 Detailed Report (v0.2.0)
Get confusion matrix, precision, recall, and ROC-AUC:
```python
from breezeml import classifiers, datasets

df = datasets.iris()
info = classifiers.detailed_report(df, "species")
print(info["accuracy"])           # 0.9667
print(info["precision"])          # 0.9683
print(info["recall"])             # 0.9667
print(info["roc_auc"])            # 0.9958
print(info["confusion_matrix"])   # [[10, 0, 0], [0, 9, 1], ...]
```

---

## ⚡ Quick Tune (v0.2.0)
Auto hyperparameter tuning — finds the best settings for you:
```python
from breezeml import classifiers, datasets

df = datasets.iris()
model, params, report = classifiers.quick_tune(df, "species", algo="random_forest")
print(params)   # {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
print(report)   # {'accuracy': 1.0, 'f1': 1.0}
```
Supports: `"logistic"`, `"svm"`, `"knn"`, `"decision_tree"`, `"random_forest"`, `"gradient_boosting"`, `"adaboost"`, `"extra_trees"`, `"mlp"`

---

## 🧊 Clustering (v0.1.2)
Unsupervised learning in one line:
```python
from breezeml import clustering, datasets
df = datasets.wine()
res = clustering.kmeans(df.drop(columns=["class"]), n_clusters=3)
print(res["silhouette"], res["labels"][:10])
```
Available: `clustering.kmeans`, `clustering.agglomerative`, `clustering.dbscan`

---

## 📄 Use Your Own CSV
```python
from breezeml import from_csv
model, report = from_csv("data.csv", target="price")
print(report)
```

---

## 💾 Save & Load
```python
from breezeml import save, load, datasets, fit
df = datasets.iris()
model = fit(df, "species")
save(model, "iris_model.joblib")
loaded = load("iris_model.joblib")
```

---

## 🛠️ Troubleshooting
- "Module not found": `pip install breezeml`
- "Columns do not match": new data must have same feature names as training
- "Version issue": `pip install --upgrade scikit-learn pandas numpy`

---

## 🗺️ Project Status
- Current: **v0.2.0**
- Roadmap: `explain()`, plots, more datasets

---

## 🤝 Contribute
PRs welcome! See CHANGELOG and examples.

## 📜 License
MIT © 2025 Akash Anipakalu Giridhar 🔥✨
