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

## 🧮 Built-in Classifiers (v0.1.2)
Use popular classifiers in one line:
```python
from breezeml import classifiers, datasets
df = datasets.iris()
model, report = classifiers.gaussian_nb(df, "species")
print(report)  # {'accuracy': ..., 'f1': ...}
```
Available: `classifiers.logistic`, `classifiers.svm`, `classifiers.linear_svm`, `classifiers.gaussian_nb`, `classifiers.multinomial_nb`, `classifiers.decision_tree`, `classifiers.random_forest`

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
- “Module not found”: `pip install breezeml`
- “Columns do not match”: new data must have same feature names as training
- “Version issue”: `pip install --upgrade scikit-learn pandas numpy`

---

## 🗺️ Project Status
- Current: **v0.1.2**
- Roadmap: `quick_tune()`, `explain()`, plots, more datasets

---

## 🤝 Contribute
PRs welcome! See CHANGELOG and examples.

## 📜 License
MIT © 2025 Akash Anipakalu Giridhar 🔥✨
