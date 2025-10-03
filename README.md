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

## 📦 Install (Windows/Mac/Linux)
```bash
pip install breezeml
```

---

## ⏱️ 60-Second Quickstart
```python
from breezeml import datasets, fit, predict, creator

print(creator())  # Easter Egg 🔥✨
df = datasets.iris()              # small flower dataset
model = fit(df, "species")        # train
print(predict(model, df.drop(columns=["species"]))[:5])  # predict
```

---

## 🔮 Auto-Magic Mode
```python
from breezeml import auto, datasets

df = datasets.diabetes()
model, report = auto(df, "target")   # auto picks regression here
print(report)                         # r2, mae, rmse
```

---

## 📄 Use Your Own CSV (the thing most people want)
1. Create a `data.csv` with one **target** column (the answer).  
2. Train + get a quick report:
```python
from breezeml import from_csv
model, report = from_csv("data.csv", target="price")
print(report)
```
3. Predict on new rows (no target column):
```python
from breezeml import fit, predict
import pandas as pd

df = pd.read_csv("data.csv")
model = fit(df, "price")

new_df = pd.read_csv("new_data.csv")
print(predict(model, new_df)[:5])
```

**CSV tips**
- Column names in `new_data.csv` must match the training columns (except the target).
- Missing values are handled automatically.
- Text columns are encoded automatically.

---

## 🧠 What models are used?
BreezeML picks sensible defaults:
- **Classification:** RandomForest (or LogisticRegression when asked)
- **Regression:** RandomForestRegressor (or LinearRegression when asked)

Choose explicitly:
```python
from breezeml import classify, regress
m, report = classify(df, "species", algo="logistic")
m, report = regress(df, "price",   algo="forest")
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

## 🧪 Tiny “It Works!” Snippets

**Classification (Iris):**
```python
from breezeml import datasets, fit, predict
df = datasets.iris()
model = fit(df, "species")
print(predict(model, df.drop(columns=["species"]))[:10])
```

**Regression (Diabetes):**
```python
from breezeml import datasets, fit, predict
df = datasets.diabetes()
model = fit(df, "target")
print(predict(model, df.drop(columns=["target"]))[:10])
```

---

## 🛠️ Troubleshooting (copy & try)
- “**Module not found**”: `pip install breezeml`
- “**Powershell script disabled**” (Windows venv):  
  Run PowerShell as Admin → `Set-ExecutionPolicy RemoteSigned` → `Y`
- “**Columns do not match**”: Make sure `new_data.csv` has the same feature columns as training data.
- “**Version issue**”: `pip install --upgrade scikit-learn pandas numpy`

---

## 🗺️ Project Status
- Current: **v0.1.1**
- Roadmap:
  - `quick_tune()` (1-line hyperparameter search)
  - `explain()` (feature importance, partial dependence)
  - `plot_confusion()` (auto confusion matrix)
  - More datasets & presets (`fast`, `balanced`, `accurate`)

---

## 🤝 Contribute
PRs welcome!  
- Run examples: `python examples/test_classification.py`  
- Optional tests: `pytest -q`  
- See **CHANGELOG.md** for version history.

---

## 📜 License
MIT © 2025 Akash Anipakalu Giridhar 🔥✨

---

### Handy links
- PyPI: https://pypi.org/project/breezeml/  
- Issues: https://github.com/venomez-viper/breezeml/issues  
- Colab demo: [Open in Colab](https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb)
