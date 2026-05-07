# Examples

## Core classification

```python
from breezeml import datasets, fit, predict

df = datasets.iris()
model = fit(df, "species")
preds = predict(model, df.drop(columns=["species"]))
```

## Core regression

```python
from breezeml import datasets, regressors

df = datasets.diabetes()
model, report = regressors.gradient_boosting(df, "target")
print(report)
```

## Cross-validation

```python
from breezeml import classifiers, datasets

df = datasets.iris()
model, report = classifiers.logistic(df, "species", cv=5)
print(report)
```

## Compare models

```python
from breezeml import classifiers, datasets

df = datasets.iris()
results = classifiers.compare(df, "species")
```

```python
from breezeml import regressors, datasets

df = datasets.diabetes()
results = regressors.compare(df, "target")
```

## Feature engineering

```python
from breezeml import datasets, features

df = datasets.iris()
selected = features.select(df, "species", method="mutual_info", k=3)
```

```python
from breezeml import datasets, features

df = datasets.iris().drop(columns=["species"])
pca_df = features.pca(df, n_components=2)
poly_df = features.polynomial(df, degree=2, columns=df.columns[:2].tolist())
```

## Plotting

```python
from breezeml import datasets, classifiers, plot

df = datasets.iris()
results = classifiers.compare(df, "species", show=False)
plot.compare_chart(results, metric="accuracy")
```

## Optional boosting backends

```python
from breezeml import classifiers, regressors

# Requires: pip install "breezeml[boost]"
# classifiers.xgboost(...)
# regressors.lightgbm(...)
```
