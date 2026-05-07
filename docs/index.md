# BreezeML

Machine learning without the boilerplate.

BreezeML is a high-level wrapper around scikit-learn built for fast experimentation without throwing away the important parts of a real ML workflow. With `0.3.0`, the library grows from a classifier-heavy toolkit into a broader modeling layer with:

- 12 classifiers
- 10 regressors
- built-in comparison leaderboards
- hyperparameter search helpers
- cross-validation support
- feature engineering utilities
- optional XGBoost and LightGBM integration
- plotting and explainability helpers

## Quickstart

```python
from breezeml import datasets, auto

df = datasets.diabetes()
model, report = auto(df, "target")
print(report)
```

## What is new in 0.3.0

- A dedicated `breezeml.regressors` module
- `cv=` support across the training stack
- `breezeml.features` for feature selection, PCA, and polynomial expansion
- optional XGBoost and LightGBM hooks
- more plotting helpers
- more built-in datasets

## Install

```bash
pip install breezeml
```

Optional extras:

```bash
pip install "breezeml[boost]"
pip install "breezeml[datasets]"
pip install "breezeml[all]"
```

## Where to go next

- Read the [examples](examples.md)
- Browse the API reference
- Review the [README](https://github.com/venomez-viper/breezeml/blob/main/README.md)
- Check the [CHANGELOG](https://github.com/venomez-viper/breezeml/blob/main/CHANGELOG.md)
