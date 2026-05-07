# BreezeML v0.1.1 Release Notes (Initial Release)

## 🚀 Key Features

### Initial PyPI Release
Welcome to BreezeML! This is the first public release of the beginner-friendly machine learning library built on scikit-learn.

### Core API
Introduced the clean, single-function API surface:
- `fit(df, target)`
- `predict(model, X)`
- `auto(df, target)`
- `from_csv(path, target)`
- `report(model, df)`
- `save(model, path)`
- `load(path)`

### EasyModel Wrapper
Added the `EasyModel` wrapper class that seamlessly encapsulates the underlying scikit-learn pipeline, task type (classification/regression), and the target column name to prevent user errors during inference.

### Built-in Datasets
Included standard scikit-learn datasets pre-formatted as Pandas DataFrames:
- `datasets.iris()`
- `datasets.wine()`
- `datasets.breast_cancer()`
- `datasets.diabetes()`

---
*Created by Akash Anipakalu Giridhar 🔥✨*
