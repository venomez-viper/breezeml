# BreezeML v0.3.0 Release Notes (The "Major Expansion" Update)

## Key Improvements & Features

### Full Regression Stack
Added a dedicated `breezeml.regressors` module with 10 one-line regressors:

- `linear`
- `ridge`
- `lasso`
- `elastic_net`
- `svr`
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `knn`
- `mlp`

Each regressor now supports:

- one-line training
- leaderboard benchmarking via `regressors.compare()`
- detailed diagnostics via `regressors.detailed_report()`
- hyperparameter search via `regressors.quick_tune()`

### Cross-Validation Across the Training Stack
Classifier and regressor training helpers now accept `cv=` and can return mean/std metrics directly in their report dictionaries. This makes BreezeML much more useful for proper model selection instead of single-split prototyping only.

### New Feature Engineering Module
Added `breezeml.features` with:

- `select()` for top-k feature selection
- `importance()` for model-based or permutation importance
- `pca()` for dimensionality reduction
- `polynomial()` for interaction and polynomial feature expansion

### Optional XGBoost and LightGBM Integration
BreezeML can now plug into modern gradient boosting libraries when installed:

- `classifiers.xgboost()`
- `classifiers.lightgbm()`
- `regressors.xgboost()`
- `regressors.lightgbm()`

These models also flow into the compare and tuning systems automatically when the packages are available.

### Expanded Plotting Helpers
Added:

- `plot.compare_chart()`
- `plot.learning_curve()`
- `plot.feature_importance()`

These sit alongside the existing confusion matrix and ROC curve helpers.

### More Built-In Datasets
Added:

- `datasets.california_housing()`
- `datasets.penguins()`
- `datasets.from_url()`

### Better Behavior in Restricted Environments
Added serial fallback paths when process-based parallel execution is blocked, so compare/tuning helpers continue to work in constrained Windows environments.

---
*Created by Akash Anipakalu Giridhar*
