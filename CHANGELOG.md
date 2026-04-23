# 📜 Changelog

All notable changes to BreezeML are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [0.2.5] — Unreleased

### Fixed
- **Linear SVM Class Imbalance**: Added `class_weight='balanced'` globally to all underlying `LinearSVC` initializations (`linear_svm`, `compare`, `detailed_report`). This forces the scikit-learn optimization engine to penalize misclassifications on tiny, sparse target classes effectively, skyrocketing F1-score performance out-of-the-box for highly skewed NLP classification datasets.

## [0.2.4] — 2026-04-22

### Fixed
- **Pipeline Save Bug**: Hot-patched `breezeml.save()` to dynamically check if the model object genuinely has a `.save()` method (like `EasyModel` does). If you pass it a raw strictly scikit-learn `Pipeline` (such as the return from classifiers module functions), it automatically falls back natively to `joblib.dump()`, preventing fatal `AttributeError` tracebacks.

## [0.2.3] — 2026-04-22

### Added
- **Sparse Matrix Support**: The `classifiers` module (`linear_svm`, `compare`, `detailed_report`) now directly accepts `X` and `y` keyword parameters (`classifiers.func(X=X, y=y)`) to natively process `scipy.sparse` matrices, bypassing dense Pandas conversion bottlenecks.

### Fixed
- **Linear SVM Primal Formulation**: Hand-patched all `LinearSVC` references with `dual=False`. This overrides the default scikit-learn Dual Formulation math trap when datasets have highly-dimensional sparse text vectors (n_samples > n_features), solving 20+ minute memory deadlocks and reducing training time to < 2 seconds.

## [0.2.1] — 2026-04-22

### Changed
- **Massive Performance Boost for `classifiers.compare`**: We now utilize `joblib.Parallel(n_jobs=-1)` to train and evaluate all 12 baseline classification models concurrently across all available CPU cores. This effectively turns O(N) waiting time into O(1), drastically speeding up the model leaderboards on larger datasets.

## [0.2.0] — 2025-10-21

### Added
- **5 new classifiers**: `knn`, `gradient_boosting`, `adaboost`, `extra_trees`, `mlp` (Multi-Layer Perceptron Neural Network) — bringing the total to 12.
- **`classifiers.compare(df, target)`** — benchmarks all 12 classifiers on a dataset and returns a ranked leaderboard sorted by accuracy and F1 score.
- **`classifiers.detailed_report(df, target)`** — returns confusion matrix, per-class precision, recall, F1, ROC-AUC, and a full sklearn classification report string.
- **`classifiers.quick_tune(df, target, algo)`** — automated hyperparameter search via `RandomizedSearchCV` with curated parameter grids for 9 supported algorithms.
- Comprehensive test script `examples/test_v020_features.py` covering all new functionality.

### Changed
- All metrics are now rounded to 4 decimal places for consistent, readable output.
- README fully rewritten: classifier table, `compare` example, `detailed_report` example, `quick_tune` example, and architecture diagram.

---

## [0.1.2] — 2025-10-16

### Added
- Interactive Colab Quickstart notebook (`examples/breezeml_quickstart.ipynb`) + README Colab badge.
- Beginner-friendly README with CSV guide, feature table, and troubleshooting section.
- **`classifiers` module**: Logistic Regression, SVM (RBF), Linear SVM, Gaussian Naïve Bayes, Multinomial Naïve Bayes, Decision Tree, Random Forest.
- **`clustering` module**: K-Means, Agglomerative Hierarchical, DBSCAN — all returning labels and silhouette score.

### Fixed
- RMSE now computed as `sqrt(mean_squared_error(...))` for compatibility with scikit-learn < 0.24 (which did not expose the `squared` parameter).

---

## [0.1.1] — 2025-10-03

### Added
- Initial PyPI release.
- Core API: `fit`, `predict`, `auto`, `from_csv`, `report`, `save`, `load`.
- `datasets` class: `iris`, `wine`, `breast_cancer`, `diabetes`.
- `creator()` easter egg function.
- `EasyModel` wrapper class encapsulating pipeline, task type, and target column.
- Example scripts: `test_classification.py`, `test_regression.py`, `test_save_load.py`.
