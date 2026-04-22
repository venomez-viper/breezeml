# 📜 Changelog

All notable changes to BreezeML are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

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
