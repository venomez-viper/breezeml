# 📜 Changelog

All notable changes to this project will be documented here.
Format based on Keep a Changelog.

## [0.2.0] - 2025-10-21
### Added
- **5 new classifiers**: `knn`, `gradient_boosting`, `adaboost`, `extra_trees`, `mlp` (Neural Net).
- **`classifiers.compare()`** — run all classifiers on a dataset and get a ranked leaderboard in one line.
- **`classifiers.detailed_report()`** — confusion matrix, precision, recall, ROC-AUC, full classification report.
- **`classifiers.quick_tune()`** — auto hyperparameter tuning via RandomizedSearchCV with built-in param grids.
- Test script for all v0.2.0 features.

### Changed
- Metrics now rounded to 4 decimal places for cleaner output.
- README fully rewritten with classifier table, compare example, detailed_report example, quick_tune example.

## [0.1.2] - 2025-10-16
### Added
- Colab Quickstart notebook + README badge.
- Beginner-friendly README with CSV guide and troubleshooting.
- **New modules**: `classifiers` (LogReg, SVM, Naïve Bayes, Decision Tree, Random Forest) and `clustering` (KMeans, Agglomerative, DBSCAN).

### Fixed
- RMSE computed as sqrt(MSE) for wider scikit-learn compatibility.

## [0.1.1] - 2025-10-03
- First PyPI release with fit/predict/auto, datasets, save/load, examples, easter egg.
