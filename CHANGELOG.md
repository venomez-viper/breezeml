# Changelog

All notable changes to BreezeML are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [1.1.0] - 2026-07-06

### Added
- **BreezeAutoML**: `breezeml.automl(df, target, time_budget=60)` screens every built-in model with cross-validation, tunes the top 3 with budget-sized `RandomizedSearchCV`, and reports honest holdout metrics from a fresh stratified split. Search history stored in `model.meta["automl"]`; `card()`, `export()`, `deploy()`, and narration all work on the result.
- **Optuna backend**: `backend="optuna"` switches tuning to TPE search. New `[automl]` extra.

### Changed
- Version bump to 1.1.0; `automl` added to the public API.

## [1.0.1] - 2026-07-05

### Changed
- **Repositioning**: BreezeML is now described everywhere as a beginner-friendly, production-aware ML workflow layer for students, analysts, and AI agents. Updated PyPI metadata, README, docs, and module docstrings.
- **Python 3.9 fix**: dependency contract test now works on Python 3.9 (`sys.stdlib_module_names` fallback via `sysconfig`).
- **Publish safety**: the PyPI publish workflow now runs ruff and the full test suite before building and uploading.

## [1.0.0] - 2026-07-05

The "legendary" release: zero lock-in, honest reporting, one-line deployment, and first-class support for AI agents.

### Added
- **`export()` / `model.export()`**: Generate a standalone scikit-learn training script that reproduces the exact trained pipeline (imputers, scaler, encoder, estimator, seed, split) with zero breezeml imports. Graduate from BreezeML anytime.
- **`card()` / `model.card()`**: Auto-generated markdown model cards with data profile, metrics, every pipeline decision explained, and auto-detected caveats (small data, class imbalance, heavy imputation, drift and fairness warnings).
- **Teaching narration**: `explain_decisions=True` on `auto()` / `classify()` / `regress()` (and `model.explain_decisions()`) narrates every automatic pipeline decision in plain English, generated from measured facts about your data.
- **`deploy()` / `model.deploy()`**: One line writes a complete serving directory - FastAPI app (`/predict`, `/health`, Swagger docs), Dockerfile, requirements, and the raw sklearn pipeline. The deployed app never imports breezeml.
- **ONNX export**: `breezeml.deploy.to_onnx()` for numeric pipelines via the `[onnx]` extra.
- **MCP server (`breezeml-mcp`)**: A Model Context Protocol server exposing `inspect_data`, `compare`, `train`, `predict`, `explain`, `model_card`, `export`, `deploy`, and `save` as agent tools. AI agents get BreezeML's statistical guardrails instead of hand-rolled sklearn.
- **Dependency contract**: CI-enforced test guaranteeing core `import breezeml` needs only scikit-learn, pandas, numpy, and joblib. "4 dependencies. Always."
- **Benchmarks**: `benchmarks/run_benchmarks.py` measuring import time, leaderboard time, accuracy, and user LOC against PyCaret and LazyPredict.
- **Training metadata**: Models trained via the core API now carry a `meta` dict (data profile, decisions, seed, metrics) powering cards, narration, and export.
- **Docs**: New guides for export, model cards, deployment, and the MCP server.

### Changed
- **Version**: 1.0.0; development status is now Production/Stable.
- **New extras**: `[deploy]` (fastapi, uvicorn), `[onnx]` (skl2onnx), `[mcp]` (mcp SDK).

## [0.3.0] - 2026-05-07

### Added
- **Dedicated `regressors` module**: Added 10 regression wrappers with benchmarking, detailed evaluation, and tuning helpers.
- **Cross-validation support**: Added `cv=` support across classifier and regressor training helpers, including mean/std metrics in reports.
- **Feature engineering toolkit**: Added `breezeml.features` with `select()`, `importance()`, `pca()`, and `polynomial()`.
- **Optional boosting integrations**: Added lazy XGBoost and LightGBM support for both classification and regression workflows.
- **Expanded plotting**: Added comparison charts, learning curves, and feature importance plots to `breezeml.plot`.
- **More datasets**: Added `datasets.california_housing()`, `datasets.penguins()`, and `datasets.from_url()`.
- **Test coverage**: Added tests for regressors, feature engineering, and optional boosting integrations.

### Changed
- **Version bump**: Updated package metadata and public API version to `0.3.0`.
- **README expansion**: Updated the README to document regressors, cross-validation, feature engineering, optional boosting, and new datasets.

### Fixed
- **Parallel fallback**: Added serial fallback paths when process-based parallel execution is blocked in constrained environments.
- **Classifier leaderboard parity**: Included the missing Multinomial Naive Bayes model in classifier leaderboard coverage.

## [0.2.9] - 2026-05-07

### Added
- **Semantic Text Embeddings**: Added `breezeml.text` to convert raw text columns into dense semantic vectors via `sentence-transformers`.
- **Explainability**: Added `breezeml.explain` to generate SHAP feature importance plots.
- **Native Plotting**: Added `breezeml.plot` for Matplotlib confusion matrices and ROC curves in one line.
- **MkDocs Site**: Prepared a `mkdocs-material` documentation site.
- **Manual Task Override**: `fit()` and `auto()` now accept `task="classification"`, `task="regression"`, or `task="auto"`.
- **Ruff Linting**: Integrated `ruff` into the CI pipeline.
- **Extended Test Matrix**: GitHub Actions CI now runs against Python 3.9 through 3.13.

### Changed
- **Deduplicated Preprocessing**: Centralized shared imputation and encoding logic into `breezeml/_preprocessing.py`.
- **Removed Unused Imports**: Cleaned up legacy imports across the codebase.

### Fixed
- **Multinomial Naive Bayes**: Shifted to `MinMaxScaler` when non-negative inputs are required.

## [0.2.7] - 2026-05-07

### Added
- **Test Suite**: Added `tests/` with `pytest` coverage for input validation, core functions, and classifiers.
- **Input Validation**: Added `_validation.py` to validate dataframes and target columns across the public API.

### Changed
- **CI Modernization**: Updated GitHub Actions to run `pytest` with `dev` dependencies.
- **License**: Replaced the stub with the full MIT License text.

### Fixed
- **`from_csv` Data Leak**: Refactored `from_csv` to use the 80/20 evaluation path via `auto()`.

## [0.2.6] - 2026-05-06

### Added
- **Cascade Classification**: Added support for hierarchical multi-level classification pipelines.
- **External Test Sets**: All classifiers accept `X_test` / `y_test`.
- **Macro F1**: Every classifier report now includes `macro_f1`.
- Added aliases `logistic_regression()` and `naive_bayes()`.

## [0.2.5] - 2026-05-06

### Fixed
- **Linear SVM Class Imbalance**: Added `class_weight="balanced"` to `LinearSVC` configurations used throughout the classifier stack.

## [0.2.4] - 2026-04-22

### Fixed
- **Pipeline Save Bug**: `breezeml.save()` now falls back to `joblib.dump()` for raw sklearn pipelines.

## [0.2.3] - 2026-04-22

### Added
- **Sparse Matrix Support**: `linear_svm`, `compare`, and `detailed_report` accept direct `X=` / `y=` sparse inputs.

### Fixed
- **Linear SVM Primal Formulation**: Set `dual=False` to avoid slow dual-form behavior on suitable sparse workloads.

## [0.2.1] - 2026-04-22

### Changed
- **Parallelized Classifier Benchmarks**: `classifiers.compare()` uses `joblib.Parallel(n_jobs=-1)` for faster leaderboards.

## [0.2.0] - 2025-10-21

### Added
- **5 new classifiers**: `knn`, `gradient_boosting`, `adaboost`, `extra_trees`, and `mlp`.
- **`classifiers.compare(df, target)`**: Benchmarks all built-in classifiers and ranks them by accuracy and F1.
- **`classifiers.detailed_report(df, target)`**: Returns confusion matrix, per-class precision/recall/F1, ROC-AUC, and a full sklearn classification report.
- **`classifiers.quick_tune(df, target, algo)`**: Runs `RandomizedSearchCV` with curated parameter grids.
- Added broader example coverage in `examples/test_v020_features.py`.

### Changed
- Rounded metrics to 4 decimal places for consistency.
- Rewrote the README with classifier tables, examples, and architecture notes.

## [0.1.2] - 2025-10-16

### Added
- Interactive Colab quickstart notebook and README badge.
- Beginner-friendly README with CSV guide, feature table, and troubleshooting section.
- **`classifiers` module**: Logistic Regression, SVM (RBF), Linear SVM, Gaussian Naive Bayes, Multinomial Naive Bayes, Decision Tree, Random Forest.
- **`clustering` module**: K-Means, Agglomerative Hierarchical, and DBSCAN.

### Fixed
- RMSE now uses `sqrt(mean_squared_error(...))` for older scikit-learn compatibility.

## [0.1.1] - 2025-10-03

### Added
- Initial PyPI release.
- Core API: `fit`, `predict`, `auto`, `from_csv`, `report`, `save`, `load`.
- `datasets`: `iris`, `wine`, `breast_cancer`, and `diabetes`.
- `creator()` easter egg function.
- `EasyModel` wrapper class encapsulating pipeline, task type, and target column.
- Example scripts: `test_classification.py`, `test_regression.py`, and `test_save_load.py`.
