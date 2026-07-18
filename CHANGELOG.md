# Changelog

All notable changes to BreezeML are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Empirical validation study** (`benchmarks/validation_study.py`, `docs/validation.md`): measures the guardrail claims on 10 datasets - `audit()` leakage detection rates (with false-positive checks), conformal empirical coverage vs the nominal 90% guarantee, and SHIP/WARN/STOP verdict correctness on real vs label-shuffled models. Results in `benchmarks/validation_results.json`.
- **Case studies page** (`docs/case-studies.md`): BreezeML applied in a real CMS Medicare Part B provider-fraud study.
- `CITATION.cff` so GitHub's "Cite this repository" button works.

### Fixed
- **`audit()` leakage probe blind spots** (found by the validation study): a direct copy of a continuous regression target and a copy of a many-class (10+) target were not flagged, because a depth-bounded probe tree cannot express them. The probe now uses an unbounded tree with a leaf-size floor plus a Spearman rank-correlation fast path for regression targets; direct-copy detection went from 5/10 to 10/10 datasets with zero false positives on clean data. Covered by new regression tests.

---

## [2.0.0] - 2026-07-10

Honest by default, agent-native by design. 2.0 consolidates the scattered honesty toolkit into one verdict, unifies the model object, ships types, and puts the flagship report in the hands of AI agents.

### Added
- **`breezeml.report()` - the honest scorecard**: one call runs the whole honesty gauntlet on a trained model - cross-validated performance vs a naive baseline, a data audit for leakage and quality, class-imbalance severity, and an optional fairness check - and returns a single **SHIP / WARN / STOP** verdict as a `Report` with `to_dict()`, `to_json()`, `to_markdown()`, and a pretty console view. A single critical finding (leakage, or a model no better than guessing) is enough to say STOP.
- **Unified `Model` object**: `fit(df, target)` returns one coherent object - `predict`, `evaluate`, `report`, `explain`, `card`, `export`, `deploy`, plus conformal uncertainty: `predict_interval(X, calib_df)` gives regression intervals and `predict_set(X, calib_df)` gives classification label sets that cover the truth at `>= 1 - alpha`. The public name is `Model` (`EasyModel` stays as an alias, so existing pickles and imports keep working).
- **Agent-native**: the MCP server exposes a `report` tool that returns the SHIP/WARN/STOP verdict as structured JSON; its docstring instructs agents to confirm a SHIP verdict before calling `deploy` or `export`.
- **Typed and stable**: ships a `py.typed` marker (PEP 561) so type checkers read BreezeML's annotations directly; public entry points are annotated. New `docs/stability.md` documents the public surface, semantic-versioning promise, and deprecation policy.

### Changed
- **BREAKING: `report(model, df)` now returns a `Report`** (the honest scorecard), not a bare metrics dict. Use `model.evaluate(df)` for a plain metrics dict.

### Migration (1.x -> 2.0)
- `metrics = report(model, df)`  ->  `metrics = model.evaluate(df)`.
- Use `report(model, df)` (or `model.report(df)`) for the full honesty verdict.
- No other public call changed; the four-dependency contract and zero-lock-in `export()` are unchanged.

## [1.9.0] - 2026-07-09

Honest Uncertainty and Cause: conformal gives honest uncertainty on every prediction, active learning spends labels wisely, automatic feature engineering enriches without leaking, and causal inference separates correlation from causation. The honesty thread holds: every module reports the naive or baseline answer next to the smart one.

### Added
- **Conformal prediction** (`breezeml.conformal`): `conformal_regressor(model, calib_df, target, alpha=0.1)` wraps an already-trained regressor and returns a `ConformalRegressor` whose `predict_interval(X)` gives a DataFrame with `lower`/`point`/`upper` columns guaranteed to cover the truth at `>= 1 - alpha` (with optional `normalize=True` for locally adaptive bands); `conformal_classifier(model, calib_df, target, alpha=0.1)` returns a `ConformalClassifier` whose `predict_set(X)` returns a list of label sets via the LAC score method. Both expose `coverage_report(df, target)` returning `target_coverage`, `empirical_coverage`, mean interval width / set size, and a `well_calibrated` flag. Distribution-free, using the finite-sample conformal quantile `ceil((n+1)(1-alpha))/n` on a held-out calibration set.
- **Active learning** (`breezeml.active`): `query(model, unlabeled_df, k, strategy)` ranks an unlabeled pool by informativeness (strategies `uncertainty`, `margin`, `entropy`, `random`) and returns `{"indices", "scores", "strategy", "k"}` for the top-k rows to label next; `simulate(df, target, initial, budget, step, strategy)` runs the active loop against a random baseline from the same starting set and returns `budgets`, `active_accuracy`, `random_accuracy`, `area_between_curves`, and `active_wins`, printing an honest verdict on whether active learning beat random.
- **Automatic feature engineering** (`breezeml.autofeat`): `engineer(df, target)` returns `(new_df, report)` after datetime expansion (year/month/day/dayofweek/is_weekend, plus hour when present), leakage-safe out-of-fold (5-fold) target-mean encoding and frequency encoding for high-cardinality categoricals, capped pairwise numeric interactions, and pruning of constant and near-duplicate (`|r| > 0.98`) columns. The `report` dict lists `added`, `dropped`, `encoded`, `datetime_expanded`, `n_features_before`, `n_features_after`, and `notes`. The input frame is never mutated.
- **Causal inference and uplift** (`breezeml.causal`): `estimate_ate(df, treatment, outcome, method="naive"|"t_learner"|"ipw")` returns `ate`, `naive_ate`, `method`, `propensity_range`, `n_treated`, `n_control`, and a `note`, always reporting the confounded naive baseline next to the adjusted estimate; `uplift(df, treatment, outcome)` returns a `(TLearnerPair, report)` pair whose `predict_uplift(X)` gives the per-row CATE; `check_confounding(df, treatment, outcome)` returns per-covariate standardized mean differences with `imbalanced`, `max_abs_smd`, `looks_randomized`, and warns that observational estimates rest on the untestable no-unmeasured-confounders assumption.
- **Docs**: new guides for all four modules (`docs/guides/conformal.md`, `docs/guides/active-learning.md`, `docs/guides/autofeat.md`, `docs/guides/causal.md`).

## [1.8.0] - 2026-07-08

Four Questions: is this difference real, can it predict many things at once, what should this user see next, and when will the event happen? Four new modules, each with the honesty thread intact: every one warns when the naive approach is wrong.

### Added
- **Statistical significance** (`breezeml.significance`): `mcnemar(model_a, model_b, df, target)` splits one shared holdout, runs McNemar's test on the discordant pairs, and returns `statistic`, `p_value`, `n_discordant`, `significant`, and a plain-English `verdict`; `paired_cv_ttest(algo_a, algo_b, df, target)` scores two algorithms on the same CV folds and returns the paired `t_statistic`, `p_value`, `mean_difference`, and verdict. Both say "keep the simpler model" when the gap is not significant. The error function, chi-square tail, Student t p-value, and incomplete beta are all pure numpy - no scipy.
- **Multi-label / multi-output** (`breezeml.multi`): `multi_label(df, targets, chain=False)` predicts several label columns at once (independent models, or a `ClassifierChain` when `chain=True`) and returns an `(EasyModel, report)` with per-target accuracy/F1 plus `subset_accuracy` (exact-match ratio) and `hamming_loss`; `multi_output(df, targets)` does the regression analogue and returns per-target r2/mae/rmse plus `average_r2`.
- **Recommender systems** (`breezeml.recommend`): the `Recommender` class and `collaborative_filter(df, user_col, item_col, rating_col=None)` build a user-item matrix and factorize it with a truncated SVD. `recommend(user, k)` returns `[(item, score), ...]` over unseen items; `recommend_report(user, k)` adds `cold_start` and `method` (`"svd"` / `"popularity"`) flags. Cold-start users fall back to global popularity honestly, and sparse matrices and single-interaction users/items get warnings.
- **Survival analysis** (`breezeml.survival`): `kaplan_meier(df, duration_col, event_col)` returns the survival curve with Greenwood standard errors, 95% pointwise CIs, `median_survival`, and `censoring_rate`; `groups_kaplan_meier(..., group_col)` fits a curve per group and adds a log-rank test (`logrank_statistic`, `logrank_p_value`, `significant`); `check_censoring()` reports the censoring rate and warns loudly against regressing on censored durations. Pure numpy, including the chi-square tail behind the log-rank test.
- **Docs**: new guides for all four modules (`docs/guides/significance.md`, `docs/guides/multi-output.md`, `docs/guides/recommenders.md`, `docs/guides/survival.md`).

## [1.7.1] - 2026-07-07

### Fixed
- **Docs audit**: architecture tree now lists all v1.7 modules; model counts corrected to 22/22/9 everywhere (README, architecture doc, and the in-library `guide()`); `[automl]` extra added to the install list; examples table trimmed to files that exist.

## [1.7.0] - 2026-07-07

The Honest Machine: tools that tell you when your data is lying, when your model treats groups unequally, and when extra complexity did not earn its keep.

### Added
- **Data audit** (`breezeml.audit`): `audit(df, target)` checks for ID-like columns, constant columns, duplicate rows, contradictory labels, high-cardinality categoricals, heavy missingness, class imbalance, and target leakage - probed by training a tiny depth-3 tree on each feature alone; a near-perfect single-feature score flags a leak. Critical findings flip the `ok` flag. `audit.contamination(train_df, test_df)` detects rows shared across a train/test split.
- **Fairness reports** (`breezeml.fairness`): `report(model, df, sensitive=...)` gives per-group size, accuracy, F1, selection rate, TPR, and FPR, plus the demographic parity ratio, a four-fifths (80%) rule verdict, and TPR/FPR gaps; per-group MAE and mean error for regression. Small groups are called out as noise, not evidence.
- **Imbalance toolkit** (`breezeml.imbalance`): `summary()` (severity verdict), `tune_threshold()` (best minority-F1 threshold on a held-out split), `calibrate()` (isotonic/sigmoid with before/after Brier scores), `cost_report()` (the threshold that minimizes your FP/FN costs), and `predict_with_threshold()`. Plus `balanced=True` on `classify()` / `auto()` for class-weighted training. Core dependencies only; no SMOTE clones.
- **Model blending** (`breezeml.blend`): `blend(df, target, method="vote"|"stack")` ensembles the top `compare()` models by soft voting or stacking, and always reports `beats_best_single`; when the blend loses, it says to keep the single model.
- **Experiment tracking** (`breezeml.track`): `log()`, `leaderboard()`, `best()`, and `clear()` on a plain, git-committable `.breezeml/runs.json`. Zero extra dependencies, no server, no account.
- **Anomaly detection** (`breezeml.anomaly`): `isolation_forest`, `local_outlier_factor`, `one_class_svm`, and `elliptic_envelope`, plus `compare()` reporting majority and unanimous consensus across detectors - because unsupervised detection has no accuracy score, agreement is the evidence.
- **Semi-supervised learning** (`breezeml.semisupervised`): `self_train(df, target)` treats NaN targets as the unlabeled pool, pseudo-labels confident rows, and always reports the supervised-only baseline plus a `helped` verdict, so you know whether the unlabeled data earned anything.
- **Native explainability** (`breezeml.explain`): `permutation_importance()` and `partial_dependence()` run on the core dependencies; no SHAP install needed.
- **Command line interface** (`breezeml.cli`): `breezeml train / compare / automl / audit / deploy / card / zen / guide` from the terminal. `breezeml audit` exits 1 on critical findings, making it a natural CI gate.
- **Model zoo wave 2**: 4 new classifiers (`bernoulli_nb`, `passive_aggressive`, `nearest_centroid`, `bagging` - 22 total), 6 new regressors (`poisson`, `quantile`, `theilsen`, `ransac`, `kernel_ridge`, `bagging` - 22 total), and 3 new clusterers (`meanshift`, `optics`, `hdbscan` - 9 total).
- **Docs**: new guides for the honest-ML toolkit (`docs/guides/honest-ml.md`) and the v1.7 toolkits (`docs/guides/toolkits.md`).

### Community
- **First external PR merged**: `plot.pr_curve()` (precision-recall curve plotting) contributed by @its-Sohan.

### Changed
- **Dev extra**: `matplotlib` added to the `[dev]` extra.

## [1.6.0] - 2026-07-06

### Added
- **`breezeml.guide()`**: an in-library map of the garden path, so nobody ever feels lost in a big library.
- **The Four Breaths**: BreezeML's layered architecture is now explicit and documented (`docs/architecture.md`): Breath 1 one-liners, Breath 2 understand-and-choose, Breath 3 automate-and-ship, Breath 4 extensions. Each layer is complete on its own; nothing above it is required.

### Changed
- **README restructured for beginners**: a "Start Here: The Garden Path" section walks new users through the four breaths; the architecture section shows the layered design. The reference material stays, but day one needs only the first screen.

## [1.5.1] - 2026-07-06

### Added
- **`breezeml.sensei()`**: seek the founder of the dojo, receive one teaching. Reproducible with a seed, as sensei prefers.

## [1.5.0] - 2026-07-06

### Added
- **The zen garden**: `breezeml.zen()` prints the Zen of BreezeML - kaze no michi, the way of the wind - fifteen machine learning haiku under falling sakura. `breezeml.haiku()` carries in a single haiku; `breezeml.fortune()` draws an omikuji (shrine fortune slip) for your ML day, from Dai-kichi to Kyou. Inspired by the guiding wind: BreezeML does not push you to the destination, it shows the way.
- **Honesty easter eggs**: a perfect 1.0 accuracy atop `classifiers.compare()` earns a gentle warning about easy problems and leaky targets; when the naive baseline wins `timeseries.compare()`, the leaderboard admits your series may be a random walk.

## [1.4.0] - 2026-07-06

### Added
- **Live training progress** (`breezeml._progress`): `classifiers.compare()`, `regressors.compare()`, `automl()` screening, and `timeseries.compare()` now show a self-updating progress bar in terminals and clean milestone lines in CI/piped logs. Pure stdlib (the 4-dependency contract rules out tqdm). Control with `progress=True/False`; defaults to the `show` setting.
- **6 new classifiers**: `hist_gradient_boosting` (sklearn's LightGBM-class booster), `ridge`, `sgd`, `lda`, `qda`, `complement_nb` - 18 total, all in `compare()` and (where tunable) `quick_tune()`.
- **6 new regressors**: `hist_gradient_boosting`, `extra_trees`, `adaboost`, `huber` (outlier-robust), `bayesian_ridge`, `sgd` - 16 total.
- **3 new clustering algorithms**: `gaussian_mixture` (soft memberships + BIC), `birch`, `spectral`.
- **New tuning grids** for hist gradient boosting, ridge, SGD, extra trees, and AdaBoost.

## [1.3.0] - 2026-07-06

### Added
- **Drift monitoring** (`breezeml.drift`): `check(model, new_df)` compares new data against reference distributions captured at training time - PSI over decile bins for numeric columns, unseen-category detection for categoricals, training-range violations, missing-rate spikes, and vanished columns. Also available as `model.check_drift()`.
- **Live `/drift` endpoint**: `deploy()` now exports `reference.json` and the generated FastAPI app buffers recent predictions to serve a real-time drift report - still with zero breezeml dependency at runtime.
- **Reference distributions in meta**: core-API models now store compact training distributions (`meta["reference"]`).

## [1.2.0] - 2026-07-06

### Added
- **Time series module** (`breezeml.timeseries`): `make_features()` (leakage-free lag/rolling/calendar features), `compare()` (walk-forward leaderboard with a mandatory naive last-value baseline), and `forecast()` (recursive multi-step forecasting with `beats_naive` / `skill_vs_naive` honesty metrics). Runs on the 4 core dependencies; XGBoost/LightGBM join automatically when installed.

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
