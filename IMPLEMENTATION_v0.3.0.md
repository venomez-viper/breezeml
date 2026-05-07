# BreezeML v0.3.0 Implementation Plan

## Goal

Transform BreezeML from a classification-focused prototyping tool into a comprehensive, production-grade ML library by filling the biggest gaps: regression parity, cross-validation, feature engineering, and gradient boosting integration.

---

## Phase 1: Regressors Module (Highest Priority)

The `classifiers` module has 12 algorithms, `compare()`, `detailed_report()`, and `quick_tune()`. Regression currently only has Random Forest and Linear Regression buried inside `breezeml.py`. This phase creates full parity.

### Step 1.1: Create `breezeml/regressors.py`

Mirror the architecture of `classifiers.py` exactly. The internal `_train()` helper should:
- Accept both `(df, target)` and `(X, y)` input styles (same as classifiers)
- Accept optional `X_test` / `y_test` for external evaluation
- Use `_build_preprocessor()` from `_preprocessing.py` for consistency
- Return `(pipeline, report)` where report contains `r2`, `mae`, `rmse`, `adjusted_r2`, `mape`

Algorithms to include (10 total):

| Function Name | sklearn Class | Notes |
|---|---|---|
| `regressors.linear` | `LinearRegression` | Baseline |
| `regressors.ridge` | `Ridge` | L2 regularization |
| `regressors.lasso` | `Lasso` | L1 regularization / feature selection |
| `regressors.elastic_net` | `ElasticNet` | L1 + L2 hybrid |
| `regressors.svr` | `SVR` | Support vector regression |
| `regressors.decision_tree` | `DecisionTreeRegressor` | Interpretable |
| `regressors.random_forest` | `RandomForestRegressor` | Strong baseline |
| `regressors.gradient_boosting` | `GradientBoostingRegressor` | High accuracy |
| `regressors.knn` | `KNeighborsRegressor` | Non-parametric |
| `regressors.mlp` | `MLPRegressor` | Neural net baseline |

### Step 1.2: Add `regressors.compare(df, target)`

Same concept as `classifiers.compare()`:
- Run all 10 regressors on the dataset using `joblib.Parallel(n_jobs=-1)`
- Return a leaderboard sorted by R2 (descending)
- Print a formatted table to stdout

Output format:
```
Rank  Regressor                R2        MAE       RMSE
------------------------------------------------------------
1     Gradient Boosting        0.8912    2.3401    3.1205
2     Random Forest            0.8734    2.5012    3.4501
...
```

### Step 1.3: Add `regressors.detailed_report(df, target)`

Returns a dict containing:
- `r2`, `adjusted_r2`, `mae`, `rmse`, `mape`, `explained_variance`
- `residuals` array (y_true - y_pred)
- `prediction_vs_actual` array of tuples for plotting

### Step 1.4: Add `regressors.quick_tune(df, target, algo)`

Same pattern as `classifiers.quick_tune()`:
- Define `_PARAM_GRIDS` dict with curated hyperparameter ranges for each algorithm
- Use `RandomizedSearchCV` with `scoring="r2"`
- Return `(best_model, best_params, report)`

### Step 1.5: Wire into `__init__.py` and `breezeml.py`

- Add `from . import regressors` to `__init__.py`
- Update `auto()` in `breezeml.py` to use `regressors.random_forest()` instead of the old inline `regress()` function (keep `regress()` for backward compatibility)

### Step 1.6: Add tests

Create `tests/test_regressors.py`:
- Test that each regressor returns a valid `(pipeline, report)` tuple
- Test that `compare()` returns a list of dicts
- Test that `quick_tune()` returns `(model, params, report)`
- Use `datasets.diabetes()` as the test dataset

---

## Phase 2: Cross-Validation Support

Every classifier and regressor currently uses a single 80/20 split. This phase adds optional k-fold cross-validation.

### Step 2.1: Add `cv` parameter to `classifiers._train()`

Modify the internal `_train()` function in `classifiers.py`:

```python
def _train(model, df=None, target=None, X=None, y=None, 
           X_test=None, y_test=None, force_minmax=False, cv=None):
```

When `cv` is not None (e.g., `cv=5`):
- Use `cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")` instead of a single split
- Return mean and std of scores in the report:
  ```python
  report = {
      "accuracy": round(float(scores.mean()), 4),
      "accuracy_std": round(float(scores.std()), 4),
      ...
  }
  ```
- Still do a final fit on all training data so the returned pipeline is usable for prediction

### Step 2.2: Propagate `cv` to all public classifier functions

Every function (`logistic`, `svm`, `random_forest`, etc.) gets an optional `cv=None` parameter that passes through to `_train()`.

Example usage:
```python
model, report = classifiers.logistic(df, "species", cv=5)
# report = {'accuracy': 0.9667, 'accuracy_std': 0.0211, 'f1': 0.9654, 'f1_std': 0.0198}
```

### Step 2.3: Add `cv` to regressors module

Same pattern as Step 2.1-2.2 but for regressors, using `scoring="r2"`.

### Step 2.4: Add `cv` to `compare()` functions

When `cv` is passed to `classifiers.compare()` or `regressors.compare()`, the leaderboard should show `mean +/- std` for each metric.

### Step 2.5: Add tests

- Test that passing `cv=3` to a classifier returns a report with `accuracy_std` key
- Test that `cv=None` (default) still works exactly as before (backward compatible)

---

## Phase 3: Feature Engineering Module

### Step 3.1: Create `breezeml/features.py`

Function: `features.select(df, target, method="mutual_info", k=10)`
- Accepts `"mutual_info"`, `"chi2"`, or `"rfe"` as method
- For `"mutual_info"`: use `sklearn.feature_selection.mutual_info_classif` or `mutual_info_regression` (auto-detect from target)
- For `"chi2"`: use `sklearn.feature_selection.chi2` (classification only)
- For `"rfe"`: use `sklearn.feature_selection.RFE` with a `RandomForestClassifier` base estimator
- Returns a new DataFrame with only the top `k` features + the target column
- Also prints a ranked list of features and their scores

Function: `features.importance(model, df, target=None)`
- For tree-based models: extract `.feature_importances_` directly
- For other models: use `sklearn.inspection.permutation_importance`
- Returns a sorted dict of `{feature_name: importance_score}`

Function: `features.pca(df, n_components=0.95)`
- Apply `sklearn.decomposition.PCA` to numeric columns
- If `n_components` is a float (0-1), treat as variance threshold
- If `n_components` is an int, use that many components
- Returns a new DataFrame with PCA columns replacing the original numeric columns

Function: `features.polynomial(df, degree=2, columns=None)`
- Use `sklearn.preprocessing.PolynomialFeatures` on specified columns (or all numeric)
- Returns a new DataFrame with the interaction/polynomial columns appended

### Step 3.2: Wire into `__init__.py`

Add `from . import features` and include in `__all__`.

### Step 3.3: Add tests

Create `tests/test_features.py`:
- Test `select()` returns a DataFrame with `k+1` columns (k features + target)
- Test `importance()` returns a dict with float values
- Test `pca()` reduces dimensionality
- Test `polynomial()` increases column count

---

## Phase 4: XGBoost and LightGBM Integration

### Step 4.1: Add optional dependencies to `pyproject.toml`

```toml
[project.optional-dependencies]
boost = ["xgboost", "lightgbm"]
all = ["sentence-transformers", "shap", "matplotlib", "xgboost", "lightgbm"]
```

### Step 4.2: Add to `classifiers.py`

Add two new classifier functions with lazy imports:

```python
def xgboost(df=None, target=None, ...):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Install xgboost: pip install breezeml[boost]")
    return _train(XGBClassifier(...), ...)

def lightgbm(df=None, target=None, ...):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("Install lightgbm: pip install breezeml[boost]")
    return _train(LGBMClassifier(...), ...)
```

### Step 4.3: Add to `regressors.py`

Same pattern with `XGBRegressor` and `LGBMRegressor`.

### Step 4.4: Add to `compare()` leaderboards

In both `classifiers.compare()` and `regressors.compare()`:
- Try to import XGBoost/LightGBM
- If available, include them in the benchmark
- If not installed, silently skip them (no error)

### Step 4.5: Add tuning grids to `quick_tune()`

Add `_PARAM_GRIDS` entries for both:
```python
"xgboost": {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
},
"lightgbm": {
    "model__n_estimators": [100, 200, 500],
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.01, 0.05, 0.1],
},
```

### Step 4.6: Add tests

Create `tests/test_boost.py`:
- Use `pytest.importorskip("xgboost")` and `pytest.importorskip("lightgbm")` so tests are skipped if the packages are not installed
- Test basic training and prediction for both classifiers and regressors

---

## Phase 5: Quick Wins

### Step 5.1: More datasets

Add to `datasets` class in `breezeml.py`:
- `datasets.penguins()` -- use `seaborn` dataset or bundle a small CSV
- `datasets.california_housing()` -- use `sklearn.datasets.fetch_california_housing`
- `datasets.from_url(url, target)` -- `pd.read_csv(url)` then return the DataFrame

### Step 5.2: Comparison visualization

Add to `plot.py`:
```python
def compare_chart(results, metric="accuracy"):
    """Bar chart from classifiers.compare() or regressors.compare() output."""
```

### Step 5.3: Learning curve plot

Add to `plot.py`:
```python
def learning_curve(model, df, target, cv=5):
    """Plot training vs validation score as dataset size grows."""
```
Uses `sklearn.model_selection.learning_curve` under the hood.

### Step 5.4: Feature importance plot

Add to `plot.py`:
```python
def feature_importance(model, df, target=None, top_n=15):
    """Bar chart of top N feature importances."""
```

---

## Phase 6: Finalization

### Step 6.1: Version bump
- Update `__init__.py` version to `"0.3.0"`
- Update `pyproject.toml` version to `"0.3.0"`

### Step 6.2: Update README.md
- Add regressors section with code examples
- Add cross-validation examples
- Add feature engineering examples
- Add XGBoost/LightGBM to the feature table
- Update the architecture diagram
- Mark completed items on the roadmap

### Step 6.3: Update CHANGELOG.md
- Add `[0.3.0]` entry with all new features

### Step 6.4: Write RELEASE_NOTES_v0.3.0.md

### Step 6.5: Run full test suite
```bash
ruff check .
pytest tests/ -v
```

### Step 6.6: Tag and publish
```bash
git add . && git commit -m "Release v0.3.0"
git tag v0.3.0
git push && git push origin v0.3.0
```

---

## Execution Order Summary

| Order | Phase | Est. Effort | Key Deliverable |
|-------|-------|-------------|-----------------|
| 1 | Phase 1: Regressors | Medium | `regressors.py` with 10 algorithms + compare/tune |
| 2 | Phase 2: Cross-Validation | Low | `cv=5` parameter on all classifiers and regressors |
| 3 | Phase 3: Feature Engineering | Medium | `features.py` with select, importance, PCA |
| 4 | Phase 4: XGBoost/LightGBM | Low | 2 new classifiers + 2 new regressors |
| 5 | Phase 5: Quick Wins | Low | More datasets, comparison charts, learning curves |
| 6 | Phase 6: Finalization | Low | README, CHANGELOG, tests, publish to PyPI |

---

## Key Architecture Rules

1. All new modules must use `from __future__ import annotations` for Python 3.9 compatibility
2. All heavy dependencies (xgboost, lightgbm) must use lazy imports with helpful error messages
3. All public functions must have docstrings
4. All reports must round to 4 decimal places
5. Mirror the `classifiers.py` patterns exactly for consistency
6. Use `_build_preprocessor()` from `_preprocessing.py` -- do NOT duplicate preprocessing logic
7. Use `check_df_target()` from `_validation.py` for input validation
8. All parallel execution uses `joblib.Parallel(n_jobs=-1)`
