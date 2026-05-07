# BreezeML v0.3.0 Release Notes (The "Regression Parity" Update)

## Key Improvements & Features

### Full `regressors` Module
Added a dedicated `breezeml.regressors` module to bring regression up to feature parity with the classification stack. BreezeML now includes 10 one-line regressors:

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

### Regression Leaderboards
Added `regressors.compare(df, target)` to benchmark all built-in regressors in one call and rank them by R2, MAE, and RMSE.

### Detailed Regression Reports
Added `regressors.detailed_report(df, target)` with:

- `r2`
- `adjusted_r2`
- `mae`
- `rmse`
- `mape`
- `explained_variance`
- residual arrays
- prediction-vs-actual pairs for plotting

### Regression Hyperparameter Tuning
Added `regressors.quick_tune(df, target, algo)` powered by `RandomizedSearchCV` for one-line model search across the new regression algorithms.

### Core API Regression Upgrade
The core `regress()`, `auto()`, and `report()` paths now use the new regression metrics, so BreezeML's top-level API reports richer regression evaluation out of the box.

### Better Resilience in Constrained Environments
Added a safe serial fallback when process-based parallel execution is blocked by the local environment, so leaderboards and tuning still work in restricted Windows setups.

---
*Created by Akash Anipakalu Giridhar*
