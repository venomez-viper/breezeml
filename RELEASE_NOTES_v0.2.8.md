# BreezeML v0.2.8 Release Notes (Refactoring & Linting)

## 🛠️ Key Improvements & Fixes

### Manual Task Override
`fit()` and `auto()` now accept a `task` parameter (`"classification"`, `"regression"`, or `"auto"`) to manually override automatic task detection instead of relying on the `<20 unique values` heuristic.

### Ruff Linting & CI Expansion
Integrated `ruff` into the CI pipeline for blazing fast linting. GitHub Actions CI now runs against Python 3.9 through 3.13.

### Deduplicated Preprocessing
Centralized identical imputation and encoding logic into `breezeml/_preprocessing.py` to follow DRY principles across the codebase.

### Multinomial Naïve Bayes Math Fix
`multinomial_nb()` now gracefully accepts data with negative features by internally shifting to `MinMaxScaler` instead of crashing on the default `StandardScaler`.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
