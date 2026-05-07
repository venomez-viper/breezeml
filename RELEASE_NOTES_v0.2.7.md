# BreezeML v0.2.7 Release Notes (Reliability & Validation)

## 🛡️ Key Improvements & Fixes

### Input Validation
Added `_validation.py` to check for valid pandas DataFrames and target columns across the public API, replacing cryptic scikit-learn errors with clear, actionable warnings.

### `from_csv` Data Leak Fix
Refactored `from_csv` to properly use the 80/20 test split (via `auto()`) instead of reporting metrics evaluated on the training data.

### Test Suite & CI Modernization
Added a `tests/` directory with a proper `pytest` suite for input validation, core functions, and classifiers. Updated GitHub Actions to natively run `pytest` with `dev` dependencies.

### License Update
Replaced empty stub with the full MIT License text.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
