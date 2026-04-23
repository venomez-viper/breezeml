# BreezeML v0.2.2 Release Notes

We are excited to announce **BreezeML v0.2.2**, which introduces significant memory optimizations and native support for high-dimensional sparse array data!

## 🚀 What's New

### Native Sparse Matrix & Array Inputs ($X, y$)
The `breezeml.classifiers` module has been heavily refactored beneath the hood to natively ingest compressed features (such as `scipy.sparse` TF-IDF vectors) or standalone NumPy arrays without requiring implicit DataFrame mapping. 

- All classifier algorithms (like `linear_svm`, `random_forest`, `gradient_boosting`) can now accept `X` and `y` direct keyword arguments.
- High-level interface utilities—including `classifiers.compare`, `classifiers.detailed_report`, and `classifiers.quick_tune`—intelligently bypass standard Pandas imputation and scaling transformations when `$X, y$` are given, keeping your sparse zeroes strictly uncompressed.
- **Why this matters**: Passing high-dimensional vectorized data in DataFrames forces you to run `toarray()`, which unpacks vectors into heavy dense grids. This solves out-of-memory kernel crashes occurring on arrays exceeding 50,000+ features. 

*Example Code Changes:*
```python
# Before (Caused 850+ MB memory ballooning and freezing)
import pandas as pd
breeze_df = pd.DataFrame(X_vec.toarray())  # Very Expensive!
breeze_df['target'] = y
model, report = linear_svm(breeze_df, 'target')

# Now in v0.2.2 (Instantaneous & Lightweight)
model, report = linear_svm(X=X_vec, y=y)
```

## 🛠️ Also Including the v0.2.1 Core Overhaul
*(Previously unreleased individually to GitHub)*
- **Massive Performance Boost**: `classifiers.compare` now runs parallelized out-of-core operations via `joblib.Parallel(n_jobs=-1)`. All 12 supported algorithms fit and test concurrently, exponentially speeding up model experimentation leaderboards!

---
*Created by Akash Anipakalu Giridhar 🔥✨*
