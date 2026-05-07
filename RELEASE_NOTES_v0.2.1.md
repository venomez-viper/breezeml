# BreezeML v0.2.1 Release Notes (Performance Optimization)

## 🚀 Key Improvements & Features

### Massive Performance Boost for `classifiers.compare`
We now utilize `joblib.Parallel(n_jobs=-1)` to train and evaluate all 12 baseline classification models concurrently across all available CPU cores.

**The Impact:** This effectively turns O(N) waiting time into O(1) time. Instead of waiting for models to train sequentially one-by-one, they all train simultaneously, drastically speeding up the model leaderboards on larger datasets.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
