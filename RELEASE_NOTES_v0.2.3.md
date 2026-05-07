# BreezeML v0.2.3 Release Notes (Sparse Matrix Support & Optimization)

## 🚀 Key Improvements & Features

### Sparse Matrix Support
The `classifiers` module functions (`linear_svm`, `compare`, `detailed_report`) now directly accept `X` and `y` keyword parameters.
- **Example:** `classifiers.func(X=X, y=y)`
- **Impact:** This allows users to natively process `scipy.sparse` matrices directly from TF-IDF or Count vectorizers, completely bypassing the massive RAM bottlenecks of dense Pandas conversions.

## 🛠️ Fixes

### Linear SVM Primal Formulation
Hand-patched all `LinearSVC` references with `dual=False`.
- **The Problem:** When processing high-dimensional text vectors where the number of samples is greater than features, the default scikit-learn "Dual Formulation" fallback equations become paralyzed and hang laptops indefinitely (20+ minutes) attempting to converge.
- **The Fix:** Overriding this behavior with the Primal formulation solves the memory deadlock entirely, reducing text-classification training time to < 2 seconds.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
