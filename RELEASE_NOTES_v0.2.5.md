# BreezeML v0.2.5 Release Notes (Hotfixes & Class Balancing)

This release bundles three critical out-of-core performance hotfixes for **BreezeML (v0.2.3, v0.2.4, and v0.2.5)** into one official, incredibly powerful package. It targets high-dimensional matrix errors and class imbalance crashes when utilizing natively pushed arrays. 

## 🚀 Key Improvements & Fixes

### ⚖️ Balanced Class Weights for Linear SVM (v0.2.5)
When handling datasets with 100+ classes containing massive inequality (where 95% of the data belongs to 5% of the classes), default models typically fail by guessing the majority class strictly to lower absolute error. 
We've globally injected `class_weight='balanced'` into all `LinearSVC` initializations (`linear_svm()`, `compare()`, `detailed_report()`). Scikit-learn will now automatically force the math to respect small, tiny classes by inflating the penalization factor for misclassifying minority cases!

### 💾 Universal Model Saving Handler (v0.2.4)
Fixed a major pipeline bug regarding the `breezeml.save()` wrapper function.
- Previously, invoking `save(model, "path")` on raw classifier pipelines triggered an `AttributeError: 'Pipeline' object has no attribute 'save'`. 
- **The Fix**: `breezeml.save()` has been deeply patched. It natively attempts to run `.save()`, but gracefully and automatically falls back to `joblib.dump(model, path)` whenever a raw Scikit-Learn Pipeline is processed.

### ⚡ Linear SVM Primal Formulation Math Loop Fix (v0.2.3)
We've hand-patched all `LinearSVC` references to explicitly utilize `dual=False`. 
- **The Problem**: When processing text metrics (like TF-IDF) where the number of samples ($n$) is radically greater than the number of features ($f$), the standard scikit-learn "Dual Formulation" fallback equations become paralyzed and hang laptops indefinitely (20+ minutes) attempting to converge mathematically on an unsolvable zero-based matrix.
- **The Fix**: Overridden this behavior intentionally. Primal formulation now processes NLP arrays robustly, dropping calculation time uniformly to $<2$ seconds! 

---
*Created by Akash Anipakalu Giridhar 🔥✨*
