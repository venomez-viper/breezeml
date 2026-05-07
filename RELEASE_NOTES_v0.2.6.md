# BreezeML v0.2.6 Release Notes (Cascade Models & External Tests)

## 🚀 Key Improvements & Features

### Cascade Classification
Chain multiple BreezeML models into a hierarchical cascade. This pattern is especially powerful for fine-grained taxonomies where each level narrows the prediction space.

### External Test Sets
Pass `X_test` / `y_test` to any classifier to evaluate on your own held-out split, completely bypassing the internal 80/20 test split.

### Macro F1
Every report dictionary returned by the `classifiers` module now explicitly includes `macro_f1` alongside weighted F1.

### Aliases
Added `logistic_regression()` as an alias for `logistic()`, and `naive_bayes()` as an alias for `multinomial_nb()`.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
