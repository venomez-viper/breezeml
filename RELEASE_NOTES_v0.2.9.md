# BreezeML v0.2.9 Release Notes (The "Full Force" NLP & Pro Update)

## 🚀 Key Improvements & Features

### 🧠 Native Semantic Text Embeddings
The new `breezeml.text` module allows you to instantly encode raw text columns into dense 384-dimensional semantic vectors using `sentence-transformers`.

### 📊 SHAP Explainability
The new `explain(model, df)` function leverages the `shap` library to calculate exact feature importance and plot summary charts, showing exactly *why* your model made its decisions.

### 📉 Native Plotting Helpers
Added `breezeml.plot` with `confusion_matrix(model, X_test, y_test)` and `roc_curve(model, X_test, y_test)` for instant Matplotlib evaluations.

### 🌐 Official MkDocs Site
Prepared an ultra-premium MkDocs Material documentation site.

### Cleaned up Unused Imports
Removed unused legacy imports flagged by Ruff to strictly enforce clean code.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
