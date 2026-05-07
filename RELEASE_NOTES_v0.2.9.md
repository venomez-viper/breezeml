# BreezeML v0.2.9 Release Notes (The "Full Force" NLP & Pro Update)

This is the largest update to BreezeML to date, rolling up all major architectural changes from **v0.2.6 through v0.2.9** into a single monumental release. We have elevated BreezeML from a fast prototyping layer into a full-fledged, production-grade library with advanced NLP capabilities, model explainability, strict input validation, and a beautiful MkDocs website.

## 🚀 Major Features

### 🧠 Native Semantic Text Embeddings
Breaking through the accuracy ceiling of TF-IDF! The new `breezeml.text` module allows you to instantly encode raw text columns into dense 384-dimensional semantic vectors using `sentence-transformers`.
- Models can now *understand* the meaning of sentences out of the box, perfect for hitting 75%+ Macro F1 on complex text classification tasks.

### 📊 SHAP Explainability (`breezeml.explain`)
Machine learning is no longer a black box. The new `explain(model, df)` function leverages the powerful `shap` library to calculate exact feature importance and plot summary charts, showing exactly *why* your model made its decisions.

### 📉 Native Plotting Helpers (`breezeml.plot`)
No more writing 20 lines of Matplotlib code to evaluate your results.
- `confusion_matrix(model, X_test, y_test)` — Beautifully color-coded confusion matrices.
- `roc_curve(model, X_test, y_test)` — Instant ROC curves to measure performance thresholds.

### 🤖 Cascade Classification & Macro F1
- **Hierarchical Models**: Chain multiple BreezeML models into cascades to handle complex multi-level taxonomies.
- **Macro F1**: Every single report dictionary now natively outputs `macro_f1` alongside the standard weighted F1 score.
- **External Test Sets**: Every classifier function now natively accepts `X_test` and `y_test` parameters if you want to bypass the internal 80/20 train/test split.

---

## 🛠️ Reliability & Fixes

### 🛡️ Enterprise-Grade Input Validation
Added `_validation.py` to deeply validate Pandas `DataFrame` inputs and target columns *before* models train, completely eliminating cryptic, deeply-nested scikit-learn tracebacks.

### 🔒 Fixes & Refactors
- **Data Leakage Fix**: Patched a critical data leak in `from_csv()` which was evaluating accuracy directly on the training set instead of a held-out split.
- **MultinomialNB Math Fix**: Multinomial Naïve Bayes now automatically and silently shifts from `StandardScaler` to `MinMaxScaler` in the background, preventing immediate crashes on negative features.
- **Manual Task Overrides**: You can now pass `task="classification"` or `task="regression"` to explicitly override automatic task guessing.
- **Deduplicated Preprocessing**: Centralized all logic into `_preprocessing.py`, enforcing the DRY (Don't Repeat Yourself) principle across the codebase.

---

## 🌐 Documentation & CI
- **Official MkDocs Site**: Integrated a dark-mode `mkdocs-material` site for premium, hosted documentation.
- **CI Modernization**: GitHub Actions now tests the codebase via `pytest` across Python 3.9 through 3.13, and strictly lints every commit via `ruff`.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
