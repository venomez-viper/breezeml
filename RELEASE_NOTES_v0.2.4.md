# BreezeML v0.2.4 Release Notes (Pipeline Hotfix)

## 🛠️ Key Fixes

### Pipeline Save Bug
Hot-patched the core `breezeml.save()` function to dynamically handle raw Scikit-Learn pipelines.

- **The Problem:** Previously, invoking `save(model, "path")` on raw classifier pipelines triggered an `AttributeError` because the scikit-learn Pipeline object genuinely has no `.save()` method (unlike the custom BreezeML `EasyModel` wrapper).
- **The Fix:** `save()` now dynamically checks for the `.save` attribute. If you pass it a raw strictly scikit-learn `Pipeline` (such as the return from functions in the `classifiers` module), it automatically and silently falls back natively to `joblib.dump()`, preventing fatal crash tracebacks.

---
*Created by Akash Anipakalu Giridhar 🔥✨*
