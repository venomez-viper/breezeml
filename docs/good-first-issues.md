# Good First Issues

Ready-to-post issue drafts for the GitHub tracker. Each is scoped for a
first-time contributor and links the relevant module.

---

**1. Add `datasets.titanic()`**
Load the Titanic dataset (via seaborn or a bundled CSV) following the pattern in `breezeml/breezeml.py::datasets`. Include a test in `tests/test_core.py`. Labels: `good first issue`, `datasets`.

**2. Add `datasets.mnist_mini()`**
A small (~2k row) MNIST subset via `sklearn.datasets.load_digits`, returned as a DataFrame with a `digit` target. Labels: `good first issue`, `datasets`.

**3. `plot.residuals(model, X_test, y_test)`**
Residuals-vs-predicted scatter for regression models, matching the style of existing helpers in `breezeml/plot.py`. Labels: `good first issue`, `plotting`.

**4. `plot.pr_curve(model, X_test, y_test)`**
Precision-recall curve helper (binary classification), mirroring `plot.roc_curve`. Labels: `good first issue`, `plotting`.

**5. Narration for cross-validated training**
`explain_decisions=True` currently narrates holdout splits. Extend `breezeml/_narrate.py` to describe k-fold CV when `cv=` is used. Labels: `good first issue`, `teaching`.

**6. Model card HTML output**
`card(model, "card.html")` should render styled HTML when the path ends in `.html` (markdown otherwise). Pure-stdlib implementation in `breezeml/card.py`. Labels: `good first issue`, `model-cards`.

**7. `clustering.compare()`**
Rank kmeans/agglomerative/dbscan by silhouette score, following `classifiers.compare()`. Labels: `good first issue`, `clustering`.

**8. Spanish/Hindi translation of the quickstart notebook**
Translate `examples/breezeml_quickstart.ipynb`; store as `examples/i18n/`. Labels: `good first issue`, `docs`.

**9. `deploy()` docker-compose option**
Optional `compose=True` flag writing a `docker-compose.yml` next to the Dockerfile in `breezeml/deploy.py`. Labels: `good first issue`, `deploy`.

**10. Windows path handling in `export()`**
`export(data_path=...)` should normalize backslashes so generated scripts run cross-platform (see `tests/test_v040_features.py` workaround). Labels: `good first issue`, `export`.
