---
title: 'BreezeML: a production-aware machine learning workflow layer with statistical honesty by default'
tags:
  - Python
  - machine learning
  - scikit-learn
  - AutoML
  - education
  - reproducibility
  - model cards
  - AI agents
authors:
  - name: Akash Anipakalu Giridhar
    orcid: 0009-0007-4162-7159
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 17 July 2026
bibliography: paper.bib
---

# Summary

BreezeML is a Python library that wraps the scikit-learn ecosystem [@pedregosa2011] in a workflow layer designed around one principle: the statistically sound path should be the shortest path. A complete, leakage-safe, stratified, seeded classification workflow is three lines:

```python
from breezeml import datasets, fit, predict

model = fit(datasets.iris(), "species")
preds = predict(model, new_df)
```

Behind that call, BreezeML applies leakage-safe preprocessing inside cross-validation folds, stratified splitting, fixed random seeds, and evaluation against a naive baseline. The returned `Model` object carries the full lifecycle: `evaluate()`, `explain()`, `card()` for an automatically caveated model card [@mitchell2019], conformal `predict_interval()` and `predict_set()` with distribution-free coverage guarantees [@vovk2005; @angelopoulos2023], `export()` for a standalone scikit-learn script, and `deploy()` for a FastAPI service with drift monitoring.

The library spans the applied workflow beyond core training: automated model comparison and tuning (optionally via Optuna [@akiba2019]), automatic feature engineering with out-of-fold target encoding, imbalance handling with threshold tuning and calibration, permutation importance and partial dependence, fairness reports with four-fifths rule verdicts [@feldman2015], statistical significance tests for model comparison (McNemar, paired cross-validation t-test) [@dietterich1998], time series forecasting with walk-forward validation, survival analysis, semi-supervised and active learning, anomaly detection, causal effect estimation, and lightweight experiment tracking.

# Statement of need

Data leakage and missing baselines are a documented, field-wide reproducibility problem: @kapoor2023 catalogue leakage-driven errors in 294 papers across 17 fields, and @kaufman2012 show how easily leakage arises in routine preprocessing. The users most exposed to these pitfalls, students, analysts, and increasingly autonomous AI agents, are the same users least equipped to detect them. Existing low-code libraries reduce boilerplate but not risk: they will happily fit preprocessing on the full dataset, report accuracy without a baseline, and ship an imbalanced classifier without a warning.

BreezeML's contribution is a design that makes methodological rigor the default rather than an option. Its target audiences are educators and students who need sound defaults while learning; analysts who need to go from a DataFrame to an audited, deployable model quickly; and researchers building agent systems who need ML tooling that resists silent statistical misuse.

# State of the field

Low-code ML layers over scikit-learn are well established. PyCaret [@ali2020] offers a broad experiment-management workflow; LazyPredict [@lazypredict] fits many models quickly for triage; auto-sklearn [@feurer2015] automates model and hyperparameter search. BreezeML differs on three axes rather than competing on breadth of estimators:

1. **Statistical guardrails as defaults.** None of the above evaluates every model against a mandatory naive baseline, audits for single-feature target leakage, or refuses naive regression on censored survival data. BreezeML does all three out of the box.
2. **Footprint.** The core installs with exactly four dependencies (scikit-learn, pandas [@mckinney2010], NumPy [@harris2020], joblib), enforced by a CI test that fails if the contract is violated. On identical hardware, a fresh BreezeML install is 274 MB in under three minutes versus 952 MB in over six minutes for PyCaret, with a 3.1 s cold import versus 6.9 s (methodology and caveats in the repository's `benchmarks/` and `docs/benchmarks.md`).
3. **Exit path.** `export()` writes a standalone scikit-learn training script reproducing the exact fitted pipeline with no BreezeML import. Contributing these guardrails upstream to any single library was not viable because they span the whole workflow (training, comparison, deployment, monitoring) and depend on an opinionated, unified API surface; hence a new package.

# Software design

Three design decisions define the library:

**An honest verdict as a first-class object.** `report(model, df)` runs cross-validated performance against a naive baseline, a leakage audit, class imbalance checks, and optional fairness analysis, and reduces them to a single SHIP / WARN / STOP verdict, serializable to JSON or Markdown. A model that fails to beat the naive baseline, or shows target leakage, is flagged before it is deployed.

**AI agents as first-class users.** A built-in Model Context Protocol (MCP) server exposes training, comparison, evaluation, and deployment as agent tools, and exposes the same SHIP / WARN / STOP verdict, so a language-model agent is expected to confirm SHIP before calling `deploy()`. To our knowledge no other ML workflow library ships this guardrail pattern for agent-driven ML.

**Teach, then get out of the way.** `explain_decisions=True` narrates every preprocessing and modeling choice in plain language as it happens, and the zero lock-in export means a student who starts with three lines graduates to reading, and eventually writing, the equivalent scikit-learn code. The main trade-off accepted throughout is implementing capabilities (significance tests, conformal wrappers, drift metrics) on the four core dependencies alone rather than importing specialized packages, trading some sophistication for a dependency contract that keeps the library viable in classrooms, constrained environments, and agent sandboxes.

The package is typed (PEP 561), tested in CI by a suite spanning 29 test modules, and versioned under a documented semantic-versioning and deprecation policy.

# Research impact statement

BreezeML is early-stage but in active use: it is distributed on PyPI, where it records on the order of two thousand downloads per month excluding mirrors (per pypistats.org), and is documented at breezeml.org with an in-browser playground. Its near-term research relevance is twofold. For ML education and applied analysis, it operationalizes published recommendations on leakage avoidance [@kapoor2023] and transparent model reporting [@mitchell2019] as executable defaults rather than guidelines. For research on autonomous AI agents, its MCP server plus machine-readable SHIP / WARN / STOP verdict provides a concrete, inspectable mechanism for constraining agent-driven model training and deployment, a pattern we expect to be of increasing interest as agent systems take on data-science tasks.

# AI usage disclosure

BreezeML was designed and developed by the author, who conceived the library, its architecture, its API, its statistical-honesty design principles, and its benchmark methodology. Generative AI (Claude, Anthropic; used via the Claude Code development environment) was used as a development assistant: helping implement portions of the codebase, documentation, and tests under the author's direction, and assisting with drafting and editing this manuscript. All AI-assisted content was reviewed, modified where needed, and validated by the author, who takes full responsibility for the accuracy, originality, and licensing compliance of all submitted materials.

# Acknowledgements

BreezeML builds on the scientific Python ecosystem, in particular scikit-learn, pandas, NumPy, and joblib.

# References
