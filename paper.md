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

The library spans the applied workflow beyond core training: automated model comparison and tuning (optionally via Optuna [@akiba2019]), automatic feature engineering with out-of-fold target encoding, imbalance handling with threshold tuning and calibration, permutation importance and partial dependence, fairness reports with four-fifths rule verdicts [@feldman2015], statistical significance tests for model comparison (McNemar, paired cross-validation t-test) [@dietterich1998], time series forecasting with walk-forward validation, survival analysis, semi-supervised and active learning, anomaly detection, causal effect estimation, and lightweight experiment tracking. The core installs with exactly four dependencies (scikit-learn, pandas [@mckinney2010], NumPy [@harris2020], joblib), a contract enforced by a CI test.

BreezeML also treats AI agents as first-class users: a built-in Model Context Protocol (MCP) server exposes training, comparison, evaluation, and deployment as tools, so language-model agents inherit the same statistical guardrails as human users.

# Statement of need

Data leakage and missing baselines are a documented, field-wide reproducibility problem: @kapoor2023 catalogue leakage-driven errors in 294 papers across 17 fields, and @kaufman2012 show how easily leakage arises in routine preprocessing. The users most exposed to these pitfalls, students, analysts, and increasingly autonomous AI agents, are the same users least equipped to detect them. Existing low-code libraries reduce boilerplate but not risk: they will happily fit preprocessing on the full dataset, report accuracy without a baseline, and ship an imbalanced classifier without a warning.

BreezeML's contribution is a design that makes methodological rigor the default rather than an option. Three decisions distinguish it from prior work:

**An honest verdict as a first-class object.** `report(model, df)` runs cross-validated performance against a naive baseline, a leakage audit, class imbalance checks, and optional fairness analysis, and reduces them to a single SHIP / WARN / STOP verdict, serializable to JSON or Markdown. A model that fails to beat the naive baseline, or shows single-feature target leakage, is flagged before it is deployed. The MCP server exposes the same verdict, so an AI agent is expected to confirm SHIP before calling `deploy()`, a guardrail pattern for agent-driven ML that, to our knowledge, no other library provides.

**Zero lock-in as a pedagogical commitment.** `export()` writes a standalone scikit-learn training script that reproduces the exact fitted pipeline with no BreezeML import, and `explain_decisions=True` narrates every preprocessing and modeling choice in plain language as it happens. The library is built to be outgrown: a student who starts with three lines graduates to reading, and eventually writing, the equivalent scikit-learn code.

**A CI-enforced dependency contract.** Comparable frameworks trade convenience for dependency weight. On identical hardware, a fresh BreezeML install is 274 MB and completes in under three minutes, versus 952 MB and over six minutes for PyCaret [@ali2020], with a 3.1 s cold import versus 6.9 s (methodology in the repository's `benchmarks/`). The four-dependency core is enforced by a test that fails CI if the contract is violated, keeping the library viable in classrooms, constrained environments, and agent sandboxes.

BreezeML is aimed at three audiences: educators and students who need sound defaults while learning; analysts who need to go from a DataFrame to an audited, deployable model quickly; and researchers building agent systems who need ML tooling that resists silent statistical misuse. It is distributed on PyPI, where it records on the order of two thousand downloads per month (excluding mirrors, per pypistats.org), documented at breezeml.org with an in-browser playground, typed (PEP 561), tested in CI across a 30-module test suite, and versioned under a documented semantic-versioning and deprecation policy.

# Acknowledgements

BreezeML builds on the scientific Python ecosystem, in particular scikit-learn, pandas, NumPy, and joblib.

# References
