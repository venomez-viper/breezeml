# Case studies

Real research projects built on BreezeML. These are working codebases, not demos;
links go to the public repositories.

## Healthcare provider fraud ranking (CMS Part B + LEIE)

**Repository:** [Healthcare-Fraud-Analytics](https://github.com/venomez-viper/Healthcare-Fraud-Analytics)

A provider-fraud risk study over the CMS Medicare Part B provider panel
(2019-2023) joined with the OIG LEIE exclusion list. The goal is
investigator-budget-aware ranking: make the top few percent of flagged
providers as dense with real fraud as possible, and explain every flag.

**How BreezeML is used.** Model training runs through BreezeML's
`classifiers` module: a scaled logistic-regression baseline and the
gradient-boosting model are built, fit, and reported via BreezeML pipelines
on a provider-grouped split. The project adds what is out of BreezeML's
scope at the domain layer: random-undersampling for the extreme class
imbalance and top-k precision/recall evaluation.

**Results.** The BreezeML-fit gradient-boosting model reaches ROC-AUC
0.809, matching the published Part B benchmark range (Herland,
Khoshgoftaar and Bauder 2018: 0.805-0.816), and reviewing the top 10% of
ranked providers catches 55.6% of known-excluded providers.

**Honesty methodology in practice.** When temporal trajectory features
produced a large lift (recall@1% 0.17 to 0.31), the project applied the
same single-feature leakage probe that `audit()` implements: each suspect
feature was scored against the label alone. `traj_years` separated classes
at AUC 0.72 by itself, a panel-position artifact rather than provider
behaviour, and the position-like features were dropped before results were
reported. This is the leakage-detection discipline BreezeML is designed
around, applied to a real, messy administrative dataset.

---

*Built something with BreezeML? Share it in
[Show & Tell](https://github.com/venomez-viper/breezeml/discussions) and we
may feature it here.*
