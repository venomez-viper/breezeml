# Empirical validation of the guardrails

BreezeML's honesty features make testable claims. This page reports the
results of testing them. Everything here is reproducible with one command:

```bash
python benchmarks/validation_study.py
```

Raw numbers land in `benchmarks/validation_results.json`. Datasets: iris,
wine, breast_cancer, digits (800-row sample), diabetes, friedman1, and
synthetic classification/regression sets from scikit-learn generators.
Measured with BreezeML 2.0.0, 5 seeds where applicable.

## Study A: does `audit()` catch injected leakage?

For each dataset we ran `audit()` four ways: on the clean data (any leakage
flag here is a false positive), with a direct copy of the target injected as
a feature, with a 10%-corrupted copy, and with a row-identifier column.

| Injection | Detected | False positives on clean data |
|---|---|---|
| Direct target copy | **10 / 10** | 0 / 10 |
| Row-ID column | **10 / 10** | 0 / 10 |
| Noisy copy (10% corrupted) | 0 / 10 | 0 / 10 |

**Known limitation, kept deliberately.** A copy with 10% corrupted rows
scores about 0.90 as a single-feature predictor, below the 0.98 flag
threshold. Lowering the threshold to catch it would start flagging
legitimately strong features, and a false "leakage" alarm teaches users to
ignore the audit. `audit()` is a screen for the classic catastrophic leaks
(post-outcome columns, encoded labels, IDs), not a proof of absence.

**This study improved the library.** The first run of Study A exposed two
blind spots in the original probe (a bounded-depth tree could not express a
leaked copy of a many-valued or continuous target): direct copies were
missed on all regression datasets and on 10-class digits. The probe now
uses an unbounded tree with a leaf-size floor plus a rank-correlation fast
path, and both cases are covered by regression tests
(`tests/test_v17_honest_ml.py`).

## Study B: does conformal prediction hit its guarantee?

`conformal_regressor()` and `conformal_classifier()` promise marginal
coverage of at least 1 - alpha. At alpha = 0.10 (nominal 90%), empirical
coverage on held-out test sets, mean over 5 shuffles:

| Dataset | Task | Mean coverage | Std |
|---|---|---|---|
| diabetes | regression | 0.906 | 0.022 |
| synth_reg_0 | regression | 0.918 | 0.038 |
| synth_reg_1 | regression | 0.907 | 0.015 |
| friedman1 | regression | 0.915 | 0.015 |
| iris | classification | 0.880 | 0.045 |
| wine | classification | 0.944 | 0.025 |
| breast_cancer | classification | 0.895 | 0.019 |
| digits_800 | classification | 0.919 | 0.040 |
| synth_clf_0 | classification | 0.918 | 0.020 |
| synth_clf_1 | classification | 0.901 | 0.027 |

All ten datasets land within noise of the nominal level (the guarantee is
marginal, so small test sets fluctuate: iris tests on 30 rows, where one
missed row moves coverage by 3.3%). `coverage_report()` itself warns when
empirical coverage drifts far from target.

## Study C: does the SHIP/WARN/STOP verdict gate correctly?

For each classification dataset we asked `report()` for a verdict twice:
on a model trained normally, and on a model trained after shuffling the
labels (destroying all signal). A correct gate says SHIP on the first and
refuses SHIP on the second.

| Condition | Verdict | Correct |
|---|---|---|
| Trained on real labels | SHIP | 6 / 6 |
| Trained on shuffled labels | STOP | 6 / 6 |

The shuffled models cannot beat the naive baseline, and the report says so
before anyone deploys them.

## Case studies

For the guardrails applied to a real research dataset (CMS Medicare Part B
provider fraud), see [case studies](case-studies.md).
