# Automatic feature engineering: enrich without leaking

`breezeml.autofeat.engineer()` turns a raw DataFrame into a richer, model-ready
one and hands back a "what I did and why" report for every transformation. It
is the higher-level companion to `breezeml.features`: where `features` offers
individual helpers (select, pca, polynomial), `engineer` runs a whole
opinionated pipeline in one call. The headline promise is that it enriches the
feature set WITHOUT leaking the target: high-cardinality target encoding uses
out-of-fold estimates, and every step is bounded by an explicit cap so the
feature count can never explode.

```python
import breezeml

# df is your raw table: a mix of numeric, categorical, and date columns.
new_df, report = breezeml.autofeat.engineer(df, "target")
print(report["n_features_before"], "->", report["n_features_after"])
print(report["added"], report["dropped"])
```

The input DataFrame is never mutated; a brand new one is returned with the
target column preserved untouched. `report` is a dict with `added`, `dropped`,
`encoded`, `datetime_expanded`, `n_features_before`, `n_features_after`, and
`notes`.

## What the pipeline does, in order

1. **Datetime expansion.** Columns that are datetime dtype (or object columns
   that parse cleanly as dates at least 90% of the time) are split into
   `year`, `month`, `day`, `dayofweek`, and `is_weekend`, plus `hour` when the
   column actually carries a time of day. The raw datetime column is dropped.
   Pass `datetime_cols=[...]` to force-treat columns that auto-detection
   misses. Integer id columns are never coerced to dates.
2. **High-cardinality categorical encoding.** A categorical with more than 20
   distinct levels (or more distinct levels than 5% of the rows) gets a
   frequency encoding (`<col>_freq`, each level's share of the rows) and, when
   the target is numeric or binary, a leakage-safe out-of-fold target-mean
   encoding (`<col>_target_enc`). The raw high-card column is dropped to avoid a
   one-hot explosion. Low-cardinality categoricals are left alone for normal
   one-hot encoding downstream.
3. **Numeric interactions.** Pairwise products (`<a>_x_<b>`) among the most
   informative numeric columns, ranked by absolute correlation with the target
   (or by variance when no numeric target is available). Capped at the top 5
   base columns and 10 interaction columns so the space cannot blow up.
4. **Pruning.** Constant columns (single value, no signal) are dropped, and one
   column from each near-perfectly-correlated numeric pair (`|r| > 0.98`) is
   dropped to kill redundancy and multicollinearity.

## Why the target encoding is leakage-safe

A naive target encoding maps each level to the mean target over ALL rows,
including the row being encoded, which leaks the label straight into the
feature. `engineer` instead uses 5-fold out-of-fold means: for each fold the
level means are computed from the OTHER folds only, so a row's own target never
touches its own encoding. Unseen levels fall back to the global mean. This is
the same discipline the honest-ML tools enforce elsewhere: a feature that has
seen its own label is a leak waiting to flatter your metrics.

## Reading the report

```python
new_df, report = breezeml.autofeat.engineer(df, "target", show=True)
# BreezeML Automatic Feature Engineering - target: 'target'
#   Features: 8 in  ->  14 out
#   High-cardinality encoded:
#     - 'city' (312 levels): frequency, target_mean_oof  [out-of-fold target mean (5-fold, leakage-safe)]
#   Numeric interactions added:
#     - age_x_income
#   Dropped:
#     - 'dup_income': near-duplicate of 'income' (|r|=0.994 > 0.98)
```

With `show=True` (the default) it prints the same accounting. `encoded` lists
each high-card column with its level count, encodings, and method;
`datetime_expanded` lists each date column and the parts it became; `dropped`
carries a reason per column; `notes` explains each step in plain English.

## When NOT to use it

- **Multiclass string targets skip target encoding.** Out-of-fold target-mean
  encoding needs a numeric or binary target. With a multiclass string target,
  high-card categoricals get frequency encoding only, and the report says so.
  You lose the strongest signal for those columns.
- **It only builds pairwise numeric interactions, and only a capped few.** If
  the signal lives in a three-way interaction or in a pair outside the top-5
  ranked columns, `engineer` will not find it. It is a strong default, not a
  search over all feature combinations.
- **It is not a substitute for domain knowledge.** Automatic features are
  generic. A single hand-built ratio that encodes how your business actually
  works often beats everything here. Use `engineer` to enrich a baseline, then
  add the features only you know to build.
- **Always re-audit and validate downstream.** The pruning and out-of-fold
  discipline reduce leakage risk, but you should still run [`audit()`](honest-ml.md)
  on the engineered frame and judge it by honest holdout metrics. More features
  is not more signal; it is more chances to overfit.
