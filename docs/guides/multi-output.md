# Multi-output: predicting many things at once

`breezeml.multi` answers a question the single-target API cannot: what if one
row has several answers at the same time? A support ticket can be tagged
"billing" AND "urgent" AND "refund" together (multi-label classification). A
sensor reading maps to temperature AND pressure at once (multi-output
regression). This module trains and scores both, with per-target metrics plus
the two whole-row scores that matter when there is more than one label.

```python
from breezeml import multi

# Several label columns predicted together
model, report = multi.multi_label(df, ["billing", "urgent", "refund"])
print(report["subset_accuracy"], report["hamming_loss"])

# Several numeric columns predicted together
model, report = multi.multi_output(df, ["temperature", "pressure"])
print(report["average_r2"])
```

Both helpers take a **list** of target columns (passing a single string raises
a helpful error). Every remaining column becomes a feature, so features and
targets can never overlap by construction. Both build on the same
ColumnTransformer preprocessing and `build_meta` metadata as the rest of
BreezeML, so `save()`, cards, and downstream tooling keep working. The returned
model stores the list on both `.target` and `.targets`.

## `multi_label()`: several label columns at once

The default trains one independent `RandomForestClassifier` per label (a
`MultiOutputClassifier`). Set `chain=True` to use a `ClassifierChain` instead,
which predicts labels in sequence and feeds each prediction forward as a
feature for the next label, so the model can capture correlations between
labels (for example, "urgent" and "refund" tending to co-occur). Pass your own
`base=` estimator to swap the per-label model.

```python
# Chain the labels to capture their correlations
model, report = multi.multi_label(
    df, ["billing", "urgent", "refund"], chain=True
)
```

The report has `per_target` (accuracy and weighted F1 for each label) plus two
whole-row numbers:

- `subset_accuracy` is the **exact-match ratio**: the fraction of rows where
  *every* label was predicted correctly. It is strict, and it is the honest
  score for multi-label work, because getting three of four labels right on a
  row is not a correct row.
- `hamming_loss` is the fraction of individual label predictions that were
  wrong, averaged over all labels and rows. It is the gentler, per-label view.

The task is recorded as `multi_label` and the split is a plain (unstratified)
80/20, because stratification is not well defined across several targets at
once.

## `multi_output()`: several numeric columns at once

`multi_output()` fits one `RandomForestRegressor` per numeric target (a
`MultiOutputRegressor`) and reports `per_target` r2/mae/rmse plus `average_r2`,
the mean R2 across the targets. As with `multi_label()`, pass `base=` to swap
the per-target regressor.

```python
model, report = multi.multi_output(df, ["temperature", "pressure"])
for name, m in report["per_target"].items():
    print(name, m["r2"], m["mae"], m["rmse"])
```

## When NOT to use it

- **Independent models are the default; correlated labels want the chain.**
  `MultiOutputClassifier` and `MultiOutputRegressor` fit each target in
  isolation and cannot exploit relationships between targets. If your labels
  co-occur in patterns, try `chain=True` on `multi_label()` and compare the
  subset accuracy; if they are genuinely independent, the chain buys nothing.
- **Do not read `subset_accuracy` as if it were single-label accuracy.** With
  many labels, exact-match accuracy is punishingly strict by design. A model
  can be useful at a low subset accuracy if its `hamming_loss` is small; look
  at both, and at the per-target scores, before judging it.
- **This is not a substitute for one strong single-target model.** If you only
  ever need one of the targets, train it on its own; the multi-output wrappers
  add per-target models and complexity you do not need for a single answer.
- **The split is unstratified.** With rare labels a plain split can leave a
  label almost absent from the test slice, making its per-target metric noisy.
  Read per-target scores on rare labels with that caveat in mind.
