# Model cards & teaching narration

## `explain_decisions=True`: learn while you train

Every automatic decision BreezeML makes can explain itself in plain English:

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model, report = breezeml.auto(df, "species", explain_decisions=True)
```

```
BreezeML decisions explained:
  1. Detected a classification task: target 'species' has only 3 distinct values...
  2. Used a stratified 80/20 train/test split because it keeps class proportions...
  3. No missing numeric values found; the median imputer stays in the pipeline...
  4. Standardized 4 numeric column(s) to mean 0 / std 1...
  5. Classes are reasonably balanced, so accuracy and weighted F1 are both fair metrics.
  6. Caution: only 150 rows. Test metrics from a single split are noisy...
```

Also available after training: `model.explain_decisions()`.

The narration is generated from measured facts about *your* data (missing
percentages, outlier detection via the IQR rule, class imbalance ratios),
not canned text.

## `card()`: an honest model card in one call

```python
text = breezeml.card(model)            # markdown string
breezeml.card(model, "MODEL_CARD.md")  # or write to file
```

The card includes:

- **Overview**: task, estimator, non-default hyperparameters, random seed
- **Training data**: rows, feature types, missing cells, class balance
- **Evaluation**: holdout strategy and all metrics
- **Pipeline decisions**: the full teaching narration
- **Limitations & caveats**: auto-detected risks (small data, imbalance,
  heavy imputation) plus drift and fairness warnings
- **Reproducibility**: pointer to `export()` for the standalone script

Model cards follow the spirit of [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993).
Attach one to every model you hand to someone else.

## When NOT to use it

Cards are generated for models trained through the core API
(`fit` / `auto` / `classify` / `regress`, v0.4+). Raw pipelines returned by
`classifiers.<algo>()` carry no metadata, so `card()` raises a clear error.
