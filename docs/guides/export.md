# Export: graduate from BreezeML anytime

`export()` turns any trained BreezeML model into a **standalone scikit-learn
training script** with zero breezeml imports. Your pipeline, your code, no
lock-in.

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model = breezeml.fit(df, "species")

breezeml.export(model, "train.py", data_path="iris.csv")
# or: model.export("train.py")
```

The generated `train.py` reproduces the exact pipeline (median imputer,
scaler, one-hot encoder, estimator with the same hyperparameters, same seed,
same split) using only scikit-learn, pandas, and joblib. Run it directly:

```bash
python train.py
# accuracy: 0.9667
# f1 (weighted): 0.9666
# Saved trained pipeline to model.joblib
```

## Why this exists

Most high-level ML libraries want to keep you inside their abstraction.
BreezeML does the opposite: the moment you outgrow it, one call hands you
plain sklearn code you fully own and can edit line by line.

## When NOT to use it

If you rely on `breezeml.text.embed()` (sentence-transformer columns) the
embedding step happens before the exported pipeline; you must re-embed new
data yourself in the exported script.
