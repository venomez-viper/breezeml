# BreezeML v0.1.0

If you can load a CSV, you can train a model.

Quickstart:
```python
from breezeml import datasets, fit, predict, creator
print(creator())
df = datasets.iris()
model = fit(df, "species")
print(predict(model, df.drop(columns=["species"]))[:5])
```
