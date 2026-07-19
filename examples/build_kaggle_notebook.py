"""Build the Kaggle launch notebook (honest-ml-titanic) as .ipynb JSON."""
import json
from pathlib import Path

_n = [0]


def _id():
    _n[0] += 1
    return f"cell-{_n[0]:02d}"


def md(s):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "id": _id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": s}

cells = [
    md("""# Honest ML on Titanic: SHIP / WARN / STOP

Most notebooks stop at "accuracy looks good."
This one asks a harder question: **does this model deserve to ship?**

We will use [BreezeML](https://github.com/venomez-viper/breezeml), an open-source workflow layer over scikit-learn built on one idea:

> The statistically sound path should be the shortest path.

Every model gets a single verdict, **SHIP**, **WARN**, or **STOP**, from four checks:

| Check | Question it answers |
|---|---|
| Baseline | Does it beat always guessing the majority class? |
| Leakage audit | Did the answer sneak into the features? |
| Imbalance | Will accuracy mislead here? |
| Fairness (optional) | Does it treat groups equally? |

The core installs with four dependencies: scikit-learn, pandas, numpy, joblib. That is the whole footprint.

*Settings: turn Internet ON in the right sidebar so pip can install.*"""),

    code("%pip install -q breezeml"),

    md("""## 1. Load the data

Nothing special yet. The usual Titanic training set."""),

    code("""import pandas as pd
import breezeml

df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()"""),

    md("""## 2. First attempt: throw everything at the model

One line to train. Leakage-safe preprocessing, stratified split, fixed seed, all default. One line to judge."""),

    code("""model = breezeml.fit(df, "Survived")
report = model.report(df)"""),

    md("""**STOP.**

Not because the accuracy is bad. Because the audit noticed columns like `PassengerId` that look like row identifiers. A model can memorize IDs, score well here, and learn nothing about survival.

Most tutorials never run this check. The report refuses to say SHIP until it is resolved.

The full audit, in plain language:"""),

    code("""audit = breezeml.audit(df, "Survived")"""),

    md("""## 3. Second attempt: fix what the audit found

Drop the identifier-like columns. Keep the signal."""),

    code("""clean = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

model = breezeml.fit(clean, "Survived")
report = model.report(clean)"""),

    md("""**SHIP.**

Beats the naive baseline. No critical findings. Mild imbalance, noted but not blocking.

Same data, same library, three lines. The difference is that now the score means something."""),

    md("""## 4. Choose a model honestly

`compare()` cross-validates 22 classifiers and ranks them. No cherry-picking."""),

    code("""leaderboard = breezeml.classifiers.compare(clean, "Survived")
pd.DataFrame(leaderboard).head(10)"""),

    md("""## 5. Write the model card

`card()` documents what the model is, how it was evaluated, and the caveats it auto-detected. The things you are supposed to disclose, written for you."""),

    code("""model.card("MODEL_CARD.md")
print(open("MODEL_CARD.md").read())"""),

    md("""## 6. Graduate: export to pure scikit-learn

`export()` writes a standalone sklearn training script that reproduces this exact pipeline. No breezeml import in it.

The library is built to be outgrown. When you are ready to write scikit-learn by hand, you leave with everything."""),

    code("""model.export("train_sklearn.py")
print(open("train_sklearn.py").read())"""),

    md("""## The path onward

This was the first layer. On the same four dependencies, BreezeML also carries AutoML, conformal prediction intervals, drift monitoring, significance tests, survival analysis, fairness reports, and an MCP server that gives AI agents these same guardrails.

- Repository and docs: [github.com/venomez-viper/breezeml](https://github.com/venomez-viper/breezeml)
- The guardrails are validated empirically, detection rates and coverage measured, limitations stated: [docs/validation.md](https://github.com/venomez-viper/breezeml/blob/main/docs/validation.md)
- Questions and ideas: [GitHub Discussions](https://github.com/venomez-viper/breezeml/discussions)

If this notebook taught you something, an upvote helps other beginners find it, and a star on the repo helps the project.

```
pip install breezeml
```"""),

    md("""## One last thing

BreezeML keeps a small zen garden inside. ML wisdom, one breath at a time."""),

    code("""breezeml.haiku()"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "honest_ml_titanic_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
