"""Build the beginner Kaggle notebook (first ML model, everything explained)."""
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
    md("""# Your First Machine Learning Model, Explained Simply

You have never trained a machine learning model before. That is fine. By the end of this notebook you will have trained one, tested it, and understood every step. No math, no jargon without an explanation.

**The task:** the Titanic sank in 1912. We have a list of passengers, who they were, and whether they survived. We will teach the computer to look at a passenger and guess: survived, or not?

That is all machine learning is. Showing a computer many examples, letting it find patterns, and asking it to guess about examples it has not seen.

One setting first: in the sidebar on the right, turn **Internet ON**. We need it to install one tool."""),

    md("""## Step 0: Install the tool

We will use a free library called **BreezeML**. It sits on top of scikit-learn (the standard Python ML library) and does the tedious parts for us, correctly.

`pip` is Python's app store. This line downloads and installs the library:"""),

    code("%pip install -q breezeml"),

    md("""## Step 1: Look at the data

Data comes as a table, like a spreadsheet. In Python this table is called a **DataFrame**. Each **row** is one passenger. Each **column** is one fact about them: age, ticket class, sex, fare paid.

Let's load the table and look at the first five rows:"""),

    code("""import glob
import pandas as pd
import breezeml

csv_path = glob.glob("/kaggle/input/**/*.csv", recursive=True)[0]
df = pd.read_csv(csv_path)
df.head()"""),

    md("""Things to notice:

- The **Survived** column holds the answer: 1 means survived, 0 means did not. This is the column we want the computer to learn to guess. It is called the **target**.
- Some Age cells say NaN. That means the value is **missing**, nobody recorded it. Real data is always messy like this.
- Some columns are words (Sex, Embarked), some are numbers. Computers can only do math on numbers, so the words will need converting.

Normally you would now write a lot of code: fill in missing ages, convert words to numbers, split the data. BreezeML does all of that with sane defaults, so we can focus on understanding."""),

    md("""## Step 2: Train a model

**Training** means: show the computer the passengers AND the answers, and let it find patterns. Maybe it notices women survived more often. Maybe first class beat third class. It finds these patterns on its own.

One important trick happens silently here: the library keeps part of the data hidden from the model during training. Like a teacher who keeps some exam questions secret. Testing the model on questions it never saw is the only honest way to measure it."""),

    code("""model = breezeml.fit(df, "Survived")"""),

    md("""That one line just did all of this:

1. Set aside the Survived column as the answer to learn.
2. Filled in missing values (missing ages became the typical age).
3. Converted word columns like Sex into numbers.
4. Split the data: most rows for learning, some kept hidden for testing.
5. Trained a model called a random forest, which is a crowd of decision trees that vote.

Now, is the model any good? Let's ask for an honest report:"""),

    code("""report = model.report(df)"""),

    md("""## Step 3: Read the report (this is the important part)

The report just told us to **STOP**. The model scored decently, so why?

Look at the audit line: some columns "look like row identifiers."

Think about `PassengerId`. It is just a row number: 1, 2, 3... It says nothing about who lives or dies. But a model can **memorize** it, like a student memorizing that question 7's answer is C without reading question 7. The memorizing student aces the practice test and fails the real exam.

This mistake is called **leakage**, and it is everywhere in real projects. Most tutorials never check for it. The report checks, and refuses to approve the model until it is fixed.

The fix is simple. Drop the columns that are just labels for rows, not facts about people:"""),

    code("""clean = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

model = breezeml.fit(clean, "Survived")
report = model.report(clean)"""),

    md("""Better. This time the report approves. Read its lines with what you now know:

- **performance**: the score on the hidden test questions. Roughly, the fraction of passengers it guesses correctly.
- **vs baseline**: the model must beat the laziest possible strategy, which is always guessing the most common answer ("everyone died" is right about 62% of the time here). A model that cannot beat lazy guessing has learned nothing. Ours beats it.
- **data audit**: no more identifier columns or other traps.
- **balance**: are the two answers (survived / died) roughly balanced in the data? Here, close enough. When one answer is very rare, plain accuracy becomes misleading, so it is worth checking.

Same data. Same library. The difference is that now the score is real."""),

    md("""## Step 4: Make a prediction

The whole point. Meet a passenger the model must judge, a 25-year-old woman in second class:"""),

    code("""passenger = pd.DataFrame([{
    "Pclass": 2, "Sex": "female", "Age": 25,
    "SibSp": 0, "Parch": 0, "Fare": 20.0, "Embarked": "S",
}])

prediction = model.predict(passenger)
print("Prediction:", "survived" if prediction[0] == 1 else "did not survive")"""),

    md("""Change the values and re-run. Make her a him. Move him to third class. Watch the prediction flip. You are now interrogating a machine learning model, which is exactly what practitioners do all day."""),

    md("""## Step 5: Try many models at once

We used one kind of model. There are many others, each finds patterns differently. Which is best for this data? Try 22 of them and rank the results:"""),

    code("""leaderboard = breezeml.classifiers.compare(clean, "Survived")
pd.DataFrame(leaderboard).head(10)"""),

    md("""Each model was tested the honest way (on data it had not seen) several times, and the scores averaged. The differences near the top are usually small. Picking the winner of this table is how real projects choose their model."""),

    md("""## Step 6: Keep what you learned

Two parting gifts from the library.

First, a **model card**: a short document describing what this model is, how it was tested, and its known weak points. Professionals ship one with every model:"""),

    code("""model.card("MODEL_CARD.md")
print(open("MODEL_CARD.md").read())"""),

    md("""Second, and this matters if you keep learning: `export()` writes the **plain scikit-learn code** that does everything we just did, with no BreezeML in it. Read it when you are ready. It is the code you would have written by hand, and one day you will:"""),

    code("""model.export("train_sklearn.py")
print(open("train_sklearn.py").read())"""),

    md("""## What you now know

- Data is a table; the **target** is the column you want to predict.
- **Training** is pattern-finding on examples; honest **testing** happens on hidden examples.
- A model must beat the **baseline** (lazy guessing) to mean anything.
- **Leakage** (answer hints hiding in the data) silently ruins models, and you know how to catch it.
- How to make a prediction, compare models, and read a model card.

That is a genuinely solid foundation. More than many people who ship models have.

**Where to go next:**

- BreezeML repository and docs: [github.com/venomez-viper/breezeml](https://github.com/venomez-viper/breezeml)
- Stuck or curious? Ask anything in [GitHub Discussions](https://github.com/venomez-viper/breezeml/discussions), beginner questions welcome.
- If this helped you, an upvote helps the next beginner find it.

```
pip install breezeml
```"""),
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

out = Path(__file__).parent / "first_ml_model_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
