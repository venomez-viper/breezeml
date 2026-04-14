# BreezeML v0.2.0 - The Classification Power-Up

What's new in v0.2.0

Colab Quickstart: https://colab.research.google.com/github/venomez-viper/breezeml/blob/main/examples/breezeml_quickstart.ipynb

## New Classifiers
5 more one-line classifiers added to `breezeml.classifiers`:
- `knn()` - K-Nearest Neighbors
- `gradient_boosting()` - Gradient Boosting
- `adaboost()` - AdaBoost ensemble
- `extra_trees()` - Extremely Randomized Trees
- `mlp()` - Multi-Layer Perceptron (Neural Network)

Total: 12 classifiers, all in one line.

## Compare All Classifiers
```python
results = classifiers.compare(df, "species")
```
Runs every classifier and prints a ranked leaderboard. Find the best model instantly.

## Detailed Report
```python
info = classifiers.detailed_report(df, "species")
```
Returns accuracy, F1, precision, recall, ROC-AUC, confusion matrix, and full classification report.

## Quick Tune
```python
model, params, report = classifiers.quick_tune(df, "species", algo="random_forest")
```
Auto hyperparameter tuning with RandomizedSearchCV. Built-in param grids for all 9 tunable classifiers.

## Try it
```bash
pip install --upgrade breezeml
```

Created and maintained by Akash Anipakalu Giridhar
"If you can load a CSV, you can train a model."
