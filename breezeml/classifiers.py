"""
BreezeML Classifiers (v0.2.7)
Easy wrappers for popular classification algorithms with sensible preprocessing.

New in v0.2.6:
  - X_test / y_test parameters on every classifier — pass your own held-out set
    instead of relying on the internal 80/20 split.
  - macro_f1 added to every report dict alongside weighted F1.
  - logistic_regression() alias for logistic().
  - naive_bayes() alias for multinomial_nb() — optimised for TF-IDF text data.
New in v0.2.2:
  - Support for direct X and y array inputs (sparse/dense).
New in v0.2.0:
  - 5 new classifiers: knn, gradient_boosting, adaboost, extra_trees, mlp
  - compare()        → run all classifiers and rank them in one line
  - detailed_report() → confusion matrix, precision, recall, ROC-AUC
  - quick_tune()     → auto hyperparameter tuning with RandomizedSearchCV
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
from ._validation import check_df_target
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _detect_types(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


def _preprocessor(num_cols, cat_cols):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols)
    ])


def _train(model, df: pd.DataFrame = None, target: str = None, X=None, y=None, X_test=None, y_test=None):
    """Train a classifier and return (pipeline, report).

    The report contains accuracy, weighted F1, and macro F1.
    Pass X_test / y_test to evaluate on your own held-out set instead of the internal 80/20 split.
    """
    if X is not None and y is not None:
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        if X_test is not None and y_test is not None:
            X_tr, y_tr = X, y_series
            X_te = X_test
            y_te = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
        else:
            stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)
        pipe = Pipeline([("model", model)])
        pipe.fit(X_tr, y_tr)
    else:
        check_df_target(df, target)
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X_df, y_df
            X_te = X_test
            y_te = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
        else:
            stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
            X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)
        pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    report = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
    }
    return pipe, report


# ─── Individual Classifiers ──────────────────────────────────────────────────

def logistic(df: pd.DataFrame = None, target: str = None, max_iter: int = 500, *, X=None, y=None, X_test=None, y_test=None):
    """Logistic Regression classifier."""
    return _train(LogisticRegression(max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def svm(df: pd.DataFrame = None, target: str = None, kernel: str = "rbf", C: float = 1.0, gamma: str | float = "scale", *, X=None, y=None, X_test=None, y_test=None):
    """Support Vector Machine (SVC) classifier."""
    return _train(SVC(kernel=kernel, C=C, gamma=gamma, probability=True), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def linear_svm(df: pd.DataFrame = None, target: str = None, C: float = 1.0, max_iter: int = 5000, *, X=None, y=None, X_test=None, y_test=None):
    """Linear SVM (LinearSVC)."""
    return _train(LinearSVC(C=C, dual=False, class_weight='balanced', max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def gaussian_nb(df: pd.DataFrame = None, target: str = None, *, X=None, y=None, X_test=None, y_test=None):
    """Gaussian Naïve Bayes — good for numeric features."""
    return _train(GaussianNB(), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def multinomial_nb(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None):
    """Multinomial Naïve Bayes — good for counts/TF-IDF (non-negative)."""
    return _train(MultinomialNB(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def decision_tree(df: pd.DataFrame = None, target: str = None, random_state: int = 42, max_depth: int | None = None, *, X=None, y=None, X_test=None, y_test=None):
    """Decision Tree classifier."""
    return _train(DecisionTreeClassifier(random_state=random_state, max_depth=max_depth), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def random_forest(df: pd.DataFrame = None, target: str = None, n_estimators: int = 200, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None):
    """Random Forest classifier."""
    return _train(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


# ── NEW in v0.2.0 ────────────────────────────────────────────────────────────

def knn(df: pd.DataFrame = None, target: str = None, n_neighbors: int = 5, *, X=None, y=None, X_test=None, y_test=None):
    """K-Nearest Neighbors classifier."""
    return _train(KNeighborsClassifier(n_neighbors=n_neighbors), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def gradient_boosting(df: pd.DataFrame = None, target: str = None, n_estimators: int = 200, learning_rate: float = 0.1,
                      random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None):
    """Gradient Boosting classifier — often the most accurate out-of-the-box."""
    return _train(
        GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test
    )


def adaboost(df: pd.DataFrame = None, target: str = None, n_estimators: int = 100, learning_rate: float = 1.0,
             random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None):
    """AdaBoost classifier — boosts weak learners sequentially."""
    return _train(
        AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test
    )


def extra_trees(df: pd.DataFrame = None, target: str = None, n_estimators: int = 200, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None):
    """Extra Trees (Extremely Randomized Trees) classifier."""
    return _train(ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test)


def mlp(df: pd.DataFrame = None, target: str = None, hidden_layer_sizes=(100,), max_iter: int = 500, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None):
    """Multi-Layer Perceptron (Neural Network) classifier."""
    return _train(
        MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state),
        df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test
    )


# ─── compare() — Run all classifiers, return a ranked leaderboard ────────────

# Registry of all built-in classifiers (name → callable with default args)
_CLASSIFIERS = {
    "Logistic Regression":   lambda: LogisticRegression(max_iter=500),
    "SVM (RBF)":             lambda: SVC(probability=True),
    "Linear SVM":            lambda: LinearSVC(dual=False, class_weight='balanced'),
    "Gaussian NB":           lambda: GaussianNB(),
    "Decision Tree":         lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest":         lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN":                   lambda: KNeighborsClassifier(),
    "Gradient Boosting":     lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost":              lambda: AdaBoostClassifier(random_state=42),
    "Extra Trees":           lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
    "MLP (Neural Net)":      lambda: MLPClassifier(max_iter=500, random_state=42),
}


def compare(df: pd.DataFrame = None, target: str = None, show: bool = True, *, X=None, y=None):
    """Run every classifier on *df* and return results sorted by accuracy.

    Parameters
    ----------
    df : DataFrame with features + target column.
    target : name of the target column.
    show : if True, print a pretty leaderboard table.

    Returns
    -------
    list[dict]  –  sorted list of {"classifier", "accuracy", "f1"} dicts.

    Example
    -------
    >>> from breezeml import classifiers, datasets
    >>> results = classifiers.compare(datasets.iris(), "species")
    """
    if X is None or y is None:
        check_df_target(df, target)

    def _run_one(name, factory_func):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, report = _train(factory_func(), df=df, target=target, X=X, y=y)
            return {"classifier": name, **report}
        except Exception:
            return {"classifier": name, "accuracy": None, "f1": None}

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(
        delayed(_run_one)(name, factory) for name, factory in _CLASSIFIERS.items()
    )

    # Sort: best accuracy first, None values at the bottom
    results.sort(key=lambda r: r["accuracy"] if r["accuracy"] is not None else -1, reverse=True)

    if show:
        print(f"\n🏆 BreezeML Classifier Leaderboard — target: '{target}'")
        print(f"{'Rank':<6}{'Classifier':<25}{'Accuracy':<12}{'F1':<12}")
        print("─" * 55)
        for i, r in enumerate(results, 1):
            acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "FAILED"
            f1 = f"{r['f1']:.4f}" if r["f1"] is not None else "FAILED"
            print(f"{i:<6}{r['classifier']:<25}{acc:<12}{f1:<12}")
        print()

    return results


# ─── detailed_report() — Full metrics for any trained classifier ─────────────

def detailed_report(df: pd.DataFrame = None, target: str = None, model=None, algo: str = "random_forest", *, X=None, y=None):
    """Get a detailed classification report with everything a beginner needs.

    Pass an already-trained *model* (pipeline), or let it train one via *algo*.

    Parameters
    ----------
    df : DataFrame with features + target column.
    target : name of the target column.
    model : a trained sklearn Pipeline (from any classifier function). Optional.
    algo : which classifier to train if model is None.
           Options: "logistic", "svm", "knn", "decision_tree", "random_forest",
                    "gradient_boosting", "adaboost", "extra_trees", "mlp"

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, roc_auc,
                    confusion_matrix, classification_report, model

    Example
    -------
    >>> from breezeml import classifiers, datasets
    >>> info = classifiers.detailed_report(datasets.iris(), "species")
    >>> print(info["accuracy"], info["confusion_matrix"])
    """
    if X is None or y is None:
        check_df_target(df, target)

    _ALGO_MAP = {
        "logistic":           lambda: LogisticRegression(max_iter=500),
        "svm":                lambda: SVC(probability=True),
        "linear_svm":         lambda: LinearSVC(dual=False, class_weight='balanced'),
        "gaussian_nb":        lambda: GaussianNB(),
        "decision_tree":      lambda: DecisionTreeClassifier(random_state=42),
        "random_forest":      lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "knn":                lambda: KNeighborsClassifier(),
        "gradient_boosting":  lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
        "adaboost":           lambda: AdaBoostClassifier(random_state=42),
        "extra_trees":        lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "mlp":                lambda: MLPClassifier(max_iter=500, random_state=42),
    }

    if X is not None and y is not None:
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)

        if model is None:
            factory = _ALGO_MAP.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_MAP.keys())}")
            pipe = Pipeline([("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model
    else:
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _preprocessor(num_cols, cat_cols)

        stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)

        if model is None:
            factory = _ALGO_MAP.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_MAP.keys())}")
            pipe = Pipeline([("pre", pre), ("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model

    pred = pipe.predict(X_te)

    # ROC-AUC (works for binary and multiclass)
    roc = None
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)
            if proba.shape[1] == 2:
                roc = round(float(roc_auc_score(y_te, proba[:, 1])), 4)
            else:
                le = LabelEncoder()
                y_te_enc = le.fit_transform(y_te)
                roc = round(float(roc_auc_score(y_te_enc, proba, multi_class="ovr", average="weighted")), 4)
    except Exception:
        roc = None

    cm = confusion_matrix(y_te, pred).tolist()
    cr = classification_report(y_te, pred, output_dict=True, zero_division=0)

    result = {
        "accuracy":              round(float(accuracy_score(y_te, pred)), 4),
        "f1":                    round(float(f1_score(y_te, pred, average="weighted")), 4),
        "macro_f1":              round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "precision":             round(float(precision_score(y_te, pred, average="weighted", zero_division=0)), 4),
        "recall":                round(float(recall_score(y_te, pred, average="weighted", zero_division=0)), 4),
        "roc_auc":               roc,
        "confusion_matrix":      cm,
        "classification_report": cr,
        "model":                 pipe,
    }
    return result


# ─── quick_tune() — Auto hyperparameter tuning in one line ───────────────────

# Pre-built param grids for each algorithm (beginner-friendly defaults)
_PARAM_GRIDS = {
    "logistic": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__max_iter": [200, 500, 1000],
    },
    "svm": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 500],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "adaboost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "extra_trees": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [5, 10, 20, None],
    },
    "mlp": {
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "model__learning_rate_init": [0.001, 0.01],
        "model__max_iter": [300, 500],
    },
}

_ALGO_FACTORIES = {
    "logistic":           lambda: LogisticRegression(max_iter=500),
    "svm":                lambda: SVC(probability=True),
    "knn":                lambda: KNeighborsClassifier(),
    "decision_tree":      lambda: DecisionTreeClassifier(random_state=42),
    "random_forest":      lambda: RandomForestClassifier(random_state=42),
    "gradient_boosting":  lambda: GradientBoostingClassifier(random_state=42),
    "adaboost":           lambda: AdaBoostClassifier(random_state=42),
    "extra_trees":        lambda: ExtraTreesClassifier(random_state=42),
    "mlp":                lambda: MLPClassifier(random_state=42),
}


def quick_tune(df: pd.DataFrame = None, target: str = None, algo: str = "random_forest",
               n_iter: int = 20, cv: int = 3, scoring: str = "accuracy", *, X=None, y=None):
    """Auto-tune a classifier's hyperparameters in one line.

    Parameters
    ----------
    df : DataFrame with features + target column.
    target : name of the target column.
    algo : which classifier to tune.
           Options: "logistic", "svm", "knn", "decision_tree", "random_forest",
                    "gradient_boosting", "adaboost", "extra_trees", "mlp"
    n_iter : number of random combinations to try (default 20).
    cv : cross-validation folds (default 3).
    scoring : metric to optimize (default "accuracy").

    Returns
    -------
    (best_model, best_params, report)
        best_model  — trained Pipeline with the best hyperparameters
        best_params — dict of the winning parameter combination
        report      — {"accuracy": ..., "f1": ...} on the held-out test set

    Example
    -------
    >>> from breezeml import classifiers, datasets
    >>> model, params, report = classifiers.quick_tune(datasets.iris(), "species")
    >>> print(params, report)
    """
    if X is None or y is None:
        check_df_target(df, target)

    factory = _ALGO_FACTORIES.get(algo)
    param_grid = _PARAM_GRIDS.get(algo)
    if factory is None or param_grid is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(_ALGO_FACTORIES.keys())}")

    if X is not None and y is not None:
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        pipe = Pipeline([("model", factory())])
        stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_df = df.drop(columns=[target])
        y_df = df[target]
        num_cols, cat_cols = _detect_types(df, target)
        pre = _preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", pre), ("model", factory())])

        stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)

    # Cap n_iter to actual grid size so we don't get warnings
    from itertools import product as _product
    grid_size = 1
    for v in param_grid.values():
        grid_size *= len(v)
    actual_iter = min(n_iter, grid_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            pipe, param_grid,
            n_iter=actual_iter, cv=cv, scoring=scoring,
            random_state=42, n_jobs=-1, error_score="raise",
        )
        search.fit(X_tr, y_tr)

    best_pipe = search.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    pred = best_pipe.predict(X_te)
    report = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
    }

    print(f"✅ Best params for {algo}: {best_params}")
    print(f"   Accuracy: {report['accuracy']}  |  F1: {report['f1']}  |  Macro F1: {report['macro_f1']}")

    return best_pipe, best_params, report


# ─── Aliases ─────────────────────────────────────────────────────────────────

logistic_regression = logistic
naive_bayes = multinomial_nb
