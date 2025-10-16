"""
BreezeML Classifiers (v0.1.2)
Easy wrappers for popular classification algorithms with sensible preprocessing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


def _train(model, df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols, cat_cols = _detect_types(df, target)
    pre = _preprocessor(num_cols, cat_cols)
    pipe = Pipeline([("pre", pre), ("model", model)])

    stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    report = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred, average="weighted"))
    }
    return pipe, report


def logistic(df: pd.DataFrame, target: str, max_iter: int = 500):
    """Logistic Regression classifier."""
    return _train(LogisticRegression(max_iter=max_iter), df, target)


def svm(df: pd.DataFrame, target: str, kernel: str = "rbf", C: float = 1.0, gamma: str | float = "scale"):
    """Support Vector Machine (SVC) classifier."""
    return _train(SVC(kernel=kernel, C=C, gamma=gamma, probability=True), df, target)


def linear_svm(df: pd.DataFrame, target: str, C: float = 1.0):
    """Linear SVM (LinearSVC)."""
    return _train(LinearSVC(C=C), df, target)


def gaussian_nb(df: pd.DataFrame, target: str):
    """Gaussian Naïve Bayes — good for numeric features."""
    return _train(GaussianNB(), df, target)


def multinomial_nb(df: pd.DataFrame, target: str, alpha: float = 1.0):
    """Multinomial Naïve Bayes — good for counts/TF-IDF (non-negative)."""
    return _train(MultinomialNB(alpha=alpha), df, target)


def decision_tree(df: pd.DataFrame, target: str, random_state: int = 42, max_depth: int | None = None):
    """Decision Tree classifier."""
    return _train(DecisionTreeClassifier(random_state=random_state, max_depth=max_depth), df, target)


def random_forest(df: pd.DataFrame, target: str, n_estimators: int = 200, random_state: int = 42):
    """Random Forest classifier."""
    return _train(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state), df, target)
