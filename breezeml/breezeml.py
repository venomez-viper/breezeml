"""
BreezeML: Beginner-friendly wrapper around scikit-learn

Created by Akash Anipakalu Giridhar 🔥✨
v0.1.2
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn import datasets as skdatasets

__all__ = [
    "classify", "regress", "fit", "predict", "from_csv",
    "report", "save", "load", "auto", "creator", "datasets"
]


class EasyModel:
    def __init__(self, pipeline, target, task):
        self.pipeline = pipeline
        self.target = target
        self.task = task

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def _detect_types(df, target):
    X = df.drop(columns=[target])
    numeric = X.select_dtypes(include=[np.number]).columns
    categorical = X.select_dtypes(exclude=[np.number]).columns
    return list(numeric), list(categorical)


def _build_preprocessor(numeric, categorical):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])


def classify(df, target, algo="forest", return_report=True):
    X = df.drop(columns=[target])
    y = df[target]
    numeric, categorical = _detect_types(df, target)
    pre = _build_preprocessor(numeric, categorical)

    model = RandomForestClassifier(random_state=42) if algo == "forest" else LogisticRegression(max_iter=200)
    pipe = Pipeline([("pre", pre), ("model", model)])

    stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    em = EasyModel(pipe, target, "classification")

    if not return_report:
        return em
    preds = pipe.predict(X_test)
    report = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="weighted"))
    }
    return em, report


def regress(df, target, algo="forest", return_report=True):
    X = df.drop(columns=[target])
    y = df[target]
    numeric, categorical = _detect_types(df, target)
    pre = _build_preprocessor(numeric, categorical)

    model = RandomForestRegressor(random_state=42) if algo == "forest" else LinearRegression()
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    em = EasyModel(pipe, target, "regression")

    if not return_report:
        return em
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)  # support for older sklearn
    rmse = float(np.sqrt(mse))
    report = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": rmse
    }
    return em, report


def fit(df, target):
    y = df[target]
    if y.dtype == "object" or y.nunique() < 20:
        m, _ = classify(df, target)
    else:
        m, _ = regress(df, target)
    return m


def predict(model, X):
    return model.predict(X)


def from_csv(path, target):
    df = pd.read_csv(path)
    model = fit(df, target)
    y = df[target]
    preds = model.predict(df.drop(columns=[target]))

    if model.task == "classification":
        return model, {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="weighted"))
        }
    else:
        mse = mean_squared_error(y, preds)
        rmse = float(np.sqrt(mse))
        return model, {
            "r2": float(r2_score(y, preds)),
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": rmse
        }


def report(model, df):
    y = df[model.target]
    preds = model.predict(df.drop(columns=[model.target]))
    if model.task == "classification":
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="weighted"))
        }
    else:
        mse = mean_squared_error(y, preds)
        rmse = float(np.sqrt(mse))
        return {
            "r2": float(r2_score(y, preds)),
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": rmse
        }


def save(model, path):
    model.save(path)


def load(path):
    return EasyModel.load(path)


def auto(df, target):
    """Automatically pick classification or regression based on target."""
    y = df[target]
    if y.dtype == "object" or y.nunique() < 20:
        return classify(df, target)
    else:
        return regress(df, target)


def creator():
    return "Created by Akash Anipakalu Giridhar 🔥✨"


class datasets:
    @staticmethod
    def iris():
        data = skdatasets.load_iris(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "species"}, inplace=True)
        return df

    @staticmethod
    def breast_cancer():
        data = skdatasets.load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "label"}, inplace=True)
        return df

    @staticmethod
    def wine():
        data = skdatasets.load_wine(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "class"}, inplace=True)
        return df

    @staticmethod
    def diabetes():
        data = skdatasets.load_diabetes(as_frame=True)
        df = data.frame.copy()
        return df
