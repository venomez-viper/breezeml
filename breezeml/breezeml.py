"""
BreezeML: Beginner-friendly wrapper around scikit-learn

Created by Akash Anipakalu Giridhar
v0.3.0
"""
import pandas as pd
import joblib

from ._validation import check_df_target
from ._preprocessing import _detect_types, _build_preprocessor
from . import regressors

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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


def classify(df, target, algo="forest", return_report=True):
    check_df_target(df, target)
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
    algo_map = {
        "forest": "random_forest",
        "random_forest": "random_forest",
        "linear": "linear",
    }
    algo_name = algo_map.get(algo, algo)
    train_fn = getattr(regressors, algo_name, None)
    if train_fn is None:
        raise ValueError(f"Unknown regression algorithm '{algo}'.")

    pipe, reg_report = train_fn(df, target)
    em = EasyModel(pipe, target, "regression")

    if not return_report:
        return em
    return em, reg_report


def fit(df, target, task="auto"):
    m, _ = auto(df, target, task=task)
    return m


def predict(model, X):
    return model.predict(X)


def from_csv(path, target):
    df = pd.read_csv(path)
    return auto(df, target)


def report(model, df):
    check_df_target(df, model.target)
    y = df[model.target]
    X = df.drop(columns=[model.target])
    preds = model.predict(X)
    if model.task == "classification":
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="weighted"))
        }
    else:
        return regressors._regression_report(y, preds, X.shape[1])


def save(model, path):
    if hasattr(model, "save"):
        model.save(path)
    else:
        joblib.dump(model, path)


def load(path):
    return EasyModel.load(path)


def auto(df, target, task="auto"):
    """Automatically pick classification or regression based on target."""
    check_df_target(df, target)
    y = df[target]

    if task == "classification":
        return classify(df, target)
    elif task == "regression":
        return regress(df, target)

    if y.dtype == "object" or y.nunique() < 20:
        return classify(df, target)
    else:
        return regress(df, target)


def creator():
    return "Created by Akash Anipakalu Giridhar"


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

    @staticmethod
    def california_housing():
        data = skdatasets.fetch_california_housing(as_frame=True)
        return data.frame.copy()

    @staticmethod
    def penguins():
        try:
            import seaborn as sns
        except ImportError as exc:
            raise ImportError("Install seaborn to use datasets.penguins(): pip install seaborn") from exc
        df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
        return df

    @staticmethod
    def from_url(url):
        return pd.read_csv(url)
