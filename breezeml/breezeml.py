"""
BreezeML: a beginner-friendly, production-aware ML workflow layer for
students, analysts, and AI agents. Train, compare, explain, export, and
deploy scikit-learn models without drowning in boilerplate.

Created by Akash Anipakalu Giridhar
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from ._validation import check_df_target
from ._preprocessing import _detect_types, _build_preprocessor
from ._meta import build_meta
from ._narrate import narrate, format_narration
from . import regressors

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn import datasets as skdatasets

__all__ = [
    "classify", "regress", "fit", "predict", "from_csv",
    "report", "save", "load", "auto", "creator", "datasets", "Model", "EasyModel"
]


class EasyModel:
    def __init__(self, pipeline, target, task, meta=None):
        self.pipeline = pipeline
        self.target = target
        self.task = task
        self.meta = meta

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, df, target=None):
        """Cross-validated performance of this model on ``df`` (metrics dict)."""
        from .report import _performance
        tgt = target or self.target
        X = df.drop(columns=[tgt])
        y = df[tgt]
        return _performance(self, X, y, self.task)["metrics"]

    def report(self, df, target=None, sensitive=None, show=True):
        """Run the full honesty gauntlet on this model - one SHIP/WARN/STOP report."""
        from .report import report as _report
        return _report(self, df, target=target, sensitive=sensitive, show=show)

    def explain(self, df, target=None, show=True):
        """Plain-English feature importances via permutation importance."""
        from .explain import permutation_importance as _pi
        return _pi(self, df, target or self.target, show=show)

    def predict_interval(self, X, calib_df, alpha=0.1):
        """Conformal prediction intervals (regression). ``calib_df`` is a held-out
        calibration set - required, because honest coverage needs unseen data."""
        if self.task != "regression":
            raise ValueError("predict_interval() is for regression; use predict_set() for classification.")
        from .conformal import conformal_regressor
        cp = conformal_regressor(self, calib_df, self.target, alpha=alpha)
        return cp.predict_interval(X)

    def predict_set(self, X, calib_df, alpha=0.1):
        """Conformal prediction sets (classification): label sets that cover the
        true class at >= 1 - alpha. ``calib_df`` is a held-out calibration set."""
        if self.task != "classification":
            raise ValueError("predict_set() is for classification; use predict_interval() for regression.")
        from .conformal import conformal_classifier
        cp = conformal_classifier(self, calib_df, self.target, alpha=alpha)
        return cp.predict_set(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def export(self, path="train.py", data_path="YOUR_DATA.csv"):
        """Export this model as a standalone scikit-learn script (no breezeml imports)."""
        from .export import export as _export
        return _export(self, path, data_path=data_path)

    def card(self, path=None):
        """Generate a markdown model card. Optionally write it to ``path``."""
        from .card import card as _card
        return _card(self, path)

    def deploy(self, out_dir="deployment", name="breezeml-model"):
        """Write a FastAPI + Docker serving directory for this model."""
        from .deploy import deploy as _deploy
        return _deploy(self, out_dir, name)

    def check_drift(self, new_df, threshold=0.25):
        """Compare new data against this model's training distributions."""
        from .drift import check as _check
        return _check(self, new_df, threshold=threshold)

    def explain_decisions(self):
        """Print a plain-English explanation of every pipeline decision."""
        if not self.meta:
            raise ValueError("No training metadata on this model (train with v0.4+ core API).")
        print(format_narration(narrate(self.meta)))


# Public, stable name for the model object (2.0). EasyModel stays as an alias
# so existing pickles and imports keep working.
Model = EasyModel


def classify(df, target, algo="forest", return_report=True, explain_decisions=False, balanced=False):
    check_df_target(df, target)
    X = df.drop(columns=[target])
    y = df[target]
    numeric, categorical = _detect_types(df, target)
    pre = _build_preprocessor(numeric, categorical)

    class_weight = "balanced" if balanced else None
    model = (
        RandomForestClassifier(random_state=42, class_weight=class_weight)
        if algo == "forest"
        else LogisticRegression(max_iter=200, class_weight=class_weight)
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    eval_report = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="weighted"))
    }

    meta = build_meta(
        df, target, "classification", model,
        task_reason=_task_reason(y, "classification"),
        stratified=stratify is not None,
        report=eval_report,
    )
    em = EasyModel(pipe, target, "classification", meta=meta)

    if explain_decisions:
        print(format_narration(narrate(meta)))

    if not return_report:
        return em
    return em, eval_report


def regress(df, target, algo="forest", return_report=True, explain_decisions=False):
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
    estimator = pipe.named_steps.get("model", pipe.steps[-1][1])
    meta = build_meta(
        df, target, "regression", estimator,
        task_reason=_task_reason(df[target], "regression"),
        stratified=False,
        report=reg_report,
    )
    em = EasyModel(pipe, target, "regression", meta=meta)

    if explain_decisions:
        print(format_narration(narrate(meta)))

    if not return_report:
        return em
    return em, reg_report


def _task_reason(y, task):
    """Human-readable reason for the detected task type."""
    if task == "classification":
        if y.dtype == "object":
            return f"target '{y.name}' contains text labels, so this is classification."
        return (
            f"target '{y.name}' has only {y.nunique()} distinct values "
            "(fewer than 20), so BreezeML treats it as classification."
        )
    return (
        f"target '{y.name}' is numeric with {y.nunique()} distinct values, "
        "so BreezeML treats it as regression."
    )


def fit(df: pd.DataFrame, target: str, task: str = "auto") -> EasyModel:
    """Train a model and return one unified :class:`Model` (the 2.0 entry point)."""
    m, _ = auto(df, target, task=task)
    return m


def predict(model: EasyModel, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def from_csv(path: str, target: str) -> tuple[EasyModel, dict]:
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


def auto(df: pd.DataFrame, target: str, task: str = "auto",
         explain_decisions: bool = False, balanced: bool = False):
    """Automatically pick classification or regression based on target."""
    check_df_target(df, target)
    y = df[target]

    if task == "classification":
        return classify(df, target, explain_decisions=explain_decisions, balanced=balanced)
    elif task == "regression":
        return regress(df, target, explain_decisions=explain_decisions)

    if y.dtype == "object" or y.nunique() < 20:
        return classify(df, target, explain_decisions=explain_decisions, balanced=balanced)
    else:
        return regress(df, target, explain_decisions=explain_decisions)


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
