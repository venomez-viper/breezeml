"""
BreezeML: a beginner-friendly, production-aware ML workflow layer for
students, analysts, and AI agents. Train, compare, explain, export, and
deploy scikit-learn models without drowning in boilerplate.
"""

__version__ = "2.0.0"
__author__ = "Akash Anipakalu Giridhar"

from .breezeml import auto, classify, creator, datasets, fit, from_csv, load, predict, regress, save, Model, EasyModel
from .report import report, Report
from .export import export
from .card import card
from .deploy import deploy
from .automl import automl
from ._fun import fortune, haiku, sensei, zen
from ._guide import guide
from .audit import audit, contamination
from .blend import blend
from . import (
    active,
    anomaly,
    autofeat,
    causal,
    classifiers,
    clustering,
    conformal,
    drift,
    explain,
    fairness,
    features,
    imbalance,
    multi,
    plot,
    recommend,
    regressors,
    semisupervised,
    significance,
    survival,
    text,
    timeseries,
    track,
)

__all__ = [
    "fit",
    "predict",
    "auto",
    "save",
    "load",
    "datasets",
    "creator",
    "classify",
    "regress",
    "from_csv",
    "report",
    "Report",
    "Model",
    "EasyModel",
    "export",
    "card",
    "deploy",
    "automl",
    "classifiers",
    "clustering",
    "regressors",
    "features",
    "text",
    "explain",
    "plot",
    "timeseries",
    "drift",
    "zen",
    "haiku",
    "fortune",
    "sensei",
    "guide",
    "audit",
    "contamination",
    "blend",
    "fairness",
    "imbalance",
    "anomaly",
    "semisupervised",
    "track",
    "significance",
    "multi",
    "recommend",
    "survival",
    "conformal",
    "active",
    "autofeat",
    "causal",
]
