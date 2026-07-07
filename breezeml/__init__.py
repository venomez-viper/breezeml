"""
BreezeML: a beginner-friendly, production-aware ML workflow layer for
students, analysts, and AI agents. Train, compare, explain, export, and
deploy scikit-learn models without drowning in boilerplate.
"""

__version__ = "1.7.0"
__author__ = "Akash Anipakalu Giridhar"

from .breezeml import auto, classify, creator, datasets, fit, from_csv, load, predict, regress, report, save
from .export import export
from .card import card
from .deploy import deploy
from .automl import automl
from ._fun import fortune, haiku, sensei, zen
from ._guide import guide
from .audit import audit, contamination
from .blend import blend
from . import (
    anomaly,
    classifiers,
    clustering,
    drift,
    explain,
    fairness,
    features,
    imbalance,
    plot,
    regressors,
    semisupervised,
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
]
