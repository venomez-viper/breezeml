"""
BreezeML: a beginner-friendly, production-aware ML workflow layer for
students, analysts, and AI agents. Train, compare, explain, export, and
deploy scikit-learn models without drowning in boilerplate.
"""

__version__ = "1.0.0"
__author__ = "Akash Anipakalu Giridhar"

from .breezeml import auto, classify, creator, datasets, fit, from_csv, load, predict, regress, report, save
from .export import export
from .card import card
from .deploy import deploy
from . import classifiers, clustering, explain, features, plot, regressors, text

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
    "classifiers",
    "clustering",
    "regressors",
    "features",
    "text",
    "explain",
    "plot",
]
