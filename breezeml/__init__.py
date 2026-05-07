"""
BreezeML public API.
"""

__version__ = "0.3.0"
__author__ = "Akash Anipakalu Giridhar"

from .breezeml import auto, classify, creator, datasets, fit, from_csv, load, predict, regress, report, save
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
    "classifiers",
    "clustering",
    "regressors",
    "features",
    "text",
    "explain",
    "plot",
]
