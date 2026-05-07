"""
🌬️ BreezeML 🔥✨
Beginner-friendly machine learning library built on scikit-learn.
"""
__version__ = "0.2.9"
__author__ = "Akash Anipakalu Giridhar"

from .breezeml import fit, predict, auto, save, load, datasets, creator, classify, regress, from_csv, report
from . import classifiers, clustering, regressors, text, explain, plot

__all__ = [
    "fit","predict","auto","save","load","datasets","creator","classify","regress","from_csv","report",
    "classifiers","clustering","regressors","text","explain","plot"
]
