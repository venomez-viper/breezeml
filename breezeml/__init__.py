"""
🌬️ BreezeML 🔥✨
Beginner-friendly machine learning library built on scikit-learn.
"""
__version__ = "0.1.2"
__author__ = "Akash Anipakalu Giridhar 🔥✨"

from .breezeml import fit, predict, auto, save, load, datasets, creator, classify, regress, from_csv, report
from . import classifiers, clustering

__all__ = [
    "fit","predict","auto","save","load","datasets","creator","classify","regress","from_csv","report",
    "classifiers","clustering"
]
