"""
BreezeML Clustering (v0.1.2)
Simple clustering helpers with light preprocessing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def _prep_numeric(df: pd.DataFrame) -> np.ndarray:
    # Use numeric columns; if none, one-hot everything first.
    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        df = pd.get_dummies(df, drop_first=False)
    else:
        df = df.select_dtypes(include=[np.number])

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return pipe.fit_transform(df)


def _silhouette_safe(X: np.ndarray, labels) -> float | None:
    try:
        if len(set(labels)) >= 2 and len(set(labels)) < len(labels):
            return float(silhouette_score(X, labels))
    except Exception:
        pass
    return None


def kmeans(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    X = _prep_numeric(df)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels),
        "centers": getattr(model, "cluster_centers_", None)
    }


def agglomerative(df: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward"):
    X = _prep_numeric(df)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels)
    }


def dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    X = _prep_numeric(df)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels)
    }
