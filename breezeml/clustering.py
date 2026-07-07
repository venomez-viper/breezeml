"""
BreezeML Clustering (v0.2.9)
Simple clustering helpers with light preprocessing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MeanShift, OPTICS, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


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


def gaussian_mixture(df: pd.DataFrame, n_clusters: int = 3, covariance_type: str = "full", random_state: int = 42):
    """Soft clustering: also returns per-row membership probabilities."""
    X = _prep_numeric(df)
    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=random_state)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels),
        "probabilities": model.predict_proba(X),
        "bic": float(model.bic(X)),
    }


def birch(df: pd.DataFrame, n_clusters: int = 3, threshold: float = 0.5):
    """Memory-efficient hierarchical clustering for larger datasets."""
    X = _prep_numeric(df)
    model = Birch(n_clusters=n_clusters, threshold=threshold)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels)
    }


def spectral(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """Graph-based clustering; strong on non-convex cluster shapes."""
    X = _prep_numeric(df)
    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, assign_labels="cluster_qr")
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels)
    }


def meanshift(df: pd.DataFrame, bandwidth: float | None = None):
    """Finds the number of clusters itself by seeking density peaks."""
    X = _prep_numeric(df)
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "n_clusters_found": int(len(set(labels))),
        "silhouette": _silhouette_safe(X, labels)
    }


def optics(df: pd.DataFrame, min_samples: int = 5):
    """DBSCAN's flexible sibling: handles clusters of varying density."""
    X = _prep_numeric(df)
    model = OPTICS(min_samples=min_samples)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "silhouette": _silhouette_safe(X, labels)
    }


def hdbscan(df: pd.DataFrame, min_cluster_size: int = 5):
    """Hierarchical DBSCAN (sklearn >= 1.3): robust density clustering with noise labels."""
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError("hdbscan requires scikit-learn >= 1.3.") from exc
    X = _prep_numeric(df)
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "n_noise": int((labels == -1).sum()),
        "silhouette": _silhouette_safe(X, labels)
    }
