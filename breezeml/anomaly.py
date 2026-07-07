"""
Anomaly detection: find the rows that do not belong.

    from breezeml import anomaly

    result = anomaly.isolation_forest(df)       # scores + flags per row
    results = anomaly.compare(df)               # all detectors, agreement report

Detectors run on the numeric view of your data with median imputation and
scaling, mirroring the supervised pipeline. Because anomaly detection has
no ground truth, ``compare()`` reports *agreement between detectors* -
rows flagged by all of them deserve your attention first.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from ._progress import ProgressBar

__all__ = ["isolation_forest", "local_outlier_factor", "one_class_svm", "elliptic_envelope", "compare"]


def _prep(df: pd.DataFrame) -> np.ndarray:
    from .clustering import _prep_numeric

    return _prep_numeric(df)


def _package(name, labels, scores, df):
    flags = labels == -1
    return {
        "detector": name,
        "labels": labels,
        "is_anomaly": flags,
        "scores": scores,
        "n_anomalies": int(flags.sum()),
        "anomaly_rate": round(float(flags.mean()), 4),
        "anomaly_indices": df.index[flags].tolist(),
    }


def isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> dict:
    """Tree-isolation based detector; the strong default for tabular data."""
    X = _prep(df)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    return {**_package("isolation_forest", labels, model.score_samples(X), df), "model": model}


def local_outlier_factor(df: pd.DataFrame, contamination: float = 0.05, n_neighbors: int = 20) -> dict:
    """Density-based: flags rows whose neighborhood is much denser than they are."""
    X = _prep(df)
    model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    labels = model.fit_predict(X)
    return {**_package("local_outlier_factor", labels, model.negative_outlier_factor_, df), "model": model}


def one_class_svm(df: pd.DataFrame, nu: float = 0.05) -> dict:
    """Boundary-based: learns the envelope of normal data. Slower on large data."""
    X = _prep(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OneClassSVM(nu=nu)
        labels = model.fit_predict(X)
    return {**_package("one_class_svm", labels, model.score_samples(X), df), "model": model}


def elliptic_envelope(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> dict:
    """Gaussian assumption: fast, great when features are roughly normal."""
    X = _prep(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EllipticEnvelope(contamination=contamination, random_state=random_state)
        labels = model.fit_predict(X)
    return {**_package("elliptic_envelope", labels, model.score_samples(X), df), "model": model}


def compare(df: pd.DataFrame, contamination: float = 0.05, show: bool = True, progress: bool | None = None) -> dict:
    """Run every detector and report where they agree.

    There is no accuracy score in unsupervised anomaly detection - but when
    four very different algorithms all flag the same row, that row is worth
    a human look. ``consensus`` counts detector votes per row.
    """
    if progress is None:
        progress = show
    detectors = {
        "isolation_forest": lambda: isolation_forest(df, contamination),
        "local_outlier_factor": lambda: local_outlier_factor(df, contamination),
        "one_class_svm": lambda: one_class_svm(df, nu=contamination),
        "elliptic_envelope": lambda: elliptic_envelope(df, contamination),
    }
    bar = ProgressBar(len(detectors), desc="Running detectors", enabled=progress)
    results = {}
    votes = np.zeros(len(df), dtype=int)
    for name, run in detectors.items():
        bar.update(name)
        try:
            result = run()
            results[name] = result
            votes += result["is_anomaly"].astype(int)
        except Exception as exc:
            results[name] = {"detector": name, "error": str(exc)[:120]}
    bar.close()

    n_working = len([r for r in results.values() if "error" not in r])
    consensus = pd.Series(votes, index=df.index, name="detector_votes")
    unanimous = consensus[consensus == n_working]
    majority = consensus[consensus >= max((n_working // 2) + 1, 2)]
    report = {
        "detectors": results,
        "consensus_votes": consensus,
        "unanimous_indices": unanimous.index.tolist(),
        "n_unanimous": int(len(unanimous)),
        "majority_indices": majority.index.tolist(),
        "n_majority": int(len(majority)),
    }
    if show:
        print(f"\nBreezeML Anomaly Consensus ({len(df)} rows, contamination={contamination})")
        print("-" * 56)
        for name, r in results.items():
            line = f"  {name:<24}"
            line += f"{r['n_anomalies']} flagged ({r['anomaly_rate']:.1%})" if "error" not in r else f"FAILED: {r['error']}"
            print(line)
        print(f"  {'MAJORITY (most agree)':<24}{report['n_majority']} rows")
        print(f"  {'UNANIMOUS (all agree)':<24}{report['n_unanimous']} rows")
        print("-" * 56)
        print("  Rows flagged by every detector deserve human review first.\n")
    return report
