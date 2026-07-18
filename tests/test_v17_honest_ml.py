"""Tests for v1.7.0 "The Honest Machine": audit, fairness, imbalance,
blend, track, anomaly, semisupervised, native explain, CLI, zoo wave 2."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import breezeml
from breezeml import (
    anomaly,
    classifiers,
    clustering,
    datasets,
    drift,  # noqa: F401  (import sanity)
    fairness,
    imbalance,
    regressors,
    semisupervised,
    track,
)
from breezeml.explain import partial_dependence, permutation_importance


# ------------------------------------------------------------------ audit

def test_audit_clean_data():
    df = datasets.iris()
    result = breezeml.audit(df, "species", show=False)
    assert result["ok"] is True
    assert result["task"] == "classification"


def test_audit_catches_id_and_leakage():
    rng = np.random.default_rng(42)
    n = 300
    y = rng.choice(["a", "b"], n)
    df = pd.DataFrame({
        "customer_id": np.arange(n),                 # ID column
        "feature": rng.normal(0, 1, n),
        "leaky": (y == "a").astype(int) + rng.normal(0, 0.01, n),  # target leak
        "constant": 1,
        "label": y,
    })
    result = breezeml.audit(df, "label", show=False)
    assert result["ok"] is False
    cats = {f["category"] for f in result["findings"]}
    assert "id_columns" in cats
    assert "target_leakage" in cats
    assert "constant_columns" in cats
    assert any(leak["column"] == "leaky" for leak in result["leak_details"])


def test_audit_catches_regression_target_copy():
    # a copy of a continuous target must be flagged; the shallow-tree probe
    # alone misses this (staircase R2 < threshold), the rank-correlation
    # fast path catches it (found by benchmarks/validation_study.py)
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(50, 10, n)
    df = pd.DataFrame({
        "feature": rng.normal(0, 1, n),
        "leaky_copy": y,
        "leaky_monotone": np.exp(y / 20),  # monotone transform of target
        "target": y,
    })
    result = breezeml.audit(df, "target", show=False)
    cats = {f["category"] for f in result["findings"]}
    assert "target_leakage" in cats
    leaked = {leak["column"] for leak in result["leak_details"]}
    assert "leaky_copy" in leaked
    assert "leaky_monotone" in leaked


def test_audit_catches_many_class_target_copy():
    # 10+ classes exceed a depth-3 tree's 8 leaves; probe depth must scale
    # with class count (found by benchmarks/validation_study.py)
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 10, n)
    df = pd.DataFrame({
        "feature": rng.normal(0, 1, n),
        "leaky_copy": y,
        "target": y.astype(str),
    })
    result = breezeml.audit(df, "target", show=False)
    cats = {f["category"] for f in result["findings"]}
    assert "target_leakage" in cats


def test_contamination_detects_overlap():
    df = datasets.iris()
    train, test = df.iloc[:120], df.iloc[100:]  # rows 100-119 shared
    result = breezeml.contamination(train, test, show=False)
    assert result["contaminated"] is True
    assert result["shared_rows"] >= 15
    clean = breezeml.contamination(df.iloc[:100], df.iloc[100:], show=False)
    assert clean["contaminated"] is False


# --------------------------------------------------------------- fairness

def _fair_df(n=400, biased=False, seed=42):
    rng = np.random.default_rng(seed)
    group = rng.choice(["m", "f"], n)
    x = rng.normal(0, 1, n)
    if biased:
        y = np.where((group == "m") & (x > -0.5), "yes", "no")
    else:
        y = np.where(x > 0, "yes", "no")
    return pd.DataFrame({"x": x, "group": group, "outcome": y})


def test_fairness_report_structure():
    df = _fair_df()
    model, _ = breezeml.auto(df.drop(columns=["group"]).assign(group=df["group"]), "outcome")
    result = fairness.report(model, df, sensitive="group", show=False)
    assert set(result["groups"].keys()) == {"m", "f"}
    for g in result["groups"].values():
        assert "accuracy" in g and "selection_rate" in g
    assert "demographic_parity_ratio" in result


def test_fairness_flags_biased_model():
    df = _fair_df(biased=True)
    model, _ = breezeml.auto(df, "outcome")
    result = fairness.report(model, df, sensitive="group", show=False)
    assert result["passes_four_fifths"] is False
    assert any("four-fifths" in n for n in result["notes"])


# -------------------------------------------------------------- imbalance

@pytest.fixture(scope="module")
def imbalanced_df():
    rng = np.random.default_rng(42)
    n = 600
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = np.where(x1 + x2 + rng.normal(0, 0.6, n) > 2.2, "rare", "common")
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_imbalance_summary(imbalanced_df):
    result = imbalance.summary(imbalanced_df["y"], show=False)
    assert result["imbalance_ratio"] > 3
    assert result["minority_class"] == "rare"


def test_balanced_training_param(imbalanced_df):
    model, report = breezeml.classify(imbalanced_df, "y", balanced=True)
    assert model.pipeline.named_steps["model"].class_weight == "balanced"
    assert report["accuracy"] > 0.5


def test_tune_threshold(imbalanced_df):
    model, _ = breezeml.classify(imbalanced_df, "y")
    result = imbalance.tune_threshold(model, imbalanced_df, "y", show=False)
    assert 0 < result["best_threshold"] < 1
    assert result["f1_at_best"] >= result["f1_at_default"]


def test_calibrate(imbalanced_df):
    model, _ = breezeml.classify(imbalanced_df, "y")
    calibrated, report = imbalance.calibrate(model, imbalanced_df, "y", show=False)
    assert hasattr(calibrated, "predict_proba")
    assert report["brier_before"] >= 0 and report["brier_after"] >= 0


def test_cost_report(imbalanced_df):
    model, _ = breezeml.classify(imbalanced_df, "y")
    result = imbalance.cost_report(model, imbalanced_df, "y", fp_cost=1, fn_cost=25, show=False)
    assert result["cost_at_best"] <= result["cost_at_default"]
    # expensive misses push the threshold DOWN (catch more positives)
    assert result["best_threshold"] <= 0.5


# ------------------------------------------------------------------ blend

def test_blend_vote():
    df = datasets.wine()
    model, report = breezeml.blend(df, "class", top_k=3, show=False)
    assert report["blend_score"] > 0.7
    assert len(report["members"]) >= 2
    assert model.meta["blend"]["method"] == "vote"
    # v1.0 features still work on blends
    assert "Model Card" in breezeml.card(model)


# ------------------------------------------------------------------ track

def test_track_log_and_leaderboard(tmp_path):
    df = datasets.iris()
    model, report = breezeml.auto(df, "species")
    entry = track.log(model, report, name="baseline", directory=str(tmp_path), show=False)
    assert entry["id"] == 1
    track.log(model, {"accuracy": 0.5}, name="worse", directory=str(tmp_path), show=False)
    ranked = track.leaderboard(directory=str(tmp_path), show=False)
    assert len(ranked) == 2
    assert ranked[0]["name"] == "baseline"
    assert track.best(directory=str(tmp_path))["name"] == "baseline"
    assert track.clear(directory=str(tmp_path)) == 2


# ---------------------------------------------------------------- anomaly

def test_anomaly_detectors_and_consensus():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (200, 3))
    # scattered anomalies (a tight far cluster would fool boundary methods
    # like One-Class SVM into calling it a second normal region)
    directions = rng.normal(0, 1, (10, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    outliers = directions * rng.uniform(6, 10, (10, 1))
    df = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])
    result = anomaly.compare(df, contamination=0.06, show=False, progress=False)
    planted = set(range(200, 210))
    # majority vote (>= 3 of 4 detectors) should recover the planted outliers
    majority_found = planted & set(result["majority_indices"])
    assert len(majority_found) >= 8, f"majority missed planted outliers: {sorted(majority_found)}"
    # isolation forest alone should nail all of them
    iso = result["detectors"]["isolation_forest"]
    assert len(planted & set(iso["anomaly_indices"])) >= 9
    assert result["n_unanimous"] >= 1


# --------------------------------------------------------- semisupervised

def test_self_train_runs_and_reports_baseline():
    df = datasets.breast_cancer()
    df = df.sample(400, random_state=42).reset_index(drop=True)
    df.loc[80:, "label"] = np.nan  # 80 labeled, rest unlabeled
    model, report = semisupervised.self_train(df, "label", show=False)
    assert report["n_labeled"] == 80
    assert report["n_unlabeled"] == 320
    assert "supervised_baseline_accuracy" in report
    assert report["accuracy"] > 0.6


def test_self_train_requires_unlabeled():
    df = datasets.iris()
    with pytest.raises(ValueError, match="unlabeled"):
        semisupervised.self_train(df, "species", show=False)


# ---------------------------------------------------------- explain native

def test_permutation_importance_core_deps():
    df = datasets.iris()
    model, _ = breezeml.auto(df, "species")
    rows = permutation_importance(model, df, "species", n_repeats=3, show=False)
    assert len(rows) == 4
    assert rows[0]["importance_mean"] >= rows[-1]["importance_mean"]
    # petal features dominate iris
    assert "petal" in str(rows[0]["feature"])


def test_partial_dependence_core_deps():
    df = datasets.diabetes().sample(150, random_state=42)
    model, _ = breezeml.auto(df, "target")
    result = partial_dependence(model, df, "target", feature="bmi")
    assert len(result["grid"]) == len(result["average_prediction"])
    assert len(result["grid"]) > 5


# -------------------------------------------------------------------- cli

def test_cli_train_and_audit(tmp_path):
    csv = tmp_path / "iris.csv"
    datasets.iris().to_csv(csv, index=False)
    out = tmp_path / "model.joblib"
    result = subprocess.run(
        [sys.executable, "-m", "breezeml.cli", "train", str(csv), "--target", "species", "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    result = subprocess.run(
        [sys.executable, "-m", "breezeml.cli", "audit", str(csv), "--target", "species"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Data Audit" in result.stdout


# ------------------------------------------------------------- zoo wave 2

NEW_CLASSIFIERS_V17 = ["bernoulli_nb", "passive_aggressive", "nearest_centroid", "bagging"]
NEW_REGRESSORS_V17 = ["poisson", "quantile", "theilsen", "ransac", "kernel_ridge", "bagging"]


@pytest.mark.parametrize("algo", NEW_CLASSIFIERS_V17)
def test_zoo2_classifier(algo):
    df = datasets.iris()
    _, report = getattr(classifiers, algo)(df, "species")
    assert report["accuracy"] > 0.5, f"{algo}: {report}"


@pytest.mark.parametrize("algo", NEW_REGRESSORS_V17)
def test_zoo2_regressor(algo):
    df = datasets.diabetes().sample(150, random_state=42)
    _, report = getattr(regressors, algo)(df, "target")
    assert report["r2"] is not None, f"{algo}: {report}"


def test_zoo2_clustering():
    df = datasets.iris().drop(columns=["species"])
    for fn in (clustering.meanshift, clustering.optics):
        result = fn(df)
        assert len(result["labels"]) == len(df)
    try:
        result = clustering.hdbscan(df)
        assert len(result["labels"]) == len(df)
    except ImportError:
        pass  # sklearn < 1.3


def test_leaderboards_grew():
    assert len(classifiers._base_classifier_factories()) >= 22
    assert len(regressors._base_regressor_factories()) >= 20
