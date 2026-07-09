"""Tests for breezeml.autofeat (v1.9): automatic, leakage-safe feature
engineering that reports every transformation."""
import numpy as np
import pandas as pd

import breezeml


def _raw_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "city": rng.choice([f"c{i}" for i in range(100)], n),  # high cardinality
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "const": 5,  # constant
        "y": rng.integers(0, 2, n),
    })
    df["x1_dup"] = df["x1"] + rng.normal(0, 0.001, n)  # r > 0.99
    return df


def test_engineer_does_not_mutate_input():
    df = _raw_df()
    before_cols = list(df.columns)
    before_shape = df.shape
    breezeml.autofeat.engineer(df, "y", show=False)
    assert list(df.columns) == before_cols
    assert df.shape == before_shape


def test_engineer_datetime_expanded_and_dropped():
    df = _raw_df()
    new, rep = breezeml.autofeat.engineer(df, "y", show=False)
    assert "ts" not in new.columns  # raw datetime dropped
    expanded = [c for c in new.columns if c.startswith("ts_")]
    assert len(expanded) >= 3  # year/month/dayofweek/hour/is_weekend
    assert rep["datetime_expanded"]  # something was expanded


def test_engineer_encodes_high_cardinality():
    df = _raw_df()
    new, rep = breezeml.autofeat.engineer(df, "y", show=False)
    # some frequency/target encoding column derived from city
    assert any("city" in str(c) for c in new.columns) or rep["encoded"]
    assert rep["encoded"]


def test_engineer_prunes_constant_and_correlated():
    df = _raw_df()
    new, rep = breezeml.autofeat.engineer(df, "y", show=False)
    assert "const" not in new.columns
    # one of the near-duplicate pair dropped
    assert not ("x1" in new.columns and "x1_dup" in new.columns)
    assert rep["dropped"]


def test_engineer_target_preserved_and_counts_consistent():
    df = _raw_df()
    new, rep = breezeml.autofeat.engineer(df, "y", show=False)
    assert "y" in new.columns
    assert rep["n_features_after"] == new.shape[1] - 1  # minus target


def test_engineer_output_is_model_ready():
    df = _raw_df()
    new, _ = breezeml.autofeat.engineer(df, "y", show=False)
    model, report = breezeml.auto(new, "y")
    assert report["accuracy"] >= 0.0  # trains without error


def test_target_encoding_is_leakage_safe():
    # a random target should NOT become perfectly predictable via encoding
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "cat": rng.choice([f"k{i}" for i in range(80)], n),
        "num": rng.normal(0, 1, n),
        "y": rng.integers(0, 2, n),
    })
    new, _ = breezeml.autofeat.engineer(df, "y", show=False)
    enc_cols = [c for c in new.columns if "target" in str(c).lower() or "te_" in str(c).lower()]
    for c in enc_cols:
        corr = abs(np.corrcoef(new[c].astype(float), new["y"].astype(float))[0, 1])
        assert corr < 0.95, f"{c} looks leaky (corr={corr})"
