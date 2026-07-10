"""Tests for the 2.0 flagship: breezeml.report() honest scorecard."""
import warnings

import breezeml
from breezeml.report import Report

warnings.filterwarnings("ignore")


def test_clean_model_ships():
    df = breezeml.datasets.iris()
    model = breezeml.fit(df, "species")
    rep = breezeml.report(model, df, show=False)
    assert isinstance(rep, Report)
    assert rep.verdict == "SHIP"
    assert rep.ok is True
    assert rep.sections["performance"]["beats_baseline"] is True
    assert "accuracy" in rep.sections["performance"]["metrics"]


def test_leakage_stops():
    df = breezeml.datasets.iris()
    df["leak"] = df["species"]  # a column identical to the target
    model = breezeml.fit(df, "species")
    rep = breezeml.report(model, df, show=False)
    assert rep.verdict == "STOP"
    assert any(r["category"] == "audit" for r in rep.reasons)


def test_no_signal_stops():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200),
                       "y": rng.integers(0, 2, size=200)})  # target is pure noise
    model = breezeml.fit(df, "y")
    rep = breezeml.report(model, df, show=False)
    assert rep.verdict == "STOP"
    assert any(r["category"] == "no_signal" for r in rep.reasons)


def test_serialisation_roundtrip():
    df = breezeml.datasets.iris()
    model = breezeml.fit(df, "species")
    rep = breezeml.report(model, df, show=False)
    d = rep.to_dict()
    assert d["verdict"] in {"SHIP", "WARN", "STOP"}
    assert rep.to_json().startswith("{")
    assert rep.to_markdown().startswith("# BreezeML honest report")


def test_model_report_method():
    df = breezeml.datasets.iris()
    model = breezeml.fit(df, "species")
    rep = model.report(df, show=False)
    assert isinstance(rep, Report)
    metrics = model.evaluate(df)
    assert "accuracy" in metrics
