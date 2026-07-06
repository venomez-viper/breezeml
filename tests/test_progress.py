"""Tests for the training progress display (v1.4)."""
import io

import numpy as np
import pandas as pd

from breezeml import classifiers, datasets, timeseries
from breezeml._progress import ProgressBar


class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_bar_tty_single_line():
    stream = _FakeTTY()
    bar = ProgressBar(4, desc="Testing", stream=stream)
    for label in ["a", "b", "c", "d"]:
        bar.update(label)
    bar.close()
    out = stream.getvalue()
    assert out.count("\r") == 5  # 4 updates + close rewrite
    assert "4/4" in out
    assert out.rstrip().endswith("in " + out.rstrip().split("in ")[-1])  # close message present


def test_bar_non_tty_milestone_lines():
    stream = io.StringIO()
    bar = ProgressBar(3, desc="Screening", stream=stream)
    bar.update("logistic")
    bar.update("svm")
    bar.close()
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    assert lines[0].startswith("Screening: 1/3 logistic")
    assert lines[1].startswith("Screening: 2/3 svm")
    assert "2/3" in lines[-1]


def test_bar_disabled_silent():
    stream = _FakeTTY()
    bar = ProgressBar(5, desc="x", stream=stream, enabled=False)
    bar.update("a")
    bar.close()
    assert stream.getvalue() == ""


def test_compare_results_unchanged_with_progress():
    df = datasets.iris()
    quiet = classifiers.compare(df, "species", show=False, progress=False)
    loud = classifiers.compare(df, "species", show=False, progress=True)
    assert [r["classifier"] for r in quiet] == [r["classifier"] for r in loud]


def test_timeseries_compare_progress_smoke():
    rng = np.random.default_rng(0)
    t = np.arange(120)
    df = pd.DataFrame({"y": 50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, 120)})
    results = timeseries.compare(df, "y", show=False, progress=True)
    assert results
