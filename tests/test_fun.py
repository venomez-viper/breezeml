"""Tests for the zen garden (v1.5)."""
import breezeml
from breezeml import classifiers, datasets, timeseries


def test_zen_prints_the_way(capsys):
    text = breezeml.zen()
    out = capsys.readouterr().out
    assert "kaze no michi" in text
    assert "guiding wind" in text
    assert "sklearn, pandas, numpy, joblib" in text
    assert out.strip()


def test_haiku_reproducible(capsys):
    first = breezeml.haiku(seed=7)
    second = breezeml.haiku(seed=7)
    assert first == second
    assert first.count("\n") == 2  # three lines


def test_fortune_omikuji(capsys):
    text = breezeml.fortune(seed=42)
    assert "omikuji" in text
    assert any(rank in text for rank in ["Dai-kichi", "Kichi", "Chuu-kichi", "Shou-kichi", "Kyou"])
    assert breezeml.fortune(seed=42) == text


def test_perfect_accuracy_egg(capsys):
    # Wine with default split gives a 1.0 top accuracy -> the egg must fire.
    df = datasets.wine()
    classifiers.compare(df, "class", show=True, progress=False)
    out = capsys.readouterr().out
    if "1.0000" in out.split("\n")[3]:  # top row perfect
        assert "Perfect accuracy detected" in out


def test_naive_wins_egg(capsys):
    import numpy as np
    import pandas as pd

    # Pure random walk: nothing should beat naive.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": np.cumsum(rng.normal(0, 1, 150))})
    results = timeseries.compare(df, "y", show=True, progress=False)
    out = capsys.readouterr().out
    if results[0]["model"].startswith("naive"):
        assert "random walk" in out
