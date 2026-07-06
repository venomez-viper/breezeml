"""Tests for breezeml.guide() (v1.6)."""
import breezeml


def test_guide_prints_the_path(capsys):
    text = breezeml.guide()
    out = capsys.readouterr().out
    assert "BREATH 1" in text
    assert "BREATH 4" in text
    assert "fit(df" in text or "fit(" in text
    assert "export" in text
    assert "4 dependencies" in text
    assert out.strip()


def test_guide_mentions_every_layer_entrypoint():
    text = breezeml.guide()
    for name in ["fit", "predict", "compare", "quick_tune", "card", "automl",
                 "export", "deploy", "drift", "timeseries", "zen"]:
        assert name in text, f"guide is missing '{name}'"
