"""Tests for breezeml.multi (multi-label classification, multi-output regression)."""
import numpy as np
import pandas as pd
import pytest

from breezeml import multi


def _make_multilabel_df(n=400, seed=0):
    """Three correlated binary labels derived from a few numeric features."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)

    label_a = (f1 + 0.5 * f2 + noise > 0).astype(int)
    # label_b correlates with label_a (shares f1) plus its own signal.
    label_b = (0.8 * f1 + f3 + noise > 0).astype(int)
    # label_c correlates with both a and b.
    label_c = (f2 - f3 + 0.5 * f1 + noise > 0).astype(int)

    return pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "label_a": label_a,
            "label_b": label_b,
            "label_c": label_c,
        }
    )


def _make_multioutput_df(n=400, seed=1):
    """Two numeric targets that are (noisy) functions of the features."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)

    y1 = 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(scale=0.2, size=n)
    y2 = -1.5 * x1 + 0.7 * x3 + rng.normal(scale=0.2, size=n)

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})


TARGETS_ML = ["label_a", "label_b", "label_c"]
TARGETS_MO = ["y1", "y2"]


def test_multi_label_reports_per_target_and_predict_shape():
    df = _make_multilabel_df()
    model, report = multi.multi_label(df, TARGETS_ML, show=False)

    # per-target metrics for all three labels
    assert set(report["per_target"].keys()) == set(TARGETS_ML)
    for name in TARGETS_ML:
        row = report["per_target"][name]
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0

    assert 0.0 <= report["subset_accuracy"] <= 1.0
    assert 0.0 <= report["hamming_loss"] <= 1.0

    # predict returns an array of shape (n, 3)
    X = df.drop(columns=TARGETS_ML)
    preds = np.asarray(model.predict(X))
    assert preds.shape == (len(df), 3)

    # model wiring
    assert model.task == "multi_label"
    assert model.targets == TARGETS_ML
    assert model.meta is not None
    assert model.meta["targets"] == TARGETS_ML
    assert "multi-label" in model.meta["task_reason"]


def test_multi_label_chain_path():
    df = _make_multilabel_df()
    model, report = multi.multi_label(df, TARGETS_ML, chain=True, show=False)

    # same report keys as the non-chain path
    assert set(report.keys()) == {"per_target", "subset_accuracy", "hamming_loss"}
    assert set(report["per_target"].keys()) == set(TARGETS_ML)
    assert 0.0 <= report["subset_accuracy"] <= 1.0
    assert 0.0 <= report["hamming_loss"] <= 1.0

    preds = np.asarray(model.predict(df.drop(columns=TARGETS_ML)))
    assert preds.shape == (len(df), 3)


def test_multi_output_reports_per_target_r2_and_average():
    df = _make_multioutput_df()
    model, report = multi.multi_output(df, TARGETS_MO, show=False)

    assert set(report["per_target"].keys()) == set(TARGETS_MO)
    for name in TARGETS_MO:
        assert report["per_target"][name]["r2"] is not None

    assert "average_r2" in report
    assert report["average_r2"] is not None
    assert report["average_r2"] > 0

    preds = np.asarray(model.predict(df.drop(columns=TARGETS_MO)))
    assert preds.shape == (len(df), 2)

    assert model.task == "multi_output"
    assert model.targets == TARGETS_MO
    assert "multi-output" in model.meta["task_reason"]


def test_missing_target_raises_value_error():
    df = _make_multilabel_df()
    with pytest.raises(ValueError):
        multi.multi_label(df, ["label_a", "not_a_column"], show=False)

    df2 = _make_multioutput_df()
    with pytest.raises(ValueError):
        multi.multi_output(df2, ["y1", "nope"], show=False)


def test_valid_targets_no_weird_overlap():
    """A normal, fully valid target list trains without complaint."""
    df = _make_multilabel_df()
    model, report = multi.multi_label(df, TARGETS_ML, show=False)
    # features are exactly the non-target columns; no overlap issues.
    assert model.meta["profile"]["n_features"] == 3


def test_duplicate_targets_raise():
    df = _make_multilabel_df()
    with pytest.raises(ValueError):
        multi.multi_label(df, ["label_a", "label_a"], show=False)


def test_string_targets_rejected():
    df = _make_multilabel_df()
    with pytest.raises(TypeError):
        multi.multi_label(df, "label_a", show=False)
