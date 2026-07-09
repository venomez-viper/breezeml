"""Tests for breezeml.causal (v1.9): treatment-effect estimation and uplift,
with loud confounding warnings."""
import numpy as np
import pandas as pd

import breezeml


def _confounded(n=2000, tau=3.0, seed=1):
    """A covariate x drives BOTH treatment assignment and outcome."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-x))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 2 * x + tau * t + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "treat": t, "outcome": y}), tau


def _randomized(n=2000, tau=3.0, seed=2):
    """Treatment assigned by a coin flip, independent of covariates."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = rng.integers(0, 2, n)
    y = 2 * x + tau * t + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "treat": t, "outcome": y}), tau


def test_naive_is_biased_under_confounding():
    df, tau = _confounded()
    naive = breezeml.causal.estimate_ate(df, "treat", "outcome", method="naive", show=False)
    tl = breezeml.causal.estimate_ate(df, "treat", "outcome", method="t_learner", show=False)
    # the adjusted estimate is closer to the truth than the naive one
    assert abs(tl["ate"] - tau) < abs(naive["ate"] - tau)


def test_t_learner_recovers_effect():
    df, tau = _confounded()
    r = breezeml.causal.estimate_ate(df, "treat", "outcome", method="t_learner", show=False)
    assert abs(r["ate"] - tau) < 0.5


def test_ipw_recovers_effect():
    df, tau = _confounded()
    r = breezeml.causal.estimate_ate(df, "treat", "outcome", method="ipw", show=False)
    assert abs(r["ate"] - tau) < 0.8  # looser: IPW is higher variance


def test_check_confounding_flags_imbalance(capsys):
    df, _ = _confounded()
    result = breezeml.causal.check_confounding(df, "treat", "outcome", show=True)
    assert result["n_imbalanced"] >= 1
    assert result["looks_randomized"] is False
    out = capsys.readouterr().out
    assert out.strip()  # a warning was printed


def test_randomized_naive_matches_adjusted():
    df, tau = _randomized()
    naive = breezeml.causal.estimate_ate(df, "treat", "outcome", method="naive", show=False)
    tl = breezeml.causal.estimate_ate(df, "treat", "outcome", method="t_learner", show=False)
    # under randomization the naive estimate is already unbiased
    assert abs(naive["ate"] - tau) < 0.4
    assert abs(naive["ate"] - tl["ate"]) < 0.5
    conf = breezeml.causal.check_confounding(df, "treat", "outcome", show=False)
    assert conf["looks_randomized"] is True


def test_uplift_predicts_per_row():
    df, _ = _confounded()
    model_pair, report = breezeml.causal.uplift(df, "treat", "outcome", show=False)
    preds = model_pair.predict_uplift(df.drop(columns=["treat", "outcome"]).head(25))
    assert len(preds) == 25
    assert "top_decile_uplift" in report or "overall_ate" in report
