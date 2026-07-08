"""Tests for breezeml.survival (Kaplan-Meier + log-rank)."""
import numpy as np
import pandas as pd
import pytest

from breezeml import survival


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def _exp_survival(n=500, scale=10.0, censor_frac=0.30, seed=0):
    """Exponential event times with an independent censoring time.

    A subject is censored when its (independent) censoring time comes first;
    the censoring time scale is tuned so roughly ``censor_frac`` of subjects
    end up censored.
    """
    rng = np.random.default_rng(seed)
    event_time = rng.exponential(scale, n)
    # choose censoring scale so that P(censor before event) ~= censor_frac.
    # for two independent exponentials, P(C < T) = rate_c / (rate_c + rate_t).
    # solve censor_frac = (1/cs) / (1/cs + 1/scale) -> cs = scale*(1-f)/f
    cs = scale * (1.0 - censor_frac) / censor_frac
    censor_time = rng.exponential(cs, n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"duration": duration, "event": event})


# --------------------------------------------------------------------------- #
# kaplan_meier basics
# --------------------------------------------------------------------------- #
def test_km_monotone_and_starts_at_one():
    df = _exp_survival()
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    surv = np.asarray(res["survival"])
    assert surv[0] == pytest.approx(1.0)
    # survival curve must be non-increasing
    assert np.all(np.diff(surv) <= 1e-12)
    # timeline is sorted ascending and starts at 0
    tl = np.asarray(res["timeline"])
    assert tl[0] == 0.0
    assert np.all(np.diff(tl) > 0)


def test_km_censoring_rate_roughly_right():
    df = _exp_survival(n=3000, censor_frac=0.30, seed=7)
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    assert res["n_censored"] + res["n_observed_events"] == res["n_subjects"]
    assert res["censoring_rate"] == pytest.approx(0.30, abs=0.05)


def test_km_median_positive_when_curve_crosses_half():
    df = _exp_survival(n=2000, scale=10.0, censor_frac=0.20, seed=3)
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    med = res["median_survival"]
    assert med is not None
    assert med > 0
    # exponential(scale=10) has true median = 10*ln(2) ~= 6.93
    assert 4.0 < med < 11.0


def test_km_confidence_bounds_valid():
    df = _exp_survival(seed=11)
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    lower = np.asarray(res["ci_lower"])
    upper = np.asarray(res["ci_upper"])
    surv = np.asarray(res["survival"])
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)
    assert np.all(lower <= surv + 1e-12)
    assert np.all(upper >= surv - 1e-12)


# --------------------------------------------------------------------------- #
# hand-computable KM check
# --------------------------------------------------------------------------- #
def test_km_matches_hand_computation_no_censoring():
    # 6 subjects, all events, times 1..6 -> textbook product-limit curve
    df = pd.DataFrame({"duration": [1, 2, 3, 4, 5, 6],
                       "event": [1, 1, 1, 1, 1, 1]})
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    # timeline includes the leading 0
    assert res["timeline"] == [0, 1, 2, 3, 4, 5, 6]
    expected = [1.0, 5 / 6, 4 / 6, 0.5, 2 / 6, 1 / 6, 0.0]
    assert np.allclose(res["survival"], expected, atol=1e-9)
    # first time S <= 0.5 is t = 3 (S exactly 0.5)
    assert res["median_survival"] == 3


def test_km_matches_hand_computation_with_censoring():
    # times 1..5, events at 1,3,5; subjects 2 and 4 censored
    df = pd.DataFrame({"duration": [1, 2, 3, 4, 5],
                       "event": [1, 0, 1, 0, 1]})
    res = survival.kaplan_meier(df, "duration", "event", show=False)
    # event times only: 0(seed),1,3,5
    assert res["timeline"] == [0, 1, 3, 5]
    # t=1: 5 at risk, 1 event -> 4/5 = 0.8
    # t=3: 3 at risk (durations 3,4,5), 1 event -> 0.8*2/3 = 0.5333
    # t=5: 1 at risk, 1 event -> 0.0
    assert np.allclose(res["survival"], [1.0, 0.8, 0.8 * 2 / 3, 0.0], atol=1e-9)
    assert res["n_censored"] == 2


# --------------------------------------------------------------------------- #
# grouped KM + log-rank
# --------------------------------------------------------------------------- #
def test_logrank_significant_for_different_hazards():
    a = _exp_survival(n=400, scale=5.0, censor_frac=0.20, seed=1)
    b = _exp_survival(n=400, scale=20.0, censor_frac=0.20, seed=2)
    a["arm"] = "fast"
    b["arm"] = "slow"
    df = pd.concat([a, b], ignore_index=True)
    res = survival.groups_kaplan_meier(df, "duration", "event", "arm", show=False)
    assert set(res["groups"].keys()) == {"fast", "slow"}
    assert res["logrank_dof"] == 1
    assert res["logrank_p_value"] < 0.05
    assert res["significant"] is True


def test_logrank_not_significant_for_identical_distributions():
    a = _exp_survival(n=500, scale=10.0, censor_frac=0.25, seed=100)
    b = _exp_survival(n=500, scale=10.0, censor_frac=0.25, seed=200)
    a["arm"] = "A"
    b["arm"] = "B"
    df = pd.concat([a, b], ignore_index=True)
    res = survival.groups_kaplan_meier(df, "duration", "event", "arm", show=False)
    assert res["logrank_p_value"] > 0.05
    assert res["significant"] is False


def test_groups_requires_two_levels():
    df = _exp_survival(n=50)
    df["arm"] = "only"
    with pytest.raises(ValueError, match="at least 2 groups"):
        survival.groups_kaplan_meier(df, "duration", "event", "arm", show=False)


# --------------------------------------------------------------------------- #
# honesty helper
# --------------------------------------------------------------------------- #
def test_check_censoring_warns_and_reports_rate(capsys):
    df = _exp_survival(n=1000, censor_frac=0.30, seed=42)
    res = survival.check_censoring(df, "duration", "event", show=True)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "censored" in out.lower()
    assert res["censoring_rate"] == pytest.approx(0.30, abs=0.05)


def test_kaplan_meier_calls_check_censoring(capsys):
    df = _exp_survival(n=200)
    survival.kaplan_meier(df, "duration", "event", show=True)
    out = capsys.readouterr().out
    assert "Censoring Check" in out
    assert "Kaplan-Meier" in out


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_rejects_non_binary_event():
    df = pd.DataFrame({"duration": [1, 2, 3], "event": [0, 1, 2]})
    with pytest.raises(ValueError, match="0/1"):
        survival.kaplan_meier(df, "duration", "event", show=False)


def test_rejects_negative_durations():
    df = pd.DataFrame({"duration": [1, -2, 3], "event": [1, 0, 1]})
    with pytest.raises(ValueError, match="negative"):
        survival.kaplan_meier(df, "duration", "event", show=False)
