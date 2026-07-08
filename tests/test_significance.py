"""
Tests for breezeml.significance: McNemar's test and the paired CV t-test,
plus the self-implemented erfc and Student t approximations.
"""
import numpy as np

from breezeml import datasets
from breezeml import classifiers
from breezeml.significance import (
    mcnemar,
    paired_cv_ttest,
    _erfc,
    _t_two_sided_p,
)


def test_mcnemar_two_different_models():
    df = datasets.breast_cancer()
    strong, _ = classifiers.random_forest(df, "label")
    weak, _ = classifiers.nearest_centroid(df, "label")

    result = mcnemar(strong, weak, df, "label", show=False)

    assert 0.0 <= result["p_value"] <= 1.0
    # Discordant counts must add up and be non-negative.
    assert result["only_a_correct"] >= 0
    assert result["only_b_correct"] >= 0
    assert result["n_discordant"] == result["only_a_correct"] + result["only_b_correct"]
    assert isinstance(result["significant"], bool)
    assert isinstance(result["verdict"], str) and result["verdict"]


def test_mcnemar_model_vs_itself():
    df = datasets.breast_cancer()
    model, _ = classifiers.random_forest(df, "label")

    # Same model against itself: zero discordant pairs, no crash on the
    # division-by-zero guard, p-value at 1, not significant.
    result = mcnemar(model, model, df, "label", show=False)

    assert result["n_discordant"] == 0
    assert result["only_a_correct"] == 0
    assert result["only_b_correct"] == 0
    assert result["p_value"] >= 0.99
    assert result["significant"] is False


def test_paired_cv_ttest_basic():
    df = datasets.breast_cancer()
    result = paired_cv_ttest("random_forest", "logistic", df, "label", cv=5, show=False)

    assert 0.0 <= result["p_value"] <= 1.0
    assert result["dof"] == 5 - 1
    assert "mean_score_a" in result
    assert "mean_score_b" in result
    assert "t_statistic" in result
    assert isinstance(result["significant"], bool)


def test_paired_cv_ttest_near_identical_models_reports_noise():
    df = datasets.breast_cancer()
    # Two identical estimators: per-fold differences are all zero, so the
    # test must fall back to "not significant" and say the gap is noise.
    result = paired_cv_ttest("logistic", "logistic", df, "label", cv=5, show=False)

    assert result["significant"] is False
    assert "likely noise" in result["verdict"].lower()
    assert result["dof"] == 4


def test_erfc_known_values():
    # erfc(0) = 1 exactly; erfc(1) is about 0.1573.
    assert abs(_erfc(0.0) - 1.0) < 1e-2
    assert abs(_erfc(1.0) - 0.1573) < 1e-2


def test_t_two_sided_p_matches_normal_tail():
    # For a large number of degrees of freedom the Student t two-sided
    # p-value converges to the standard normal one. The normal two-sided
    # p for z = 1.96 is about 0.05.
    p_normal = _erfc(1.96 / np.sqrt(2.0))
    assert abs(p_normal - 0.05) < 1e-2

    p_t = _t_two_sided_p(1.96, dof=2000)
    assert abs(p_t - 0.05) < 1e-2
