"""Tests for breezeml.active: uncertainty/margin/entropy/random querying and
the honest active-vs-random simulation curve."""
import pytest

import breezeml
from breezeml import active, datasets


@pytest.fixture(scope="module")
def trained():
    """A forest trained on part of breast_cancer, plus an unlabeled pool."""
    df = datasets.breast_cancer()
    labeled = df.sample(150, random_state=0)
    unlabeled = df.drop(index=labeled.index).reset_index(drop=True)
    model, _ = breezeml.classify(labeled, "label")
    return model, unlabeled


# ------------------------------------------------------------------- query

@pytest.mark.parametrize("strategy", ["uncertainty", "margin", "entropy", "random"])
def test_query_returns_k_valid_indices(trained, strategy):
    model, unlabeled = trained
    k = 12
    result = active.query(model, unlabeled, k=k, strategy=strategy,
                          target="label", show=False)
    assert result["strategy"] == strategy
    assert result["k"] == k
    assert len(result["indices"]) == k
    assert len(result["scores"]) == k
    valid = set(unlabeled.index)
    assert all(idx in valid for idx in result["indices"])
    # no duplicate picks
    assert len(set(result["indices"])) == k


def test_query_drops_target_column_if_present(trained):
    model, unlabeled = trained
    assert "label" in unlabeled.columns  # target still present in the pool
    result = active.query(model, unlabeled, k=5, strategy="uncertainty",
                          target="label", show=False)
    assert len(result["indices"]) == 5


def test_informative_strategies_differ_from_random(trained):
    model, unlabeled = trained
    rand = active.query(model, unlabeled, k=15, strategy="random",
                        target="label", show=False)
    for strat in ("uncertainty", "margin", "entropy"):
        picks = active.query(model, unlabeled, k=15, strategy=strat,
                             target="label", show=False)
        # soft check: informative ordering should not equal random's set
        assert set(picks["indices"]) != set(rand["indices"]), strat


def test_uncertainty_and_entropy_scores_non_negative(trained):
    model, unlabeled = trained
    for strat in ("uncertainty", "entropy"):
        result = active.query(model, unlabeled, k=20, strategy=strat,
                              target="label", show=False)
        assert all(s >= 0 for s in result["scores"]), strat


def test_query_k_capped_at_pool_size(trained):
    model, unlabeled = trained
    huge = len(unlabeled) + 500
    result = active.query(model, unlabeled, k=huge, strategy="uncertainty",
                          target="label", show=False)
    assert result["k"] == len(unlabeled)
    assert len(result["indices"]) == len(unlabeled)


def test_query_errors_without_predict_proba():
    from sklearn.svm import SVC

    df = datasets.breast_cancer()
    X = df.drop(columns=["label"])
    model = SVC(probability=False).fit(X, df["label"])  # no predict_proba
    with pytest.raises(TypeError, match="predict_proba"):
        active.query(model, X, k=5, show=False)


def test_query_bad_strategy_raises(trained):
    model, unlabeled = trained
    with pytest.raises(ValueError, match="strategy"):
        active.query(model, unlabeled, strategy="banana", target="label", show=False)


# ---------------------------------------------------------------- simulate

@pytest.fixture(scope="module")
def sim():
    df = datasets.breast_cancer()
    return active.simulate(df, "label", initial=20, budget=120, step=20,
                           strategy="uncertainty", show=False)


def test_simulate_structure(sim):
    for key in ("budgets", "active_accuracy", "random_accuracy",
                "area_between_curves", "active_wins"):
        assert key in sim
    n = len(sim["budgets"])
    assert n >= 2
    assert len(sim["active_accuracy"]) == n
    assert len(sim["random_accuracy"]) == n


def test_simulate_accuracies_in_unit_interval(sim):
    for a, r in zip(sim["active_accuracy"], sim["random_accuracy"]):
        assert 0.0 <= a <= 1.0
        assert 0.0 <= r <= 1.0


def test_simulate_budgets_monotonic_increasing(sim):
    budgets = sim["budgets"]
    assert budgets[0] == 20
    assert all(b2 > b1 for b1, b2 in zip(budgets, budgets[1:]))
    assert budgets[-1] <= 120


def test_simulate_learning_happened(sim):
    # More labels should not make the model worse. Assert structural learning,
    # NOT that active beats random (that is dataset-dependent and reported).
    active_acc = sim["active_accuracy"]
    assert active_acc[-1] >= active_acc[0] - 0.05


def test_simulate_active_wins_is_bool(sim):
    assert isinstance(sim["active_wins"], bool)
    assert isinstance(sim["area_between_curves"], float)


def test_simulate_random_strategy_also_runs():
    df = datasets.breast_cancer()
    result = active.simulate(df, "label", initial=20, budget=60, step=20,
                             strategy="random", show=False)
    assert len(result["budgets"]) == len(result["active_accuracy"])
