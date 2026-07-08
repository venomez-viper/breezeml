"""Tests for breezeml.recommend (collaborative filtering)."""
import numpy as np
import pandas as pd
import pytest

from breezeml import recommend


# --------------------------------------------------------------- fixtures

def _two_group_ratings(n_users=50, n_items=30, seed=42):
    """Synthetic ratings with two user groups liking disjoint item sets.

    Group A (even users) rates items 0..14; group B (odd users) rates items
    15..29. This gives clear structure the SVD should recover.
    """
    rng = np.random.default_rng(seed)
    group_a_items = list(range(0, n_items // 2))
    group_b_items = list(range(n_items // 2, n_items))
    rows = []
    for u in range(n_users):
        items = group_a_items if u % 2 == 0 else group_b_items
        # Each user rates a random subset of their group's items.
        picks = rng.choice(items, size=rng.integers(5, len(items)), replace=False)
        for it in picks:
            rows.append({
                "user": f"u_{u}",
                "item": f"i_{it}",
                "stars": float(rng.integers(3, 6)),  # 3..5, they like their group
            })
    df = pd.DataFrame(rows)
    return df, group_a_items, group_b_items


# -------------------------------------------------------------- basic fit

def test_recommend_returns_k_unseen_items():
    df, _, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars", show=False)

    user = "u_0"
    seen = set(df[df["user"] == user]["item"])
    out = rec.recommend(user, k=5)

    assert len(out) == 5
    items = [it for it, _ in out]
    assert len(set(items)) == 5, "recommendations must be distinct"
    assert all(it not in seen for it in items), "must not recommend already-seen items"
    assert all(isinstance(score, float) for _, score in out)


def test_recommendations_lean_toward_group():
    """An even (group A) user should be steered toward group A items."""
    df, group_a_items, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars", show=False)

    group_a_names = {f"i_{it}" for it in group_a_items}
    out = rec.recommend("u_0", k=5)
    items = [it for it, _ in out]
    overlap = sum(1 for it in items if it in group_a_names)
    assert overlap >= 1, f"expected some group-A leaning, got {items}"


# --------------------------------------------------------------- cold start

def test_cold_start_report():
    df, _, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars", show=False)

    report = rec.recommend_report("does_not_exist", k=5)
    assert report["cold_start"] is True
    assert report["method"] == "popularity"
    assert len(report["recommendations"]) == 5
    assert "note" in report and report["note"]


def test_known_user_report_is_svd():
    df, _, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars", show=False)

    report = rec.recommend_report("u_0", k=5)
    assert report["cold_start"] is False
    assert report["method"] == "svd"
    assert len(report["recommendations"]) == 5


# ------------------------------------------------------------------- stats

def test_stats_shape_and_density():
    df, _, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars", show=False)

    s = rec.stats
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    assert s["shape"] == (n_users, n_items)
    assert s["n_users"] == n_users
    assert s["n_items"] == n_items

    # Density = observed pairs / (users * items).
    observed = df.drop_duplicates(subset=["user", "item"]).shape[0]
    expected_density = observed / (n_users * n_items)
    assert abs(s["density"] - expected_density) < 1e-6
    assert 0.0 < s["density"] <= 1.0


# ---------------------------------------------------------------- implicit

def test_implicit_path_works():
    df, group_a_items, _ = _two_group_ratings()
    rec = recommend.collaborative_filter(df, "user", "item", rating_col=None, show=False)

    assert rec.stats["implicit"] is True
    out = rec.recommend("u_0", k=5)
    assert len(out) == 5
    seen = set(df[df["user"] == "u_0"]["item"])
    assert all(it not in seen for it in [i for i, _ in out])


# ------------------------------------------------------------ sparse warning

def test_sparse_warning_prints(capsys):
    """A tiny, very sparse matrix should trigger the density warning."""
    rng = np.random.default_rng(0)
    rows = []
    # 60 users x 60 items but only one interaction each -> density ~1.7%... too high.
    # Make it far sparser: 200 users, 200 items, 150 lonely interactions.
    for i in range(150):
        rows.append({"user": f"u_{i}", "item": f"i_{int(rng.integers(0, 200))}"})
    df = pd.DataFrame(rows)
    recommend.collaborative_filter(df, "user", "item", show=True)
    out = capsys.readouterr().out
    assert "sparse" in out.lower()
    assert "WARNING" in out


def test_single_interaction_warning(capsys):
    df = pd.DataFrame({
        "user": ["a", "b", "c", "a", "b"],
        "item": ["x", "y", "z", "y", "x"],
    })
    recommend.collaborative_filter(df, "user", "item", show=True)
    out = capsys.readouterr().out
    assert "single interaction" in out.lower()


# ------------------------------------------------------------------ errors

def test_missing_column_raises():
    df, _, _ = _two_group_ratings()
    with pytest.raises(ValueError, match="not found"):
        recommend.collaborative_filter(df, "user", "nope", show=False)


def test_recommend_before_fit_raises():
    rec = recommend.Recommender()
    with pytest.raises(RuntimeError, match="not fitted"):
        rec.recommend("u_0", k=5)
