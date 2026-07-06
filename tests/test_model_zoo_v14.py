"""Tests for the v1.4 model zoo expansion."""
import pytest

from breezeml import classifiers, clustering, datasets, regressors

NEW_CLASSIFIERS = ["hist_gradient_boosting", "ridge", "sgd", "lda", "qda", "complement_nb"]
NEW_REGRESSORS = ["hist_gradient_boosting", "extra_trees", "adaboost", "huber", "bayesian_ridge", "sgd"]


@pytest.fixture(scope="module")
def iris():
    return datasets.iris()


@pytest.fixture(scope="module")
def diabetes():
    return datasets.diabetes().sample(150, random_state=42)


@pytest.mark.parametrize("algo", NEW_CLASSIFIERS)
def test_new_classifier_trains(iris, algo):
    fn = getattr(classifiers, algo)
    model, report = fn(iris, "species")
    assert report["accuracy"] > 0.5, f"{algo}: {report}"
    assert "macro_f1" in report


@pytest.mark.parametrize("algo", NEW_REGRESSORS)
def test_new_regressor_trains(diabetes, algo):
    fn = getattr(regressors, algo)
    model, report = fn(diabetes, "target")
    assert report["r2"] is not None, f"{algo}: {report}"
    assert "rmse" in report


def test_classifier_leaderboard_includes_new_models(iris):
    results = classifiers.compare(iris, "species", show=False, progress=False)
    names = {r["classifier"] for r in results}
    assert {"Hist Gradient Boosting", "Ridge Classifier", "SGD (linear)", "LDA", "QDA", "Complement NB"} <= names
    assert len(results) >= 18


def test_regressor_leaderboard_includes_new_models(diabetes):
    results = regressors.compare(diabetes, "target", show=False, progress=False)
    names = {r["regressor"] for r in results}
    assert {"Hist Gradient Boosting", "Extra Trees", "AdaBoost", "Huber", "Bayesian Ridge", "SGD (linear)"} <= names
    assert len(results) >= 16


def test_new_models_tunable(iris):
    model, params, report = classifiers.quick_tune(iris, "species", algo="hist_gradient_boosting", n_iter=3, cv=2)
    assert params
    assert report["accuracy"] > 0.7


def test_new_regressor_tunable(diabetes):
    model, params, report = regressors.quick_tune(diabetes, "target", algo="extra_trees", n_iter=3, cv=2)
    assert params


def test_new_clustering_algorithms(iris):
    df = iris.drop(columns=["species"])
    for fn, kwargs in [
        (clustering.gaussian_mixture, {"n_clusters": 3}),
        (clustering.birch, {"n_clusters": 3}),
        (clustering.spectral, {"n_clusters": 3}),
    ]:
        result = fn(df, **kwargs)
        assert len(result["labels"]) == len(df)
        assert result["silhouette"] is not None and result["silhouette"] > 0.2


def test_gaussian_mixture_soft_membership(iris):
    df = iris.drop(columns=["species"])
    result = clustering.gaussian_mixture(df, n_clusters=3)
    assert result["probabilities"].shape == (len(df), 3)
    assert "bic" in result
