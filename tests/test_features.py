from breezeml import datasets, features, regressors


def test_select_returns_reduced_dataframe():
    df = datasets.iris()
    selected = features.select(df, "species", method="mutual_info", k=3)
    assert "species" in selected.columns
    assert selected.shape[1] == 4


def test_importance_returns_dict():
    df = datasets.diabetes()
    model, _ = regressors.random_forest(df, "target")
    scores = features.importance(model, df, target="target")
    assert isinstance(scores, dict)
    first_value = next(iter(scores.values()))
    assert isinstance(first_value, float)


def test_pca_reduces_numeric_columns():
    df = datasets.iris()
    reduced = features.pca(df.drop(columns=["species"]), n_components=2)
    assert reduced.shape[1] == 2


def test_polynomial_increases_column_count():
    df = datasets.iris().drop(columns=["species"])
    expanded = features.polynomial(df, degree=2, columns=df.columns[:2].tolist())
    assert expanded.shape[1] > df.shape[1]
