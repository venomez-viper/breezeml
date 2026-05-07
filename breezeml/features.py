"""
BreezeML feature engineering helpers.
"""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, chi2, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from ._validation import check_df_target


def _infer_task(y: pd.Series) -> str:
    return "classification" if y.dtype == "object" or y.nunique() < 20 else "regression"


def _encoded_features(df: pd.DataFrame, target: str):
    check_df_target(df, target)
    X = df.drop(columns=[target])
    y = df[target]
    X_encoded = pd.get_dummies(X, drop_first=False)
    return X_encoded, y


def _pipeline_feature_names(pipeline, X):
    if hasattr(pipeline, "named_steps") and "pre" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["pre"]
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    return list(X.columns)


def select(df: pd.DataFrame, target: str, method: str = "mutual_info", k: int = 10) -> pd.DataFrame:
    """Select the top-k features and return a reduced dataframe."""
    X_encoded, y = _encoded_features(df, target)
    task = _infer_task(y)
    k = min(k, X_encoded.shape[1])

    if method == "mutual_info":
        if task == "classification":
            scores = mutual_info_classif(X_encoded, y, random_state=42)
        else:
            scores = mutual_info_regression(X_encoded, y, random_state=42)
    elif method == "chi2":
        if task != "classification":
            raise ValueError("chi2 feature selection is only available for classification tasks.")
        scaled = MinMaxScaler().fit_transform(X_encoded)
        scores, _ = chi2(scaled, y)
    elif method == "rfe":
        estimator = RandomForestClassifier(n_estimators=200, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=200, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k)
        selector.fit(X_encoded, y)
        scores = selector.ranking_.astype(float) * -1.0
    else:
        raise ValueError("method must be one of: mutual_info, chi2, rfe")

    ranking = pd.Series(scores, index=X_encoded.columns).sort_values(ascending=False)
    selected_columns = ranking.head(k).index.tolist()

    print("Top selected features:")
    for name, score in ranking.head(k).items():
        print(f"  {name}: {round(float(score), 4)}")

    reduced = X_encoded[selected_columns].copy()
    reduced[target] = y.values
    return reduced


def importance(model, df: pd.DataFrame, target: str | None = None) -> dict[str, float]:
    """Return sorted feature importances for a trained model."""
    pipeline = getattr(model, "pipeline", model)
    target_name = target or getattr(model, "target", None)

    if target_name and target_name in df.columns:
        X = df.drop(columns=[target_name])
        y = df[target_name]
    else:
        X = df.copy()
        y = None

    feature_names = _pipeline_feature_names(pipeline, X)
    final_model = pipeline.named_steps["model"] if isinstance(pipeline, Pipeline) and "model" in pipeline.named_steps else pipeline

    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    else:
        if y is None:
            raise ValueError("target must be provided for permutation importance on models without native feature_importances_.")
        result = permutation_importance(pipeline, X, y, n_repeats=5, random_state=42, n_jobs=1)
        values = result.importances_mean

    pairs = sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)
    return {name: round(float(score), 4) for name, score in pairs}


def pca(df: pd.DataFrame, n_components=0.95) -> pd.DataFrame:
    """Replace numeric columns with PCA components."""
    numeric = df.select_dtypes(include="number")
    non_numeric = df.drop(columns=numeric.columns)
    if numeric.empty:
        raise ValueError("PCA requires at least one numeric column.")

    scaled = StandardScaler().fit_transform(numeric)
    pca_model = PCA(n_components=n_components, random_state=42)
    transformed = pca_model.fit_transform(scaled)
    columns = [f"pca_{i+1}" for i in range(transformed.shape[1])]
    pca_df = pd.DataFrame(transformed, columns=columns, index=df.index)
    return pd.concat([non_numeric, pca_df], axis=1)


def polynomial(df: pd.DataFrame, degree: int = 2, columns: list[str] | None = None) -> pd.DataFrame:
    """Append polynomial features for numeric columns."""
    target_columns = columns or df.select_dtypes(include="number").columns.tolist()
    if not target_columns:
        raise ValueError("Polynomial features require at least one numeric column.")

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    transformed = poly.fit_transform(df[target_columns])
    names = poly.get_feature_names_out(target_columns)
    poly_df = pd.DataFrame(transformed, columns=names, index=df.index)

    remaining = df.drop(columns=target_columns)
    existing = [column for column in poly_df.columns if column not in df.columns]
    poly_df = poly_df[existing]
    return pd.concat([df, poly_df], axis=1)
