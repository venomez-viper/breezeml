"""
Explainability: understand WHY the model predicts what it predicts.

Two paths:
- ``permutation_importance()`` and ``partial_dependence()`` run on the 4
  core dependencies (sklearn.inspection) - no extras needed.
- ``explain()`` produces SHAP summary plots via ``pip install breezeml[explain]``.
"""
from __future__ import annotations


def permutation_importance(model, df, target, n_repeats=5, random_state=42, show=True):
    """Model-agnostic feature importance on core dependencies only.

    Shuffles each feature and measures how much the score drops - features
    the model truly relies on hurt the most when scrambled. Unlike
    tree-impurity importances, this works for ANY model and is computed on
    data the model did not memorize feature order from.

    Returns a list of {feature, importance_mean, importance_std}, sorted.
    """
    from sklearn.inspection import permutation_importance as _perm

    from ._validation import check_df_target

    check_df_target(df, target)
    pipeline = getattr(model, "pipeline", model)
    X = df.drop(columns=[target])
    y = df[target]

    result = _perm(pipeline, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    rows = [
        {
            "feature": col,
            "importance_mean": round(float(result.importances_mean[i]), 4),
            "importance_std": round(float(result.importances_std[i]), 4),
        }
        for i, col in enumerate(X.columns)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    if show:
        print(f"\nPermutation importance (score drop when shuffled, {n_repeats} repeats)")
        print("-" * 56)
        for r in rows[:15]:
            bar = "#" * max(int(40 * r["importance_mean"] / max(rows[0]["importance_mean"], 1e-9)), 0)
            print(f"  {str(r['feature'])[:28]:<30}{r['importance_mean']:<9}{bar}")
        print()
    return rows


def partial_dependence(model, df, target, feature, grid_resolution=25):
    """How predictions change as ONE feature moves, all else held constant.

    Returns {"grid": [...], "average_prediction": [...]} - plot it, or read
    the direction: rising values mean the feature pushes predictions up.
    Core dependencies only.
    """
    from sklearn.inspection import partial_dependence as _pd

    from ._validation import check_df_target

    check_df_target(df, target)
    pipeline = getattr(model, "pipeline", model)
    X = df.drop(columns=[target])
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(X.columns)}")

    result = _pd(pipeline, X, features=[feature], grid_resolution=grid_resolution, kind="average")
    return {
        "feature": feature,
        "grid": [float(v) for v in result["grid_values"][0]],
        "average_prediction": [float(v) for v in result["average"][0]],
    }


def explain(model, df, target_col=None):
    """
    Generate a SHAP summary plot explaining the feature importance of a trained BreezeML model.

    Parameters
    ----------
    model : EasyModel or sklearn.pipeline.Pipeline
        The trained model pipeline.
    df : pd.DataFrame
        The dataset to explain the model on (can be training or test set).
    target_col : str, optional
        If the target column is still in the dataframe, specify it here to drop it.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "The 'shap' and 'matplotlib' libraries are required for explainability.\n"
            "Please install them using: pip install breezeml[explain]"
        )

    # Extract pipeline and drop target if present
    pipeline = getattr(model, "pipeline", model)
    X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()

    # Apply preprocessing to get the actual features the model sees
    try:
        preprocessor = pipeline.named_steps["pre"]
        X_transformed = preprocessor.transform(X)
        
        # Reconstruct feature names from the ColumnTransformer
        feature_names = []
        if hasattr(preprocessor, "transformers_"):
            for name, trans, cols in preprocessor.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                elif name == "cat":
                    # For one-hot encoded features, get the generated names
                    if hasattr(trans.named_steps["onehot"], "get_feature_names_out"):
                        cat_names = trans.named_steps["onehot"].get_feature_names_out(cols)
                        feature_names.extend(cat_names)
                    else:
                        feature_names.extend([f"{c}_encoded" for c in cols])
                elif name == "remainder" and trans != "drop":
                    feature_names.extend(cols)
    except Exception:
        # If pipeline structure is unexpected, fallback to raw
        X_transformed = X
        feature_names = X.columns.tolist()

    # Check if feature lengths match
    if hasattr(X_transformed, "shape") and len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

    final_model = pipeline.named_steps["model"]
    
    print("BreezeML 🌬️ Calculating SHAP values. This might take a moment...")
    
    # Choose the correct explainer
    # Tree models
    if type(final_model).__name__ in ["RandomForestClassifier", "RandomForestRegressor", 
                                      "DecisionTreeClassifier", "DecisionTreeRegressor",
                                      "GradientBoostingClassifier", "GradientBoostingRegressor",
                                      "ExtraTreesClassifier", "ExtraTreesRegressor"]:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_transformed)
    
    # Linear models
    elif type(final_model).__name__ in ["LogisticRegression", "LinearRegression", "LinearSVC"]:
        explainer = shap.LinearExplainer(final_model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        
    # Generic fallback
    else:
        # For SVM with RBF, KNN, MLP, etc., use KernelExplainer on a sample for speed
        background = shap.sample(X_transformed, 100) if X_transformed.shape[0] > 100 else X_transformed
        explainer = shap.KernelExplainer(final_model.predict, background)
        shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
