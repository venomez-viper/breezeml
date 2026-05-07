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
