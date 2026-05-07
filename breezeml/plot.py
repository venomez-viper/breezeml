def confusion_matrix(model, X_test, y_test, cmap="Blues"):
    """
    Plot a confusion matrix for the given model and test set.
    """
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The 'matplotlib' library is required for plotting.\n"
            "Please install it using: pip install breezeml[plot]"
        )

    print("BreezeML 🌬️ Generating confusion matrix plot...")
    
    # Extract prediction function (handles both EasyModel and raw pipelines)
    if hasattr(model, "predict"):
        preds = model.predict(X_test)
    else:
        preds = model(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap=cmap, ax=ax)
    plt.title("Confusion Matrix", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()


def roc_curve(model, X_test, y_test):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    """
    try:
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The 'matplotlib' library is required for plotting.\n"
            "Please install it using: pip install breezeml[plot]"
        )

    pipeline = getattr(model, "pipeline", model)
    if not hasattr(pipeline, "predict_proba") and not hasattr(pipeline, "decision_function"):
        raise TypeError("This model does not support probability predictions required for an ROC curve.")
        
    print("BreezeML 🌬️ Generating ROC curve plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    
    plt.title("ROC Curve", pad=20, fontsize=14)
    plt.plot([0, 1], [0, 1], "k--", label="Chance Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
