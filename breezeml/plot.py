"""
BreezeML plotting helpers.
"""
from __future__ import annotations

from . import features


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "The 'matplotlib' library is required for plotting.\n"
            "Please install it using: pip install breezeml[plot]"
        ) from exc


def confusion_matrix(model, X_test, y_test, cmap="Blues"):
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError as exc:
        raise ImportError("scikit-learn is required for confusion matrix plotting.") from exc

    plt = _require_matplotlib()
    print("BreezeML Generating confusion matrix plot...")

    preds = model.predict(X_test) if hasattr(model, "predict") else model(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap=cmap, ax=ax)
    plt.title("Confusion Matrix", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()


def pr_curve(model, X_test, y_test):
    try:
        from sklearn.metrics import PrecisionRecallDisplay
    except ImportError as exc:
        raise ImportError("scikit-learn is required for PR curve plotting.") from exc

    plt = _require_matplotlib()
    pipeline = getattr(model, "pipeline", model)
    if not hasattr(pipeline, "predict_proba") and not hasattr(pipeline, "decision_function"):
        raise TypeError("This model does not support probability predictions required for a PR curve.")

    if len(set(y_test)) > 2:
        raise ValueError("pr_curve supports binary classification only.")

    print("BreezeML Generating PR curve plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    plt.title("Precision-Recall Curve", pad=20, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def roc_curve(model, X_test, y_test):
    try:
        from sklearn.metrics import RocCurveDisplay
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ROC plotting.") from exc

    plt = _require_matplotlib()
    pipeline = getattr(model, "pipeline", model)
    if not hasattr(pipeline, "predict_proba") and not hasattr(pipeline, "decision_function"):
        raise TypeError("This model does not support probability predictions required for an ROC curve.")

    print("BreezeML Generating ROC curve plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    plt.title("ROC Curve", pad=20, fontsize=14)
    plt.plot([0, 1], [0, 1], "k--", label="Chance Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_chart(results, metric="accuracy"):
    """Plot a bar chart from compare() results."""
    plt = _require_matplotlib()
    if not results:
        raise ValueError("results cannot be empty.")

    label_key = "classifier" if "classifier" in results[0] else "regressor"
    filtered = [row for row in results if row.get(metric) is not None]
    labels = [row[label_key] for row in filtered]
    values = [row[metric] for row in filtered]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color="#1f77b4")
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{label_key.title()} Comparison by {metric}")
    plt.tight_layout()
    plt.show()


def learning_curve(model, df, target, cv=5):
    """Plot training and validation curves as sample size grows."""
    try:
        from sklearn.model_selection import learning_curve as sk_learning_curve
    except ImportError as exc:
        raise ImportError("scikit-learn is required for learning curve plotting.") from exc

    plt = _require_matplotlib()
    pipeline = getattr(model, "pipeline", model)
    X = df.drop(columns=[target])
    y = df[target]

    train_sizes, train_scores, valid_scores = sk_learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Training score", marker="o")
    ax.plot(train_sizes, valid_scores.mean(axis=1), label="Validation score", marker="o")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def feature_importance(model, df, target=None, top_n=15):
    """Plot the top-N feature importances."""
    plt = _require_matplotlib()
    scores = features.importance(model, df, target=target)
    items = list(scores.items())[:top_n]
    labels = [name for name, _ in items]
    values = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1], color="#ff7f0e")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()
