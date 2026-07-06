"""
BreezeML classifiers.
"""
from __future__ import annotations

import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from ._preprocessing import _build_preprocessor, _detect_types
from ._progress import ProgressBar
from ._validation import check_df_target


def _run_compare_tasks(factories, run_one, progress, desc):
    """Run leaderboard tasks in parallel with a live progress bar.

    Uses joblib's generator mode (joblib >= 1.3) so the bar advances as
    each model finishes; falls back to batch mode on older joblib, and to
    a serial loop where process pools are blocked.
    """
    bar = ProgressBar(len(factories), desc=desc, enabled=progress)
    items = list(factories.items())
    tasks = [delayed(run_one)(name, factory) for name, factory in items]
    results = []
    try:
        try:
            for result in Parallel(n_jobs=-1, return_as="generator")(tasks):
                results.append(result)
                bar.update(next(iter(result.values())))
        except TypeError:  # joblib < 1.3: no generator support
            results = Parallel(n_jobs=-1)(tasks)
            bar.count = len(results)
    except PermissionError:
        results = []
        for name, factory in items:
            results.append(run_one(name, factory))
            bar.update(name)
    bar.close()
    return results


def _as_series(y):
    return y if isinstance(y, pd.Series) else pd.Series(y)


def _round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def _cross_validate_classification(pipe, X_data, y_data, cv):
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, average="weighted"),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    }
    kwargs = {"estimator": pipe, "X": X_data, "y": y_data, "cv": cv, "scoring": scoring, "n_jobs": -1}
    try:
        scores = cross_validate(**kwargs)
    except PermissionError:
        scores = cross_validate(**{**kwargs, "n_jobs": 1})
    return {
        "accuracy": _round_or_none(scores["test_accuracy"].mean()),
        "accuracy_std": _round_or_none(scores["test_accuracy"].std()),
        "f1": _round_or_none(scores["test_f1"].mean()),
        "f1_std": _round_or_none(scores["test_f1"].std()),
        "macro_f1": _round_or_none(scores["test_macro_f1"].mean()),
        "macro_f1_std": _round_or_none(scores["test_macro_f1"].std()),
    }


def _train(
    model,
    df: pd.DataFrame = None,
    target: str = None,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    force_minmax=False,
    cv=None,
):
    """Train a classifier and return (pipeline, report)."""
    if X is not None and y is not None:
        y_series = _as_series(y)
        X_data = X
        pipe = Pipeline([("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X, y_series
            X_te = X_test
            y_te = _as_series(y_test)
            pipe.fit(X_tr, y_tr)
        elif cv is not None:
            report = _cross_validate_classification(pipe, X_data, y_series, cv)
            pipe.fit(X_data, y_series)
            return pipe, report
        else:
            stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)
            pipe.fit(X_tr, y_tr)
    else:
        check_df_target(df, target)
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols, force_minmax=force_minmax)
        pipe = Pipeline([("pre", pre), ("model", model)])

        if X_test is not None and y_test is not None:
            X_tr, y_tr = X_df, y_df
            X_te = X_test
            y_te = _as_series(y_test)
            pipe.fit(X_tr, y_tr)
        elif cv is not None:
            report = _cross_validate_classification(pipe, X_df, y_df, cv)
            pipe.fit(X_df, y_df)
            return pipe, report
        else:
            stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
            X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)
            pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    report = {
        "accuracy": _round_or_none(accuracy_score(y_te, pred)),
        "f1": _round_or_none(f1_score(y_te, pred, average="weighted")),
        "macro_f1": _round_or_none(f1_score(y_te, pred, average="macro", zero_division=0)),
    }
    return pipe, report


def _load_xgb_classifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install XGBoost support with: pip install breezeml[boost]") from exc
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )


def _load_lgbm_classifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("Install LightGBM support with: pip install breezeml[boost]") from exc
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        verbose=-1,
    )


def logistic(df: pd.DataFrame = None, target: str = None, max_iter: int = 500, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(LogisticRegression(max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def svm(
    df: pd.DataFrame = None,
    target: str = None,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(SVC(kernel=kernel, C=C, gamma=gamma, probability=True), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def linear_svm(df: pd.DataFrame = None, target: str = None, C: float = 1.0, max_iter: int = 5000, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(LinearSVC(C=C, dual=False, class_weight="balanced", max_iter=max_iter), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def gaussian_nb(df: pd.DataFrame = None, target: str = None, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(GaussianNB(), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def multinomial_nb(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(MultinomialNB(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, force_minmax=True, cv=cv)


def decision_tree(df: pd.DataFrame = None, target: str = None, random_state: int = 42, max_depth: int | None = None, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(DecisionTreeClassifier(random_state=random_state, max_depth=max_depth), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def random_forest(df: pd.DataFrame = None, target: str = None, n_estimators: int = 200, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def knn(df: pd.DataFrame = None, target: str = None, n_neighbors: int = 5, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(KNeighborsClassifier(n_neighbors=n_neighbors), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def gradient_boosting(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def adaboost(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def extra_trees(df: pd.DataFrame = None, target: str = None, n_estimators: int = 200, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def mlp(df: pd.DataFrame = None, target: str = None, hidden_layer_sizes=(100,), max_iter: int = 500, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def xgboost(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        _load_xgb_classifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def lightgbm(
    df: pd.DataFrame = None,
    target: str = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        _load_lgbm_classifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, random_state=random_state),
        df=df,
        target=target,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
    )


def hist_gradient_boosting(
    df: pd.DataFrame = None,
    target: str = None,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    random_state: int = 42,
    *,
    X=None,
    y=None,
    X_test=None,
    y_test=None,
    cv=None,
):
    return _train(
        HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, random_state=random_state),
        df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv,
    )


def ridge(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(RidgeClassifier(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def sgd(df: pd.DataFrame = None, target: str = None, loss: str = "log_loss", alpha: float = 1e-4, max_iter: int = 1000, random_state: int = 42, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(SGDClassifier(loss=loss, alpha=alpha, max_iter=max_iter, random_state=random_state), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def lda(df: pd.DataFrame = None, target: str = None, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(LinearDiscriminantAnalysis(), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def qda(df: pd.DataFrame = None, target: str = None, reg_param: float = 0.0, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(QuadraticDiscriminantAnalysis(reg_param=reg_param), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, cv=cv)


def complement_nb(df: pd.DataFrame = None, target: str = None, alpha: float = 1.0, *, X=None, y=None, X_test=None, y_test=None, cv=None):
    return _train(ComplementNB(alpha=alpha), df=df, target=target, X=X, y=y, X_test=X_test, y_test=y_test, force_minmax=True, cv=cv)


def _base_classifier_factories():
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=500),
        "SVM (RBF)": lambda: SVC(probability=True),
        "Linear SVM": lambda: LinearSVC(dual=False, class_weight="balanced"),
        "Gaussian NB": lambda: GaussianNB(),
        "Multinomial NB": lambda: MultinomialNB(),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": lambda: KNeighborsClassifier(),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=42),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "MLP (Neural Net)": lambda: MLPClassifier(max_iter=500, random_state=42),
        "Hist Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=42),
        "Ridge Classifier": lambda: RidgeClassifier(),
        "SGD (linear)": lambda: SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "QDA": lambda: QuadraticDiscriminantAnalysis(),
        "Complement NB": lambda: ComplementNB(),
    }


def _classifier_factories():
    factories = _base_classifier_factories()
    try:
        _load_xgb_classifier()
        factories["XGBoost"] = lambda: _load_xgb_classifier()
    except ImportError:
        pass
    try:
        _load_lgbm_classifier()
        factories["LightGBM"] = lambda: _load_lgbm_classifier()
    except ImportError:
        pass
    return factories


def compare(df: pd.DataFrame = None, target: str = None, show: bool = True, progress: bool | None = None, *, X=None, y=None, cv=None):
    if X is None or y is None:
        check_df_target(df, target)
    if progress is None:
        progress = show

    def _run_one(name, factory_func):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                force = name in ("Multinomial NB", "Complement NB")
                _, report = _train(factory_func(), df=df, target=target, X=X, y=y, force_minmax=force, cv=cv)
            return {"classifier": name, **report}
        except Exception as exc:
            print(f"Warning: {name} failed with error: {exc}")
            return {"classifier": name, "accuracy": None, "f1": None}

    results = _run_compare_tasks(_classifier_factories(), _run_one, progress, "Training classifiers")

    results.sort(key=lambda row: row["accuracy"] if row["accuracy"] is not None else -1, reverse=True)

    if show:
        title = f"\nBreezeML Classifier Leaderboard - target: '{target}'"
        print(title)
        if cv is None:
            print(f"{'Rank':<6}{'Classifier':<25}{'Accuracy':<12}{'F1':<12}")
            print("-" * 55)
            for i, row in enumerate(results, 1):
                acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "FAILED"
                f1_value = f"{row['f1']:.4f}" if row["f1"] is not None else "FAILED"
                print(f"{i:<6}{row['classifier']:<25}{acc:<12}{f1_value:<12}")
        else:
            print(f"{'Rank':<6}{'Classifier':<25}{'Accuracy':<22}{'F1':<22}")
            print("-" * 75)
            for i, row in enumerate(results, 1):
                if row["accuracy"] is None:
                    acc = "FAILED"
                    f1_value = "FAILED"
                else:
                    acc = f"{row['accuracy']:.4f} +/- {row['accuracy_std']:.4f}"
                    f1_value = f"{row['f1']:.4f} +/- {row['f1_std']:.4f}"
                print(f"{i:<6}{row['classifier']:<25}{acc:<22}{f1_value:<22}")
        top = results[0].get("accuracy")
        if top is not None and top >= 1.0:
            print("Perfect accuracy detected. Either the problem is easy or the target")
            print("leaked into the features. Both deserve a second look before bragging.")
        print()

    return results


def _algo_factories():
    factories = {
        "logistic": lambda: LogisticRegression(max_iter=500),
        "svm": lambda: SVC(probability=True),
        "linear_svm": lambda: LinearSVC(dual=False, class_weight="balanced"),
        "gaussian_nb": lambda: GaussianNB(),
        "multinomial_nb": lambda: MultinomialNB(),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "knn": lambda: KNeighborsClassifier(),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
        "adaboost": lambda: AdaBoostClassifier(random_state=42),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "mlp": lambda: MLPClassifier(max_iter=500, random_state=42),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=42),
        "ridge": lambda: RidgeClassifier(),
        "sgd": lambda: SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
        "lda": lambda: LinearDiscriminantAnalysis(),
        "qda": lambda: QuadraticDiscriminantAnalysis(),
        "complement_nb": lambda: ComplementNB(),
    }
    try:
        _load_xgb_classifier()
        factories["xgboost"] = lambda: _load_xgb_classifier()
    except ImportError:
        pass
    try:
        _load_lgbm_classifier()
        factories["lightgbm"] = lambda: _load_lgbm_classifier()
    except ImportError:
        pass
    return factories


def detailed_report(df: pd.DataFrame = None, target: str = None, model=None, algo: str = "random_forest", *, X=None, y=None):
    if X is None or y is None:
        check_df_target(df, target)

    algo_factories = _algo_factories()
    if X is not None and y is not None:
        y_series = _as_series(y)
        stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)
        if model is None:
            factory = algo_factories.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")
            pipe = Pipeline([("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model
    else:
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols, force_minmax=(algo == "multinomial_nb"))
        stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)
        if model is None:
            factory = algo_factories.get(algo)
            if factory is None:
                raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")
            pipe = Pipeline([("pre", pre), ("model", factory())])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = model

    pred = pipe.predict(X_te)
    roc = None
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)
            if proba.shape[1] == 2:
                roc = _round_or_none(roc_auc_score(y_te, proba[:, 1]))
            else:
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y_te)
                roc = _round_or_none(roc_auc_score(y_encoded, proba, multi_class="ovr", average="weighted"))
    except Exception:
        roc = None

    return {
        "accuracy": _round_or_none(accuracy_score(y_te, pred)),
        "f1": _round_or_none(f1_score(y_te, pred, average="weighted")),
        "macro_f1": _round_or_none(f1_score(y_te, pred, average="macro", zero_division=0)),
        "precision": _round_or_none(precision_score(y_te, pred, average="weighted", zero_division=0)),
        "recall": _round_or_none(recall_score(y_te, pred, average="weighted", zero_division=0)),
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "classification_report": classification_report(y_te, pred, output_dict=True, zero_division=0),
        "model": pipe,
    }


_PARAM_GRIDS = {
    "logistic": {"model__C": [0.01, 0.1, 1, 10], "model__max_iter": [200, 500, 1000]},
    "svm": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
    "knn": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean", "manhattan"]},
    "decision_tree": {"model__max_depth": [3, 5, 10, 20, None], "model__min_samples_split": [2, 5, 10], "model__criterion": ["gini", "entropy"]},
    "random_forest": {"model__n_estimators": [100, 200, 500], "model__max_depth": [5, 10, 20, None], "model__min_samples_split": [2, 5, 10]},
    "gradient_boosting": {"model__n_estimators": [100, 200, 500], "model__learning_rate": [0.01, 0.1, 0.2], "model__max_depth": [3, 5, 7]},
    "adaboost": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "extra_trees": {"model__n_estimators": [100, 200, 500], "model__max_depth": [5, 10, 20, None]},
    "mlp": {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)], "model__learning_rate_init": [0.001, 0.01], "model__max_iter": [300, 500]},
    "hist_gradient_boosting": {"model__learning_rate": [0.01, 0.05, 0.1, 0.2], "model__max_iter": [100, 200, 400], "model__max_depth": [None, 3, 6, 10]},
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "sgd": {"model__alpha": [1e-5, 1e-4, 1e-3], "model__penalty": ["l2", "l1", "elasticnet"], "model__loss": ["log_loss", "hinge"]},
    "xgboost": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [3, 5, 7, 10],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [100, 200, 500],
        "model__num_leaves": [31, 63, 127],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
}


def quick_tune(df: pd.DataFrame = None, target: str = None, algo: str = "random_forest", n_iter: int = 20, cv: int = 3, scoring: str = "accuracy", *, X=None, y=None):
    if X is None or y is None:
        check_df_target(df, target)

    algo_factories = _algo_factories()
    factory = algo_factories.get(algo)
    param_grid = _PARAM_GRIDS.get(algo)
    if factory is None or param_grid is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_factories.keys())}")

    if X is not None and y is not None:
        y_series = _as_series(y)
        pipe = Pipeline([("model", factory())])
        stratify = y_series if (y_series.nunique() > 1 and y_series.nunique() < len(y_series)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_series, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_df = df.drop(columns=[target])
        y_df = _as_series(df[target])
        num_cols, cat_cols = _detect_types(df, target)
        pre = _build_preprocessor(num_cols, cat_cols, force_minmax=(algo == "multinomial_nb"))
        pipe = Pipeline([("pre", pre), ("model", factory())])
        stratify = y_df if (y_df.nunique() > 1 and y_df.nunique() < len(y_df)) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=stratify)

    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    actual_iter = min(n_iter, grid_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_kwargs = {
            "estimator": pipe,
            "param_distributions": param_grid,
            "n_iter": actual_iter,
            "cv": cv,
            "scoring": scoring,
            "random_state": 42,
            "n_jobs": -1,
            "error_score": "raise",
        }
        search = RandomizedSearchCV(**search_kwargs)
        try:
            search.fit(X_tr, y_tr)
        except PermissionError:
            search = RandomizedSearchCV(**{**search_kwargs, "n_jobs": 1})
            search.fit(X_tr, y_tr)

    best_pipe = search.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    pred = best_pipe.predict(X_te)
    report = {
        "accuracy": _round_or_none(accuracy_score(y_te, pred)),
        "f1": _round_or_none(f1_score(y_te, pred, average="weighted")),
        "macro_f1": _round_or_none(f1_score(y_te, pred, average="macro", zero_division=0)),
    }

    print(f"Best params for {algo}: {best_params}")
    print(f"   Accuracy: {report['accuracy']}  |  F1: {report['f1']}  |  Macro F1: {report['macro_f1']}")

    return best_pipe, best_params, report


logistic_regression = logistic
naive_bayes = multinomial_nb
