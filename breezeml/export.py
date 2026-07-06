"""
Export trained BreezeML models as standalone scikit-learn Python scripts.

The generated script has ZERO breezeml imports: it reproduces the exact
preprocessing + estimator pipeline using only scikit-learn, pandas, numpy,
and joblib, so users can graduate from BreezeML at any time.
"""
from __future__ import annotations

import importlib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ._meta import _safe_params

__all__ = ["export", "export_code"]

_HEADER = '''"""
Standalone training script exported by BreezeML.

This file reproduces the exact pipeline BreezeML trained, using only
scikit-learn / pandas / joblib. No breezeml import required.

Usage:
    1. Point DATA_PATH at your training CSV.
    2. python {script_name}
"""
'''


def _public_import_path(cls) -> tuple[str, str]:
    """Resolve the shortest public module path exposing ``cls``.

    e.g. sklearn.ensemble._forest.RandomForestClassifier -> ("sklearn.ensemble",
    "RandomForestClassifier").
    """
    name = cls.__name__
    parts = cls.__module__.split(".")
    for i in range(1, len(parts) + 1):
        candidate = ".".join(parts[:i])
        if any(seg.startswith("_") for seg in parts[:i]):
            continue
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        if getattr(module, name, None) is cls:
            return candidate, name
    return cls.__module__, name


def _estimator_construction(estimator) -> tuple[str, str]:
    """Return (import_line, constructor_expr) for an estimator."""
    module, name = _public_import_path(type(estimator))
    params = _safe_params(estimator)
    args = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"from {module} import {name}", f"{name}({args})"


def _unwrap(model):
    """Accept an EasyModel or a raw sklearn Pipeline."""
    pipeline = getattr(model, "pipeline", model)
    target = getattr(model, "target", "TARGET_COLUMN")
    task = getattr(model, "task", None)
    meta = getattr(model, "meta", None) or {}
    return pipeline, target, task, meta


def export_code(model, data_path: str = "YOUR_DATA.csv") -> str:
    """Generate the standalone training script as a string."""
    pipeline, target, task, meta = _unwrap(model)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Expected an EasyModel or sklearn Pipeline, got {type(pipeline).__name__}"
        )

    estimator = pipeline.named_steps.get("model", pipeline.steps[-1][1])
    estimator_import, estimator_expr = _estimator_construction(estimator)

    pre = pipeline.named_steps.get("pre")
    has_pre = isinstance(pre, ColumnTransformer)

    numeric, categorical, scaler_name = [], [], "StandardScaler"
    if has_pre:
        transformers = getattr(pre, "transformers_", None) or pre.transformers
        for name, transformer, columns in transformers:
            if name == "num":
                numeric = list(columns)
                if isinstance(transformer, Pipeline):
                    scaler = transformer.named_steps.get("scaler")
                    if isinstance(scaler, MinMaxScaler):
                        scaler_name = "MinMaxScaler"
            elif name == "cat":
                categorical = list(columns)

    task = task or ("classification" if hasattr(estimator, "predict_proba") else "regression")
    is_classification = task == "classification"

    lines = [_HEADER.format(script_name="train.py")]
    lines.append("import joblib")
    lines.append("import pandas as pd")
    lines.append("from sklearn.model_selection import train_test_split")
    lines.append("from sklearn.pipeline import Pipeline")
    if has_pre:
        lines.append("from sklearn.compose import ColumnTransformer")
        lines.append("from sklearn.impute import SimpleImputer")
        lines.append(f"from sklearn.preprocessing import OneHotEncoder, {scaler_name}")
    if is_classification:
        lines.append("from sklearn.metrics import accuracy_score, f1_score")
    else:
        lines.append("from sklearn.metrics import mean_absolute_error, r2_score")
    lines.append(estimator_import)
    lines.append("")
    lines.append(f'DATA_PATH = "{data_path}"')
    lines.append(f'TARGET = "{target}"')
    random_state = meta.get("random_state", 42)
    test_size = meta.get("test_size", 0.2)
    lines.append(f"RANDOM_STATE = {random_state}")
    lines.append(f"TEST_SIZE = {test_size}")
    lines.append("")
    lines.append("df = pd.read_csv(DATA_PATH)")
    lines.append("X = df.drop(columns=[TARGET])")
    lines.append("y = df[TARGET]")
    lines.append("")

    if has_pre:
        lines.append(f"numeric = {numeric!r}")
        lines.append(f"categorical = {categorical!r}")
        lines.append("")
        lines.append("preprocessor = ColumnTransformer([")
        lines.append("    (\"num\", Pipeline([")
        lines.append("        (\"imputer\", SimpleImputer(strategy=\"median\")),")
        lines.append(f"        (\"scaler\", {scaler_name}()),")
        lines.append("    ]), numeric),")
        lines.append("    (\"cat\", Pipeline([")
        lines.append("        (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),")
        lines.append("        (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\")),")
        lines.append("    ]), categorical),")
        lines.append("])")
        lines.append("")
        lines.append(f"model = {estimator_expr}")
        lines.append('pipeline = Pipeline([("pre", preprocessor), ("model", model)])')
    else:
        lines.append(f"model = {estimator_expr}")
        lines.append('pipeline = Pipeline([("model", model)])')

    lines.append("")
    if is_classification:
        lines.append("stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None")
        lines.append(
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)"
        )
    else:
        lines.append(
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)"
        )
    lines.append("pipeline.fit(X_train, y_train)")
    lines.append("")
    lines.append("preds = pipeline.predict(X_test)")
    if is_classification:
        lines.append('print("accuracy:", round(accuracy_score(y_test, preds), 4))')
        lines.append('print("f1 (weighted):", round(f1_score(y_test, preds, average="weighted"), 4))')
    else:
        lines.append('print("r2:", round(r2_score(y_test, preds), 4))')
        lines.append('print("mae:", round(mean_absolute_error(y_test, preds), 4))')
    lines.append("")
    lines.append('joblib.dump(pipeline, "model.joblib")')
    lines.append('print("Saved trained pipeline to model.joblib")')
    lines.append("")

    return "\n".join(lines)


def export(model, path: str = "train.py", data_path: str = "YOUR_DATA.csv") -> str:
    """Write the standalone training script to ``path`` and return the path.

    Parameters
    ----------
    model : EasyModel or sklearn Pipeline
        A trained BreezeML model.
    path : str
        Destination .py file.
    data_path : str
        Placeholder CSV path baked into the script.
    """
    code = export_code(model, data_path=data_path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path
