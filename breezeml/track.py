"""
Experiment tracking on zero extra dependencies.

Every serious ML project ends with "wait, which run was the good one?"
``breezeml.track`` answers it with a plain JSON file - no server, no
account, no MLflow install. Runs live in ``.breezeml/runs.json`` next to
your data, human-readable and git-committable.

    model, report = breezeml.auto(df, "churn")
    breezeml.track.log(model, report, name="baseline")

    ... days later ...
    breezeml.track.leaderboard()          # every run, ranked
    best = breezeml.track.best()          # highest headline metric
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

__all__ = ["log", "runs", "leaderboard", "best", "clear"]

_DIR = ".breezeml"
_FILE = "runs.json"


def _path(directory: str | None = None) -> str:
    return os.path.join(directory or _DIR, _FILE)


def _load(directory: str | None = None) -> list:
    path = _path(directory)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _headline(entry: dict):
    report = entry.get("report", {})
    for key in ("accuracy", "r2", "f1", "mae"):
        if isinstance(report.get(key), (int, float)):
            return key, report[key]
    return None, None


def log(model, report: dict, name: str | None = None, directory: str | None = None, show: bool = True) -> dict:
    """Record a training run: metrics, estimator, data profile, timestamp.

    Parameters
    ----------
    model : EasyModel (or anything with .meta / .task)
    report : dict
        The report returned by the training call.
    name : str, optional
        Human name for the run ("baseline", "tuned-gb", ...).
    directory : str, optional
        Where the ``.breezeml`` folder lives (default: current directory).
    """
    meta = getattr(model, "meta", None) or {}
    profile = meta.get("profile", {})
    entry = {
        "id": len(_load(directory)) + 1,
        "name": name or f"run-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "task": getattr(model, "task", meta.get("task", "unknown")),
        "target": getattr(model, "target", None),
        "estimator": meta.get("estimator", type(getattr(model, "pipeline", model)).__name__),
        "n_rows": profile.get("n_rows"),
        "report": {k: v for k, v in report.items() if isinstance(v, (int, float, str, bool))},
    }
    if meta.get("automl"):
        entry["automl"] = {k: meta["automl"][k] for k in ("best_model", "best_cv_score", "time_used") if k in meta["automl"]}

    entries = _load(directory)
    entries.append(entry)
    folder = directory or _DIR
    os.makedirs(folder, exist_ok=True)
    with open(_path(directory), "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    if show:
        key, value = _headline(entry)
        print(f"Logged run #{entry['id']} '{entry['name']}' ({entry['estimator']}, {key}={value})")
    return entry


def runs(directory: str | None = None) -> list:
    """All logged runs, oldest first."""
    return _load(directory)


def leaderboard(directory: str | None = None, show: bool = True) -> list:
    """Every run ranked by its headline metric (accuracy or R2 first)."""
    entries = _load(directory)
    scored = []
    for entry in entries:
        key, value = _headline(entry)
        scored.append({**entry, "_metric": key, "_value": value})
    # mae: lower is better - rank those ascending among themselves, after score metrics
    scored.sort(key=lambda e: (
        0 if e["_metric"] in ("accuracy", "r2", "f1") else 1,
        -(e["_value"] if e["_metric"] in ("accuracy", "r2", "f1") and e["_value"] is not None else 0),
        e["_value"] if e["_metric"] == "mae" and e["_value"] is not None else float("inf"),
    ))
    if show:
        print(f"\nBreezeML Run Leaderboard ({len(scored)} runs)")
        print(f"{'Rank':<6}{'Name':<24}{'Estimator':<28}{'Metric':<20}{'When':<18}")
        print("-" * 96)
        for i, e in enumerate(scored, 1):
            metric = f"{e['_metric']}={e['_value']}" if e["_metric"] else "-"
            print(f"{i:<6}{e['name'][:22]:<24}{str(e['estimator'])[:26]:<28}{metric:<20}{e['timestamp'][:16]:<18}")
        print()
    return scored


def best(directory: str | None = None) -> dict | None:
    """The best run by headline metric (None if no runs logged)."""
    ranked = leaderboard(directory, show=False)
    return ranked[0] if ranked else None


def clear(directory: str | None = None) -> int:
    """Delete all logged runs. Returns how many were removed."""
    entries = _load(directory)
    path = _path(directory)
    if os.path.exists(path):
        os.remove(path)
    return len(entries)
