"""
BreezeML honesty benchmark: BreezeML vs PyCaret vs LazyPredict.

Measures, per library (skipping any that are not installed):
  1. import time (cold, in a subprocess)
  2. time to produce a full model leaderboard on sklearn's Wine dataset
  3. best holdout accuracy found
  4. lines of user code required (from the snippets below)

Run:
    python benchmarks/run_benchmarks.py

The script never installs anything. To include a competitor:
    pip install pycaret        # heavy: ~1.5 GB of dependencies
    pip install lazypredict
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time

# User-code snippets each library needs for "load wine -> leaderboard".
# These are what the LOC column counts. Kept honest and minimal.
SNIPPETS = {
    "breezeml": """\
from breezeml import classifiers, datasets
df = datasets.wine()
results = classifiers.compare(df, "class", show=False)
""",
    "pycaret": """\
from pycaret.classification import setup, compare_models
from sklearn.datasets import load_wine
df = load_wine(as_frame=True).frame
setup(data=df, target="target", session_id=42, verbose=False)
best = compare_models(verbose=False)
""",
    "lazypredict": """\
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)
clf = LazyClassifier()
models, _ = clf.fit(X_train, X_test, y_train, y_test)
""",
}

IMPORT_TARGETS = {
    "breezeml": "import breezeml",
    "pycaret": "import pycaret.classification",
    "lazypredict": "import lazypredict.Supervised",
}


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def measure_import_time(statement: str) -> float:
    """Cold import time in a fresh interpreter (median of 3)."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", statement],
            capture_output=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return float("nan")
        times.append(elapsed)
    times.sort()
    return times[1]


def measure_leaderboard(library: str) -> dict:
    """Run the library's leaderboard snippet in-process, timed."""
    start = time.perf_counter()
    best_accuracy = None
    try:
        if library == "breezeml":
            from breezeml import classifiers, datasets

            df = datasets.wine()
            results = classifiers.compare(df, "class", show=False)
            best_accuracy = max(r["accuracy"] for r in results if r["accuracy"] is not None)
        elif library == "pycaret":
            from pycaret.classification import compare_models, pull, setup
            from sklearn.datasets import load_wine

            df = load_wine(as_frame=True).frame
            setup(data=df, target="target", session_id=42, verbose=False)
            compare_models(verbose=False)
            best_accuracy = float(pull()["Accuracy"].max())
        elif library == "lazypredict":
            from lazypredict.Supervised import LazyClassifier
            from sklearn.datasets import load_wine
            from sklearn.model_selection import train_test_split

            data = load_wine()
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=42
            )
            clf = LazyClassifier()
            models, _ = clf.fit(X_train, X_test, y_train, y_test)
            best_accuracy = float(models["Accuracy"].max())
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "leaderboard_seconds": round(time.perf_counter() - start, 2),
        "best_accuracy": round(best_accuracy, 4) if best_accuracy is not None else None,
    }


def main():
    rows = []
    for library in ["breezeml", "pycaret", "lazypredict"]:
        if not _installed(library):
            rows.append({"library": library, "installed": False})
            continue
        row = {"library": library, "installed": True}
        row["import_seconds"] = round(measure_import_time(IMPORT_TARGETS[library]), 2)
        row.update(measure_leaderboard(library))
        row["user_loc"] = len([ln for ln in SNIPPETS[library].splitlines() if ln.strip()])
        rows.append(row)

    print(json.dumps(rows, indent=2))

    print("\n| Library | Import time | Leaderboard time | Best accuracy | User LOC |")
    print("|---|---|---|---|---|")
    for row in rows:
        if not row["installed"]:
            print(f"| {row['library']} | *not installed* | - | - | - |")
            continue
        print(
            f"| {row['library']} | {row.get('import_seconds', '?')}s "
            f"| {row.get('leaderboard_seconds', '?')}s "
            f"| {row.get('best_accuracy', '?')} "
            f"| {row['user_loc']} |"
        )


if __name__ == "__main__":
    main()
