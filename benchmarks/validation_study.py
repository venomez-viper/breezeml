"""Empirical validation of BreezeML's statistical guardrails.

Three studies, each validating a documented claim:

  A. Leakage detection -- does ``audit()`` catch injected target leakage,
     and does it stay quiet on clean data?
  B. Conformal coverage -- do ``conformal_regressor`` / ``conformal_classifier``
     hit their nominal coverage guarantee empirically?
  C. Verdict sanity -- does ``report()`` say SHIP on a sound model and
     refuse SHIP when the labels carry no signal (shuffled labels)?

Run:  python benchmarks/validation_study.py
Writes benchmarks/validation_results.json and prints Markdown tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# validate the repository source, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)

import breezeml
from breezeml import audit, fit
from breezeml.conformal import conformal_classifier, conformal_regressor
from breezeml.report import report

RNG = np.random.default_rng(0)
ALPHA = 0.10          # nominal 90% coverage
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- datasets
def _to_df(X, y, target="target"):
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df[target] = y
    return df


def classification_datasets():
    out = {}
    for name, loader in [("iris", load_iris), ("wine", load_wine),
                         ("breast_cancer", load_breast_cancer)]:
        d = loader()
        out[name] = _to_df(d.data, d.target)
    d = load_digits()
    idx = RNG.choice(len(d.data), 800, replace=False)
    out["digits_800"] = _to_df(d.data[idx], d.target[idx])
    for i, (n, feats, informative) in enumerate([(1000, 20, 10), (1500, 30, 5)]):
        X, y = make_classification(n_samples=n, n_features=feats,
                                   n_informative=informative, random_state=i)
        out[f"synth_clf_{i}"] = _to_df(X, y)
    return out


def regression_datasets():
    out = {}
    d = load_diabetes()
    out["diabetes"] = _to_df(d.data, d.target)
    for i, (n, feats) in enumerate([(1000, 15), (1500, 25)]):
        X, y = make_regression(n_samples=n, n_features=feats, noise=15.0,
                               random_state=i)
        out[f"synth_reg_{i}"] = _to_df(X, y)
    X, y = make_friedman1(n_samples=1200, noise=1.0, random_state=0)
    out["friedman1"] = _to_df(X, y)
    return out


# ---------------------------------------------------------------- study A
def leakage_findings(df):
    result = audit(df, "target", show=False)
    cats = {f["category"] for f in result["findings"]
            if f["severity"] == "critical"}
    return cats


def study_a():
    rows = []
    datasets = {**classification_datasets(), **regression_datasets()}
    for name, df in datasets.items():
        n = len(df)
        y = df["target"]

        # clean data: a target_leakage finding here is a false positive
        clean_fp = "target_leakage" in leakage_findings(df)

        # direct copy of the target
        d1 = df.copy()
        d1["leaky_copy"] = pd.factorize(y)[0] if y.dtype == object else y
        direct = "target_leakage" in leakage_findings(d1)

        # noisy copy: 10% of rows corrupted
        d2 = df.copy()
        noisy = np.asarray(pd.factorize(y)[0] if y.dtype == object else y,
                           dtype=float).copy()
        flip = RNG.choice(n, max(1, n // 10), replace=False)
        noisy[flip] = RNG.permutation(noisy[flip])
        d2["leaky_noisy"] = noisy
        noisy_hit = "target_leakage" in leakage_findings(d2)

        # id column
        d3 = df.copy()
        d3["record_id"] = np.arange(n)
        idcol = "id_columns" in {f["category"]
                                 for f in audit(d3, "target", show=False)["findings"]}

        rows.append({"dataset": name, "clean_false_positive": clean_fp,
                     "direct_copy_detected": direct,
                     "noisy_copy_detected": noisy_hit,
                     "id_column_detected": idcol})
    return rows


# ---------------------------------------------------------------- study B
def study_b():
    rows = []
    for name, df in regression_datasets().items():
        covs = []
        for seed in SEEDS:
            d = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(d)
            tr, cal, te = d[: int(.6 * n)], d[int(.6 * n): int(.8 * n)], d[int(.8 * n):]
            model = fit(tr, "target")
            cp = conformal_regressor(model, cal, "target", alpha=ALPHA)
            r = cp.coverage_report(te, "target", show=False)
            covs.append(r["empirical_coverage"])
        rows.append({"dataset": name, "task": "regression",
                     "nominal": 1 - ALPHA,
                     "mean_coverage": round(float(np.mean(covs)), 4),
                     "std": round(float(np.std(covs)), 4)})
    for name, df in classification_datasets().items():
        covs = []
        for seed in SEEDS:
            d = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(d)
            tr, cal, te = d[: int(.6 * n)], d[int(.6 * n): int(.8 * n)], d[int(.8 * n):]
            model = fit(tr, "target")
            cp = conformal_classifier(model, cal, "target", alpha=ALPHA)
            r = cp.coverage_report(te, "target", show=False)
            covs.append(r["empirical_coverage"])
        rows.append({"dataset": name, "task": "classification",
                     "nominal": 1 - ALPHA,
                     "mean_coverage": round(float(np.mean(covs)), 4),
                     "std": round(float(np.std(covs)), 4)})
    return rows


# ---------------------------------------------------------------- study C
def study_c():
    rows = []
    for name, df in classification_datasets().items():
        model = fit(df, "target")
        clean_verdict = report(model, df, show=False).verdict

        shuffled = df.copy()
        shuffled["target"] = RNG.permutation(shuffled["target"].to_numpy())
        model_s = fit(shuffled, "target")
        shuffled_verdict = report(model_s, shuffled, show=False).verdict

        rows.append({"dataset": name, "clean_verdict": clean_verdict,
                     "shuffled_verdict": shuffled_verdict,
                     "gate_correct": clean_verdict == "SHIP"
                     and shuffled_verdict in ("WARN", "STOP")})
    return rows


# ---------------------------------------------------------------- main
def md_table(rows):
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    results = {}

    print("## Study A: leakage detection\n")
    results["leakage_detection"] = study_a()
    print(md_table(results["leakage_detection"]))

    print("\n## Study B: conformal coverage (nominal 90%)\n")
    results["conformal_coverage"] = study_b()
    print(md_table(results["conformal_coverage"]))

    print("\n## Study C: SHIP/WARN/STOP verdict sanity\n")
    results["verdict_sanity"] = study_c()
    print(md_table(results["verdict_sanity"]))

    results["meta"] = {
        "breezeml_version": breezeml.__version__,
        "alpha": ALPHA, "seeds": SEEDS,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = Path(__file__).parent / "validation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out} in {results['meta']['runtime_seconds']}s")


if __name__ == "__main__":
    main()
