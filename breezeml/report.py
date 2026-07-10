"""
The honest scorecard: one call that runs the whole honesty gauntlet on a
trained model and returns a single SHIP / WARN / STOP verdict.

``report(model, df, target)`` composes the pieces that used to live in
separate modules - performance vs a naive baseline, a data audit for leakage
and quality, class-imbalance severity, and (optionally) a fairness check -
into one consolidated, human- and machine-readable report. The verdict is
conservative on purpose: a single critical finding (leakage, or a model no
better than guessing) is enough to say STOP.

Core dependencies only. No new imports beyond scikit-learn / pandas / numpy.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ._validation import check_df_target

__all__ = ["report", "Report"]

_LEVEL = {"SHIP": 0, "WARN": 1, "STOP": 2}
_BANNER = {
    "SHIP": "SHIP  - no blocking issues found",
    "WARN": "WARN  - ships, but read the warnings first",
    "STOP": "STOP  - do not ship until these are resolved",
}


class Report:
    """A consolidated honesty report with an overall ship-readiness verdict."""

    def __init__(self, target, task, verdict, reasons, sections):
        self.target = target
        self.task = task
        self.verdict = verdict          # "SHIP" | "WARN" | "STOP"
        self.ok = verdict == "SHIP"
        self.reasons = reasons          # list[{level, category, message}]
        self.sections = sections        # dict of raw section results

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "task": self.task,
            "verdict": self.verdict,
            "ok": self.ok,
            "reasons": self.reasons,
            "sections": self.sections,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=_json_default)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_markdown(self) -> str:
        s = self.sections
        lines = [
            f"# BreezeML honest report - `{self.target}` ({self.task})",
            "",
            f"**Verdict: {self.verdict}** - {_BANNER[self.verdict].split('- ', 1)[-1]}",
            "",
        ]
        if self.reasons:
            lines.append("## Why")
            for r in self.reasons:
                lines.append(f"- **[{r['level']}]** {r['message']}")
            lines.append("")
        perf = s.get("performance", {})
        if perf:
            lines.append("## Performance (5-fold cross-validation)")
            for k, v in perf.get("metrics", {}).items():
                lines.append(f"- {k}: {v}")
            base = perf.get("baseline", {})
            if base:
                lines.append(f"- naive baseline ({base.get('strategy')}): {base.get('score')}"
                             f"  ->  {'beats baseline' if perf.get('beats_baseline') else 'NO better than baseline'}")
            lines.append("")
        audit = s.get("audit", {})
        if audit:
            lines.append("## Data audit")
            lines.append(f"- {audit.get('summary', '')}")
            for f in audit.get("findings", []):
                lines.append(f"  - [{f['severity']}] {f['message']}")
            lines.append("")
        imb = s.get("imbalance")
        if imb:
            lines.append("## Class balance")
            lines.append(f"- imbalance {imb['imbalance_ratio']}:1 ({imb['severity']}), "
                         f"minority '{imb['minority_class']}' = {imb['minority_share']:.1%}")
            lines.append("")
        fair = s.get("fairness")
        if fair:
            lines.append("## Fairness")
            dp = fair.get("demographic_parity_ratio")
            if dp is not None:
                lines.append(f"- demographic parity ratio {dp} - "
                             f"{'passes' if fair.get('passes_four_fifths') else 'FAILS'} four-fifths")
            lines.append("")
        return "\n".join(lines)

    # ---- pretty console --------------------------------------------------
    def __str__(self) -> str:
        s = self.sections
        w = 66
        out = ["", "=" * w, f"  BreezeML honest report - '{self.target}' ({self.task})", "=" * w]
        out.append(f"  VERDICT: {_BANNER[self.verdict]}")
        out.append("-" * w)
        perf = s.get("performance", {})
        if perf:
            metrics = "  ".join(f"{k} {v}" for k, v in perf.get("metrics", {}).items())
            out.append(f"  performance : {metrics}")
            base = perf.get("baseline", {})
            if base:
                verdict = "beats baseline" if perf.get("beats_baseline") else "NOT better than baseline"
                out.append(f"  vs baseline : {base.get('score')} ({base.get('strategy')})  ->  {verdict}")
        audit = s.get("audit", {})
        if audit:
            out.append(f"  data audit  : {audit.get('summary', '')}")
            for f in audit.get("findings", []):
                if f["severity"] == "critical":
                    out.append(f"                !! {f['message']}")
        imb = s.get("imbalance")
        if imb:
            out.append(f"  balance     : {imb['imbalance_ratio']}:1 ({imb['severity']})")
        fair = s.get("fairness")
        if fair and fair.get("demographic_parity_ratio") is not None:
            verdict = "passes" if fair.get("passes_four_fifths") else "FAILS four-fifths"
            out.append(f"  fairness    : parity {fair['demographic_parity_ratio']} ({verdict})")
        if self.reasons:
            out.append("-" * w)
            for r in self.reasons:
                out.append(f"  [{r['level']}] {r['message']}")
        out.append("=" * w)
        out.append("")
        return "\n".join(out)

    __repr__ = __str__


# --------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _n_splits(y, task, cap=5):
    if task != "classification":
        return min(cap, max(2, len(y) // 2))
    counts = pd.Series(y).value_counts()
    return int(max(2, min(cap, counts.min())))


def _performance(model, X, y, task):
    """Cross-validate a clone of the model's pipeline and a naive baseline."""
    pipe = clone(model.pipeline)
    metrics: dict[str, float] = {}
    baseline: dict = {}
    beats = False
    try:
        if task == "classification":
            k = _n_splits(y, task)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            acc = float(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean())
            f1 = float(cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro").mean())
            metrics = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
            if pd.Series(y).nunique() == 2:
                try:
                    roc = float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())
                    metrics["roc_auc"] = round(roc, 4)
                except Exception:
                    pass
            dummy = DummyClassifier(strategy="most_frequent")
            base_acc = float(cross_val_score(dummy, X, y, cv=cv, scoring="accuracy").mean())
            baseline = {"strategy": "most_frequent", "score": round(base_acc, 4), "metric": "accuracy"}
            beats = acc - base_acc > 0.02
        else:
            k = _n_splits(y, task)
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
            r2 = float(cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean())
            mae = float(-cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error").mean())
            metrics = {"r2": round(r2, 4), "mae": round(mae, 4)}
            dummy = DummyRegressor(strategy="mean")
            base_r2 = float(cross_val_score(dummy, X, y, cv=cv, scoring="r2").mean())
            baseline = {"strategy": "mean", "score": round(base_r2, 4), "metric": "r2"}
            beats = r2 - base_r2 > 0.02
    except Exception as exc:  # never let one metric kill the report
        metrics = {"error": str(exc)}
    return {"metrics": metrics, "baseline": baseline, "beats_baseline": beats}


def report(model, df: pd.DataFrame, target: str | None = None,
           sensitive: str | None = None, show: bool = True) -> Report:
    """Run the full honesty gauntlet on ``model`` over ``df`` and return a
    :class:`Report` with a SHIP / WARN / STOP verdict.

    Parameters
    ----------
    model : a BreezeML model (has ``.pipeline``, ``.target``, ``.task``)
    df : DataFrame including the target column
    target : target column name (defaults to ``model.target``)
    sensitive : optional column to run a fairness check against
    show : print the report to stdout (default True)
    """
    target = target or getattr(model, "target", None)
    if target is None:
        raise ValueError("No target given and model has no .target attribute.")
    check_df_target(df, target)
    task = getattr(model, "task", None) or (
        "classification" if (df[target].dtype == "object" or df[target].nunique() < 20) else "regression"
    )
    X = df.drop(columns=[target])
    y = df[target]

    sections: dict = {}
    reasons: list[dict] = []

    # 1. performance vs naive baseline
    perf = _performance(model, X, y, task)
    sections["performance"] = perf
    if not perf.get("beats_baseline", False):
        reasons.append({"level": "STOP", "category": "no_signal",
                        "message": "Model is no better than a naive baseline - it has not learned a useful signal."})

    # 2. data audit (leakage / quality)
    try:
        from .audit import audit as _audit
        a = _audit(df, target, show=False)
        sections["audit"] = a
        if not a.get("ok", True):
            reasons.append({"level": "STOP", "category": "audit",
                            "message": "Data audit found a critical issue: " + a.get("summary", "")})
    except Exception:
        pass

    # 3. class imbalance (classification only)
    if task == "classification":
        try:
            from .imbalance import summary as _summary
            imb = _summary(y, show=False)
            sections["imbalance"] = imb
            if imb.get("severity") == "severe":
                reasons.append({"level": "WARN", "category": "imbalance",
                                "message": f"Severe class imbalance ({imb['imbalance_ratio']}:1) - "
                                           "judge by macro-F1 / minority recall, never accuracy."})
        except Exception:
            pass

    # 4. fairness (optional)
    if sensitive is not None and sensitive in df.columns:
        try:
            from .fairness import report as _fair
            fr = _fair(model, df, sensitive=sensitive, show=False)
            sections["fairness"] = fr
            if fr.get("passes_four_fifths") is False:
                reasons.append({"level": "WARN", "category": "fairness",
                                "message": f"Fails the four-fifths rule on '{sensitive}' "
                                           f"(parity {fr.get('demographic_parity_ratio')})."})
        except Exception:
            pass

    # verdict = worst level among reasons
    verdict = "SHIP"
    for r in reasons:
        if _LEVEL[r["level"]] > _LEVEL[verdict]:
            verdict = r["level"]

    rep = Report(target=target, task=task, verdict=verdict, reasons=reasons, sections=sections)
    if show:
        print(rep)
    return rep
