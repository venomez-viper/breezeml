# Launch Kit (v2.0)

Copy-paste drafts for announcing BreezeML 2.0. Post from your own accounts.
Post the HN one first; Reddit needs an aged account with some karma.

---

## Show HN (news.ycombinator.com/submit)

**Title:**
Show HN: BreezeML - ML that refuses to ship a model that can't beat a coin flip

**URL:** https://github.com/venomez-viper/breezeml

**First comment (post immediately after submitting):**

I built BreezeML after watching the same failure mode over and over:
someone trains a model, accuracy looks great, and nobody checked it against
a naive baseline or noticed the target leaked into a feature. Kapoor and
Narayanan catalogued this exact problem across 294 published papers, so it
is not a beginner-only mistake.

BreezeML is a workflow layer over scikit-learn where the honest path is the
short path:

    from breezeml import datasets, fit
    model = fit(datasets.iris(), "species")   # leakage-safe, stratified, seeded
    model.report()                            # SHIP / WARN / STOP

report() runs cross-validated performance against a naive baseline, a
leakage audit, imbalance checks, and optional fairness, then gives one
verdict. Train on shuffled labels and it says STOP, every time (we tested).

Other promises:

1. 4 dependencies, always: sklearn, pandas, numpy, joblib. A CI test fails
   the build if anyone adds a fifth.
2. Zero lock-in: export() writes a standalone sklearn script reproducing
   your exact pipeline, no breezeml import. Graduate anytime.
3. AI agents get the same guardrails: a built-in MCP server exposes the
   verdict as JSON, and agents are told to confirm SHIP before deploy().

The part I'd most like feedback on: we published an empirical validation of
our own guardrails (docs/validation.md). The first run caught a real blind
spot in our leakage detector - a depth-bounded probe tree cannot express a
copy of a continuous target - which we fixed and regression-tested. Numbers,
limitations, and the false-positive tradeoff are all in the doc.

Happy to answer anything, including where competing tools beat us
(LazyPredict's leaderboard is faster; it's in our benchmarks).

---

## r/MachineLearning (flair: Project, [P] tag)

**Title:**
[P] BreezeML 2.0: sklearn workflow layer that gives every model a SHIP/WARN/STOP verdict (and we validated the guardrails empirically)

**Body:**

Open source (MIT), pip install breezeml. 4 core dependencies, CI-enforced.

The core idea: leakage-safe defaults plus one honest gate.

    from breezeml import fit
    model = fit(df, "target")     # stratified, seeded, leakage-safe
    model.report()                # SHIP / WARN / STOP with reasons

report() = cross-validated score vs a mandatory naive baseline + a
single-feature leakage audit + imbalance severity + optional fairness
(four-fifths rule). One critical finding = STOP.

What I think is worth your attention:

- **We validated our own claims** (docs/validation.md): on 10 datasets,
  injected target copies are detected 10/10 with 0 false positives on
  clean data; conformal intervals hit 0.88-0.94 empirical coverage at
  nominal 90%; label-shuffled models get STOP 6/6, real models SHIP 6/6.
  The first run of the study exposed a real bug in our leakage probe
  (bounded-depth trees can't express a continuous target copy), which is
  now fixed and regression-tested. Limitations are documented, including
  what we deliberately don't flag and why.
- **Zero lock-in.** export() generates a standalone sklearn training
  script with no breezeml import.
- **Agent guardrails.** The MCP server returns the verdict as structured
  JSON and instructs agents to confirm SHIP before deploy/export. If you
  are building data-science agents, this is a concrete mechanism against
  silent statistical misuse.
- Also in the box on 4 deps: conformal prediction, significance tests
  (McNemar, paired CV t-test), survival analysis, drift monitoring, causal
  effect estimation.

Benchmarks vs PyCaret and LazyPredict, with the places they win stated
plainly: docs/benchmarks.md. Real-world case study (CMS Medicare fraud
ranking, ROC-AUC 0.809 matching the published benchmark): docs/case-studies.md.

Feedback and tear-downs welcome.

---

## X / LinkedIn thread

1/ BreezeML 2.0 is out. One call tells you if your model deserves to ship:
model.report() returns SHIP, WARN, or STOP. Baseline check, leakage audit,
imbalance, fairness. One verdict.

2/ Train on shuffled labels? STOP, 6/6 datasets. Real signal? SHIP, 6/6.
We published the validation study instead of asking you to trust us:
github.com/venomez-viper/breezeml (docs/validation.md)

3/ The study even caught a bug in our own leakage detector. A
depth-bounded tree cannot recognize a copy of a continuous target. Fixed,
regression-tested, documented. That is what "honest by default" costs.

4/ Still 4 core dependencies (sklearn, pandas, numpy, joblib), enforced by
a CI test that fails if anyone adds a fifth. Still zero lock-in:
export() writes a pure-sklearn script you can walk away with.

5/ And for AI agents: the built-in MCP server hands the SHIP/WARN/STOP
verdict to Claude and friends as JSON, and tells them to confirm SHIP
before deploying. Guardrails for agent-driven ML, not just humans.

6/ pip install breezeml
   Docs, benchmarks, validation, case study: github.com/venomez-viper/breezeml

---

## 90-second MCP demo script (screen recording)

Setup before recording:
- pip install breezeml[mcp]
- claude mcp add breezeml -- breezeml-mcp
- Have a real-ish CSV ready (e.g. churn.csv)

Shot list:
1. (0:00) Terminal: type `claude`, then the prompt:
   "Train a model on churn.csv predicting churn, check whether it is
   actually safe to ship, and if the report says SHIP, deploy it."
2. (0:10) Agent calls inspect_data -> JSON profile appears.
3. (0:20) Agent calls compare -> leaderboard of 20+ models.
4. (0:35) Agent calls train -> metrics + plain-English decisions.
5. (0:50) Agent calls report -> zoom on the SHIP/WARN/STOP verdict JSON
   for 3 seconds. This is the money shot.
6. (1:05) Agent confirms SHIP, calls deploy -> cd api && uvicorn app:app,
   curl /predict with one record, prediction JSON on screen.
7. (1:20) Close card: "The agent had to prove the model beats a baseline
   before it was allowed to deploy. pip install breezeml"
