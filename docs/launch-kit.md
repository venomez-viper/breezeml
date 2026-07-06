# Launch Kit (v1.0.0)

Copy-paste drafts for announcing the release. Post from your own accounts.

---

## Show HN (news.ycombinator.com/submit)

**Title:**
Show HN: BreezeML 1.0 - sklearn without boilerplate, 4 deps, zero lock-in, MCP built in

**URL:** https://github.com/venomez-viper/breezeml

**First comment (post immediately after submitting):**

I built BreezeML because every "low-code ML" library I tried either broke my
environment (PyCaret pulls in hundreds of packages) or stopped at a toy
leaderboard (LazyPredict gives you rankings but no pipeline).

v1.0 makes four promises:

1. 4 dependencies, always: core installs with only scikit-learn, pandas,
   numpy, joblib. A CI test fails the build if anyone adds a fifth.
2. Zero lock-in: export() writes a standalone sklearn script reproducing
   your exact pipeline - imputers, encoder, seed, split - with no breezeml
   import. You can graduate anytime.
3. It teaches: explain_decisions=True narrates every pipeline choice in
   plain English, computed from your actual data (imbalance ratios, missing
   percentages, IQR outlier checks). card() writes an honest model card
   with auto-detected caveats.
4. AI agents can use it: breezeml-mcp is a built-in MCP server, so Claude
   and other agents get stratified splits and leakage-safe pipelines
   instead of hand-rolling sklearn.

Honest benchmarks against PyCaret and LazyPredict (including where they
win) are in the docs. Happy to answer anything.

---

## r/MachineLearning (flair: Project)

**Title:**
[P] BreezeML 1.0: a 4-dependency sklearn wrapper with zero lock-in export, self-explaining pipelines, and an MCP server for AI agents

**Body:**

Open source (MIT), pip install breezeml.

The pitch in one block:

    import breezeml
    df = breezeml.datasets.iris()
    model, report = breezeml.auto(df, "species", explain_decisions=True)
    breezeml.card(model, "MODEL_CARD.md")   # honest model card
    breezeml.export(model, "train.py")      # pure-sklearn script, no breezeml import
    breezeml.deploy(model, "api/")          # FastAPI + Dockerfile

What I think is actually novel:

- **The dependency contract.** Core needs only sklearn/pandas/numpy/joblib
  and a CI test fails if anyone adds a fifth hard dependency. This is a
  direct response to the PyCaret dependency-hell experience.
- **export() as an anti-lock-in feature.** It generates a standalone
  sklearn training script reproducing the exact pipeline. Most libraries
  have every incentive not to build this.
- **Teaching narration.** Explanations are computed from measured data
  facts (class imbalance ratio, missing cell percentage, IQR outliers),
  not canned strings.
- **MCP server.** breezeml-mcp exposes train/compare/explain/export/deploy
  as tools, so agents get statistically sound defaults.

Benchmarks vs PyCaret and LazyPredict (with the caveats stated, including
where LazyPredict is faster): see docs/benchmarks.md in the repo.

Feedback welcome - there are 10 "good first issue"s open if anyone wants
to contribute.

---

## X / LinkedIn thread

1/ BreezeML 1.0 is out. Machine learning without the boilerplate - and
without the dependency hell. 4 core deps (sklearn, pandas, numpy, joblib),
enforced by CI. Forever.

2/ Most ML libraries lock you in. BreezeML has a "graduate anytime" button:
export(model, "train.py") writes a standalone sklearn script reproducing
your exact pipeline. Zero breezeml imports.

3/ It teaches while it trains. explain_decisions=True narrates every
choice: why a stratified split, why median imputation, why accuracy is
misleading on your imbalanced classes. Computed from your data, not canned.

4/ One line to production: deploy(model, "api/") writes a FastAPI app +
Dockerfile serving the raw sklearn pipeline. And card() writes an honest
model card with auto-detected caveats.

5/ The part I'm most excited about: breezeml-mcp. BreezeML is the first
low-code ML library with a built-in MCP server. Claude and other AI agents
can train, compare, explain, and deploy models with sound stats underneath.

6/ pip install breezeml
   Repo + honest benchmarks vs PyCaret/LazyPredict: github.com/venomez-viper/breezeml

---

## 90-second MCP demo script (screen recording)

Setup before recording:
- pip install breezeml[mcp]
- claude mcp add breezeml -- breezeml-mcp
- Have a real-ish CSV ready (e.g. churn.csv)

Shot list:
1. (0:00) Terminal: type `claude`, then the prompt:
   "Train a model on churn.csv predicting churn. Explain your
   preprocessing choices, write a model card, and deploy it as an API."
2. (0:10) Agent calls inspect_data -> show the JSON profile appearing.
3. (0:25) Agent calls compare -> leaderboard of 12+ models.
4. (0:40) Agent calls train -> metrics + the plain-English decisions list.
   Zoom on the narration for 3 seconds.
5. (0:55) Agent calls model_card -> open MODEL_CARD.md, scroll caveats.
6. (1:10) Agent calls deploy -> cd api && uvicorn app:app, then curl
   /predict with one record. Show the prediction JSON.
7. (1:25) Close card: "DataFrame to deployed, explained model. One prompt.
   pip install breezeml"
