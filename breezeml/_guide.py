"""
The guide: an in-library map so nobody ever feels lost.

BreezeML is layered like a garden path - each layer is optional, and you
can stop walking whenever you have what you came for.
"""
from __future__ import annotations

__all__ = ["guide"]

_GUIDE = """\
====================================================================
  BreezeML guide : the garden path
  Walk only as far as you need. Every layer is optional.
====================================================================

  BREATH 1 - first model (all you need on day one)
  --------------------------------------------------
    from breezeml import datasets, fit, predict
    df = datasets.iris()
    model = fit(df, "species")          # trains a sound pipeline
    predict(model, new_df)              # done.

  BREATH 2 - understand and choose
  --------------------------------------------------
    auto(df, target, explain_decisions=True)   # narrates every choice
    classifiers.compare(df, target)             # leaderboard, 22 models
    regressors.compare(df, target)              # leaderboard, 22 models
    classifiers.quick_tune(df, target, algo=..) # tune one model
    card(model, "MODEL_CARD.md")                # honest model card

  BREATH 3 - automate and ship
  --------------------------------------------------
    audit(df, target)                    # catch leaks BEFORE training
    automl(df, target, time_budget=60)   # search everything, honestly
    blend(df, target)                    # ensemble the leaderboard winners
    imbalance.tune_threshold(model, ..)  # stop using 0.5 on rare classes
    fairness.report(model, df, "group")  # who does the model serve?
    export(model, "train.py")            # pure-sklearn script, no lock-in
    deploy(model, "api/")                # FastAPI + Docker, ready to run
    drift.check(model, new_df)           # is production data drifting?
    timeseries.forecast(df, y, horizon)  # forecasting w/ naive baseline
    track.log(model, report, "name")     # remember which run was good

  BREATH 4 - beyond
  --------------------------------------------------
    features / clustering / anomaly / semisupervised   # toolkits
    text / explain / plot                               # more toolkits
    breezeml CLI + breezeml-mcp        # terminal + AI agents
    zen() . haiku() . fortune() . sensei()              # the garden

  Rules of the garden:
    - 4 dependencies, always. Extras are optional: [boost] [deploy]
      [automl] [mcp] [nlp] [explain] [plot] [onnx]
    - Your test set is sacred; splits are stratified; seeds are set.
    - When you outgrow BreezeML, export() and walk free.

  Docs: https://github.com/venomez-viper/breezeml
====================================================================
"""


def guide() -> str:
    """Print the garden path: what to use first, next, and later."""
    print(_GUIDE)
    return _GUIDE
