"""
The BreezeML command line: sound ML without opening Python.

    breezeml train data.csv --target churn
    breezeml compare data.csv --target churn
    breezeml automl data.csv --target churn --budget 120
    breezeml audit data.csv --target churn
    breezeml deploy model.joblib --out api/
    breezeml card model.joblib --out MODEL_CARD.md
    breezeml zen
"""
from __future__ import annotations

import argparse
import sys


def _load_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def _cmd_train(args):
    import breezeml

    df = _load_csv(args.data)
    model, report = breezeml.auto(df, args.target, explain_decisions=args.explain)
    print("Report:", report)
    breezeml.save(model, args.out)
    print(f"Model saved to {args.out}")


def _cmd_compare(args):
    import breezeml

    df = _load_csv(args.data)
    y = df[args.target]
    if y.dtype == "object" or y.nunique() < 20:
        breezeml.classifiers.compare(df, args.target)
    else:
        breezeml.regressors.compare(df, args.target)


def _cmd_automl(args):
    import breezeml

    df = _load_csv(args.data)
    model, report = breezeml.automl(df, args.target, time_budget=args.budget)
    breezeml.save(model, args.out)
    print(f"Best model ({report['best_model']}) saved to {args.out}")


def _cmd_audit(args):
    import breezeml

    df = _load_csv(args.data)
    result = breezeml.audit(df, args.target)
    sys.exit(0 if result["ok"] else 1)


def _cmd_deploy(args):
    import breezeml

    model = breezeml.load(args.model)
    breezeml.deploy(model, args.out, name=args.name)


def _cmd_card(args):
    import breezeml

    model = breezeml.load(args.model)
    breezeml.card(model, args.out)
    print(f"Model card written to {args.out}")


def _cmd_zen(args):
    import breezeml

    breezeml.zen()


def _cmd_guide(args):
    import breezeml

    breezeml.guide()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breezeml",
        description="BreezeML: train, compare, audit, and deploy ML models from the terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train the auto-selected model on a CSV")
    p.add_argument("data"), p.add_argument("--target", required=True)
    p.add_argument("--out", default="model.joblib")
    p.add_argument("--explain", action="store_true", help="narrate every pipeline decision")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("compare", help="Leaderboard of all built-in models")
    p.add_argument("data"), p.add_argument("--target", required=True)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("automl", help="Budget-aware model search")
    p.add_argument("data"), p.add_argument("--target", required=True)
    p.add_argument("--budget", type=int, default=60), p.add_argument("--out", default="model.joblib")
    p.set_defaults(func=_cmd_automl)

    p = sub.add_parser("audit", help="Data quality + leakage audit (exit 1 on critical findings)")
    p.add_argument("data"), p.add_argument("--target", required=True)
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("deploy", help="Write a FastAPI + Docker serving directory")
    p.add_argument("model"), p.add_argument("--out", default="deployment")
    p.add_argument("--name", default="breezeml-model")
    p.set_defaults(func=_cmd_deploy)

    p = sub.add_parser("card", help="Generate a model card from a saved model")
    p.add_argument("model"), p.add_argument("--out", default="MODEL_CARD.md")
    p.set_defaults(func=_cmd_card)

    sub.add_parser("zen", help="The Zen of BreezeML").set_defaults(func=_cmd_zen)
    sub.add_parser("guide", help="The garden path: what to use first").set_defaults(func=_cmd_guide)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
