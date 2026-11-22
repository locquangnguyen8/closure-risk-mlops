"""closure-risk: run the full pipeline or individual stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Training modules are imported inside their commands: the serving image has no scikit-learn or MLflow.

DATA = Path("data")
REPORTS = Path("reports")
MODELS = Path("models")


def load_features() -> pd.DataFrame:
    """Load the feature table, generating the synthetic data first if needed."""
    path = DATA / "features.csv"
    if not path.exists():
        cmd_simulate(None)
    return pd.read_csv(path, parse_dates=["date"])


def cmd_simulate(_args) -> int:
    from .features import build_features
    from .simulate import simulate

    DATA.mkdir(exist_ok=True)
    raw = simulate()
    raw.to_csv(DATA / "raw.csv", index=False)
    feats = build_features(raw)
    feats.to_csv(DATA / "features.csv", index=False)
    print(
        f"{len(raw)} estuary-days simulated; {len(feats)} open-day training rows; positive rate {feats['target'].mean():.3f}"
    )
    return 0


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def backtest_report(result: dict) -> str:
    cols = ["balanced_accuracy", "roc_auc", "recall", "precision", "brier"]
    lines = [
        "# Rolling-origin backtest",
        "",
        "12 monthly folds (2024-07 to 2025-06), 7-day embargo between training and test data.",
        "",
        "| Model | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, m in result["overall"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m[c]) for c in cols) + " |")
    lines += [
        "",
        f"**Selected for production: `{result['selected_family']}`** (best balanced accuracy; ties go to the simpler model).",
        "",
        "## Balanced accuracy by fold",
        "",
        "| Month | logistic | xgboost | Positives / rows |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {f['month']} | {_fmt(f['logistic'])} | {_fmt(f['xgboost'])} | {f['positives']} / {f['n']} |"
        for f in result["folds"]
    ]
    return "\n".join(lines) + "\n"


def monitoring_report(run) -> str:
    lines = [
        "# Production monitoring simulation",
        "",
        "Drift is introduced from 2025-07: heavier rain and a stronger catchment response.",
        "",
        "| Month | Model | Balanced accuracy | 28-day balanced accuracy | Recall | Max PSI (90 days) | Action |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in run.months:
        lines.append(
            f"| {m['month']} | {m['model']} | {_fmt(m['balanced_accuracy'])} | {_fmt(m['balanced_accuracy_28d'])} | "
            f"{_fmt(m['recall'])} | {_fmt(m['max_psi'])} | {m['action']} |"
        )
    lines += ["", "## Events", ""] + [f"- {e}" for e in run.events]
    return "\n".join(lines) + "\n"


def cmd_backtest(args) -> int:
    from .backtest import rolling_backtest

    result = rolling_backtest(load_features())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS / "backtest.md").write_text(backtest_report(result), encoding="utf-8")
    print(backtest_report(result))
    if args.mlflow:
        from . import tracking

        tracking.configure()
        tracking.log_backtest(result)
    return 0


def cmd_monitor(args) -> int:
    from .monitor import simulate_production

    on_model = None
    if args.mlflow:
        from . import tracking

        tracking.configure()
        on_model = tracking.log_model
    bt = REPORTS / "backtest.json"
    family = json.loads(bt.read_text(encoding="utf-8"))["selected_family"] if bt.exists() else "logistic"
    print(f"Model family: {family}" + ("" if bt.exists() else " (default; run `closure-risk backtest` to select)"))
    run = simulate_production(load_features(), on_model=on_model, family=family)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.md").write_text(monitoring_report(run), encoding="utf-8")
    run.champion.save(MODELS / "champion")
    print(monitoring_report(run))
    print(f"Champion {run.champion.card.version} saved to {MODELS / 'champion'}")
    if args.mlflow:
        tracking.log_monitoring(run.months, run.events, REPORTS)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="closure-risk")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("simulate", help="generate the synthetic dataset").set_defaults(func=cmd_simulate)
    for name, func, help_text in [
        ("backtest", cmd_backtest, "rolling-origin backtest vs baselines"),
        ("monitor", cmd_monitor, "simulate production: drift, retraining, promotion"),
    ]:
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--mlflow", action="store_true", help="log runs and models to MLflow")
        s.set_defaults(func=func)
    s = sub.add_parser("serve", help="serve the champion model over HTTP")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
