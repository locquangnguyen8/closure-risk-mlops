"""MLflow experiment tracking and model registry.

Every trained model is logged as a run (parameters, metrics, model card and the
model itself) and registered. The serving alias `champion` always points at the
model in production, so promotions and rollbacks are a registry change.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from .model import ClosureRiskModel

EXPERIMENT = "closure-risk"
REGISTERED_MODEL = "closure-risk-7d"


def configure(tracking_uri: str | None = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)


def log_backtest(result: dict) -> None:
    with mlflow.start_run(run_name="rolling-backtest"):
        mlflow.set_tag("stage", "backtest")
        for model_name, metrics in result["overall"].items():
            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("selected_family", result["selected_family"])
        for i, fold in enumerate(result["folds"]):
            for family in ("logistic", "xgboost"):
                mlflow.log_metric(f"{family}_fold_balanced_accuracy", fold[family], step=i)
        mlflow.log_dict(result, "backtest.json")


def _set_champion(client: MlflowClient, version: str) -> None:
    """Point the `champion` alias at `version` and mark the model it replaces as retired."""
    try:
        previous = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    except mlflow.exceptions.MlflowException:
        previous = None
    if previous and previous != version:
        client.set_model_version_tag(REGISTERED_MODEL, previous, "role", "retired")
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
    client.set_model_version_tag(REGISTERED_MODEL, version, "role", "champion")


_registry_versions: dict[str, str] = {}  # our model version -> MLflow registry version


def log_model(model: ClosureRiskModel, role: str) -> str:
    """Log and register a new model, or, for a model already registered (a promoted
    challenger), just move the `champion` alias to it. Returns the registry version."""
    client = MlflowClient()
    if model.card.version in _registry_versions:
        version = _registry_versions[model.card.version]
        if role == "champion":
            _set_champion(client, version)
        return version
    with mlflow.start_run(run_name=f"{model.card.version}-{role}") as run:
        mlflow.set_tags({"stage": "training", "role": role, "model_version": model.card.version})
        mlflow.log_params(
            model.card.params
            | {
                "family": model.card.family,
                "threshold": model.card.threshold,
                "trained_through": model.card.trained_through,
            }
        )
        mlflow.log_metrics({"n_training_rows": model.card.n_training_rows, "positive_rate": model.card.positive_rate})
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            mlflow.log_artifacts(tmp, artifact_path="model_files")
        if model.booster is not None:
            mlflow.xgboost.log_model(model.booster, name="model")
        else:
            mlflow.sklearn.log_model(model.sklearn_model, name="model")
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL)
    _registry_versions[model.card.version] = mv.version
    if role == "champion":
        _set_champion(client, mv.version)
    else:
        client.set_model_version_tag(REGISTERED_MODEL, mv.version, "role", role)
    return mv.version


def log_monitoring(run_months: list[dict], events: list[str], report_dir: Path) -> None:
    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.set_tag("stage", "monitoring")
        for i, m in enumerate(run_months):
            mlflow.log_metrics({"live_balanced_accuracy": m["balanced_accuracy"], "max_psi": m["max_psi"]}, step=i)
        mlflow.log_text("\n".join(events), "events.txt")
        mlflow.log_text(json.dumps(run_months, indent=2, default=str), "months.json")
        if (report_dir / "monitoring.md").exists():
            mlflow.log_artifact(str(report_dir / "monitoring.md"))
