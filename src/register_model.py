"""Register the best run's model and move it through staging to production.

The best run is selected by test ROC-AUC across the whole experiment. The model
is registered in the (Azure ML or local) MLflow Model Registry. Stage handling
uses MLflow stages where supported and falls back to registry aliases/tags,
which keeps the script working both locally and against Azure ML.
"""

from __future__ import annotations

import argparse
import json

import mlflow
from mlflow.tracking import MlflowClient

from config import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    get_tracking_uri,
)


def find_best_run(client: MlflowClient, metric: str) -> mlflow.entities.Run:
    """Return the run with the highest value of ``metric``."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")
    # Filter/order client-side: Azure ML MLflow does not reliably support
    # server-side numeric metric filters (e.g. ``metrics.roc_auc > 0``).
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], max_results=500
    )
    candidates = [r for r in runs if metric in r.data.metrics]
    if not candidates:
        raise RuntimeError(f"No runs with metric '{metric}' found.")
    return max(candidates, key=lambda r: r.data.metrics[metric])


def promote(client: MlflowClient, name: str, version: str, stage: str) -> str:
    """Promote a version to a stage, falling back to aliases/tags on Azure ML."""
    try:
        client.transition_model_version_stage(name=name, version=version, stage=stage)
        return f"stage={stage}"
    except Exception as exc:  # Azure ML registry may not support stages.
        print(f"Stage transition unavailable ({exc}); using alias/tag instead.")
        alias = stage.lower()
        try:
            client.set_registered_model_alias(name=name, alias=alias, version=version)
            return f"alias={alias}"
        except Exception:
            client.set_model_version_tag(name, version, "stage", alias)
            return f"tag:stage={alias}"


def main(metric: str, model_name: str) -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    client = MlflowClient()

    best_run = find_best_run(client, metric)
    run_id = best_run.info.run_id
    score = best_run.data.metrics.get(metric)
    print(f"Best run: {run_id} ({metric}={score:.4f})")

    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=model_name).version
    print(f"Registered '{model_name}' version {version}")

    staging_marker = promote(client, model_name, version, "Staging")
    production_marker = promote(client, model_name, version, "Production")

    summary = {
        "model_name": model_name,
        "version": version,
        "source_run_id": run_id,
        "selection_metric": metric,
        "selection_score": score,
        "staging": staging_marker,
        "production": production_marker,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "registered_model.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Registration summary written to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register and promote best model.")
    parser.add_argument("--metric", default="roc_auc", help="Selection metric")
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    args = parser.parse_args()
    main(args.metric, args.model_name)
