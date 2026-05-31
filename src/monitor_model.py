"""Simulate production monitoring and track metric drift in MLflow.

The production model is loaded from the registry and scored against several
incoming batches. Each successive batch has progressively stronger sensor drift
injected into the numeric features, so the logged metrics (and the input-feature
means) visibly degrade over time. Per-batch metrics are logged to MLflow with a
``step`` index and also written to ``reports/monitoring_metrics.csv``.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import (
    EXPERIMENT_NAME,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    get_tracking_uri,
)
from data_preprocessing import NUMERIC_FEATURES, load_and_split


def resolve_model_uri(model_name: str) -> str:
    """Prefer the version recorded by register_model.py; fall back to latest."""
    summary = REPORTS_DIR / "registered_model.json"
    if summary.exists():
        version = json.loads(summary.read_text())["version"]
        return f"models:/{model_name}/{version}"
    return f"models:/{model_name}/latest"


def make_batches(X, y, n_batches: int):
    """Yield drifting batches; drift scales the numeric feature distribution."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(X))
    splits = np.array_split(idx, n_batches)
    for i, split in enumerate(splits):
        Xb = X.iloc[split].copy()
        drift = 1.0 + 0.08 * i  # 0%, 8%, 16%, ... drift per batch.
        noise = rng.normal(1.0, 0.02 * i + 1e-9, size=(len(Xb), len(NUMERIC_FEATURES)))
        Xb[NUMERIC_FEATURES] = Xb[NUMERIC_FEATURES].to_numpy() * drift * noise
        yield i, Xb, y.iloc[split]


def main(model_name: str, n_batches: int) -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = mlflow.sklearn.load_model(resolve_model_uri(model_name))
    _, X_test, _, y_test = load_and_split()

    rows = []
    with mlflow.start_run(run_name="production_monitoring"):
        mlflow.set_tag("stage", "monitoring")
        for step, Xb, yb in make_batches(X_test, y_test, n_batches):
            y_pred = model.predict(Xb)
            y_proba = model.predict_proba(Xb)[:, 1]
            metrics = {
                "recall": float(recall_score(yb, y_pred, zero_division=0)),
                "precision": float(precision_score(yb, y_pred, zero_division=0)),
                "f1": float(f1_score(yb, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(yb, y_proba)),
                "predicted_failure_rate": float(np.mean(y_pred)),
                "actual_failure_rate": float(np.mean(yb)),
                "avg_torque": float(Xb["Torque [Nm]"].mean()),
            }
            mlflow.log_metrics(metrics, step=step)
            rows.append({"batch": step, **metrics})
            print(f"batch {step}: roc_auc={metrics['roc_auc']:.3f} f1={metrics['f1']:.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "monitoring_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Monitoring metrics written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate monitoring with drift.")
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--batches", type=int, default=6, help="Number of batches")
    args = parser.parse_args()
    main(args.model_name, args.batches)
