"""Visualize modelling results for the slide deck.

Reads the CSV/JSON outputs produced by the pipeline and the registered model,
then renders: baseline model comparison, Optuna optimization history, the best
model's confusion matrix and ROC curve, and the monitoring drift curve. All
figures are written to ``reports/figures/``.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from config import REGISTERED_MODEL_NAME, REPORTS_DIR, get_tracking_uri
from data_preprocessing import load_and_split

FIG_DIR = REPORTS_DIR / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def model_comparison() -> None:
    df = pd.read_csv(REPORTS_DIR / "baseline_results.csv")
    metrics = ["roc_auc", "pr_auc", "f1", "recall"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Baseline model comparison")
    ax.legend()
    _save(fig, "05_model_comparison.png")


def optuna_history() -> None:
    df = pd.read_csv(REPORTS_DIR / "tuning_results.csv")
    if "value" not in df.columns:
        print("no 'value' column in tuning_results.csv; skipping history")
        return
    values = df["value"].to_numpy()
    running_best = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, "o", alpha=0.5, label="trial CV ROC-AUC")
    ax.plot(running_best, "-", color="#e53935", label="running best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Optuna optimization history")
    ax.legend()
    _save(fig, "06_optuna_history.png")


def best_model_diagnostics() -> None:
    summary = json.loads((REPORTS_DIR / "registered_model.json").read_text())
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{summary['version']}"
    mlflow.set_tracking_uri(get_tracking_uri())
    model = mlflow.sklearn.load_model(model_uri)

    _, X_test, _, y_test = load_and_split()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues",
                                                     values_format="d", ax=ax)
    ax.set_title("Best model — confusion matrix")
    _save(fig, "07_confusion_matrix_best.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Best model — ROC curve")
    _save(fig, "08_roc_curve.png")


def drift_monitoring() -> None:
    df = pd.read_csv(REPORTS_DIR / "monitoring_metrics.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["batch"], df["roc_auc"], "o-", color="#1e88e5", label="ROC-AUC")
    ax1.plot(df["batch"], df["f1"], "s-", color="#43a047", label="F1")
    ax1.set_xlabel("Incoming batch (increasing drift →)")
    ax1.set_ylabel("Model quality")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(df["batch"], df["predicted_failure_rate"], "d--", color="#e53935",
             label="predicted failure rate")
    ax2.set_ylabel("Predicted failure rate")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax1.set_title("Production monitoring — metric drift")
    _save(fig, "09_drift_monitoring.png")


def main() -> None:
    model_comparison()
    optuna_history()
    best_model_diagnostics()
    drift_monitoring()
    print("Result visualizations complete.")


if __name__ == "__main__":
    main()
