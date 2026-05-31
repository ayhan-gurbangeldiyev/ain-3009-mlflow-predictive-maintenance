"""Train and compare baseline models, logging everything to MLflow.

Each model is trained inside a single MLflow run with its parameters, a full
set of classification metrics, a confusion-matrix artifact, and the serialized
scikit-learn pipeline. Results are also written to ``reports/baseline_results.csv``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from mlflow.models import infer_signature

from config import EXPERIMENT_NAME, RANDOM_STATE, REPORTS_DIR, get_tracking_uri
from data_preprocessing import load_and_split, make_pipeline


def build_models() -> dict:
    """Return the baseline estimators. ``class_weight`` handles the imbalance."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(pipeline, X_test, y_test) -> dict:
    """Compute the headline classification metrics on the test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def _log_confusion_matrix(y_test, y_pred, name: str) -> None:
    """Render and log a confusion-matrix PNG as an MLflow artifact."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"confusion_matrix_{name}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(out))


def main(data_path: str | None) -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split(data_path)
    results = []

    for name, estimator in build_models().items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_train, y_train)

            metrics = evaluate(pipeline, X_test, y_test)
            y_pred = pipeline.predict(X_test)

            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model_family", name)
            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_text(
                classification_report(y_test, y_pred, zero_division=0),
                "classification_report.txt",
            )
            _log_confusion_matrix(y_test, y_pred, name)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}")
            results.append({"model": name, **metrics})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "baseline_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nBaseline results written to {out_csv}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--data", default=None, help="Path to ai4i2020.csv")
    args = parser.parse_args()
    main(args.data)
