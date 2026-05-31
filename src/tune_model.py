"""Hyperparameter tuning with Optuna, with every trial logged to MLflow.

Optuna searches the Random Forest hyperparameter space and maximizes the
mean cross-validated ROC-AUC. Each trial becomes a nested MLflow run, and the
best refitted pipeline is logged as a dedicated parent-level run with test
metrics and the serialized model.
"""

from __future__ import annotations

import argparse

import mlflow
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from mlflow.models import infer_signature

from config import EXPERIMENT_NAME, RANDOM_STATE, REPORTS_DIR, get_tracking_uri
from data_preprocessing import load_and_split, make_pipeline
from train_models import evaluate


def main(data_path: str | None, n_trials: int) -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split(data_path)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            estimator = RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, **params
            )
            pipeline = make_pipeline(estimator)
            score = cross_val_score(
                pipeline, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1
            ).mean()
            mlflow.set_tag("stage", "tuning")
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", float(score))
        return score

    with mlflow.start_run(run_name="optuna_tuning"):
        mlflow.set_tag("stage", "tuning_parent")
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", float(study.best_value))

        # Refit the best configuration on the full training set.
        best_estimator = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, **best_params
        )
        best_pipeline = make_pipeline(best_estimator)
        best_pipeline.fit(X_train, y_train)

        test_metrics = evaluate(best_pipeline, X_test, y_test)
        mlflow.log_metrics(test_metrics)

        signature = infer_signature(X_train, best_pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        print(f"Best CV ROC-AUC: {study.best_value:.4f}")
        print(f"Best params: {best_params}")
        print(f"Test metrics: {test_metrics}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "tuning_results.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"Tuning results written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the model with Optuna.")
    parser.add_argument("--data", default=None, help="Path to ai4i2020.csv")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    args = parser.parse_args()
    main(args.data, args.trials)
