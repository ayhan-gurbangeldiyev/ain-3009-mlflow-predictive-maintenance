"""Airflow DAG orchestrating the predictive-maintenance ML pipeline.

Matches the course Airflow material (weeks 6-7): a Directed Acyclic Graph of
tasks wired with the ``>>`` operator, using ``BashOperator`` action operators.
The DAG runs the full lifecycle in order:

    validate_data >> train_models >> tune_model >> register_model >> monitor_model

Deployment:
    Copy this file into your Airflow ``dags/`` folder (or point AIRFLOW__CORE__
    DAGS_FOLDER here). Set the two environment variables below so the tasks use
    the project's virtualenv and root:

        PROJECT_HOME=/path/to/PRJ-AyhanGurbangeldiyev-2020053
        VENV_PYTHON=$PROJECT_HOME/.venv/bin/python

Trigger it manually from the Airflow UI or with `airflow dags trigger`.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

# Default to the repository root based on this DAG file location.
# This avoids using a hardcoded local path from the developer's computer.
DEFAULT_PROJECT_HOME = Path(__file__).resolve().parents[2]

PROJECT_HOME = os.getenv(
    "PROJECT_HOME",
    str(DEFAULT_PROJECT_HOME),
)
VENV_PYTHON = os.getenv("VENV_PYTHON", f"{PROJECT_HOME}/.venv/bin/python")

# Every task runs from the project root with src/ on PYTHONPATH.
COMMON_ENV = f"cd {PROJECT_HOME} && PYTHONPATH=src {VENV_PYTHON}"

default_args = {
    "owner": "ayhan",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="predmaint_pipeline",
    description="End-to-end predictive-maintenance ML lifecycle",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # manual / externally triggered
    catchup=False,
    tags=["mlops", "mlflow", "predictive-maintenance"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=f"{COMMON_ENV} src/data_preprocessing.py",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"{COMMON_ENV} src/train_models.py",
    )

    tune_model = BashOperator(
        task_id="tune_model",
        bash_command=f"{COMMON_ENV} src/tune_model.py --trials 20",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"{COMMON_ENV} src/register_model.py",
    )

    monitor_model = BashOperator(
        task_id="monitor_model",
        bash_command=f"{COMMON_ENV} src/monitor_model.py --batches 6",
    )

    validate_data >> train_models >> tune_model >> register_model >> monitor_model
