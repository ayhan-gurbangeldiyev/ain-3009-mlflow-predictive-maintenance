"""Central configuration for the Predictive Maintenance MLflow project.

Tracking can target either an Azure Machine Learning workspace (recommended)
or a local SQLite store as an offline fallback. The Azure ML MLflow tracking
URI is read from the ``AZURE_ML_TRACKING_URI`` environment variable, which is
populated by ``azure/setup_workspace.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project paths.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
REPORTS_DIR = ROOT_DIR / "reports"
ENV_PATH = ROOT_DIR / ".env"

# Load environment variables from the project .env file if present.
load_dotenv(ENV_PATH)

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "predictive-maintenance")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "PredictiveMaintenanceModel")

# Random seed for reproducibility across the whole project.
RANDOM_STATE = 42


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI.

    Prefers the Azure ML workspace URI; falls back to a local SQLite database
    so the pipeline still runs without a cloud connection.
    """
    azure_uri = os.getenv("AZURE_ML_TRACKING_URI")
    if azure_uri:
        return azure_uri
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{ROOT_DIR / 'mlflow.db'}"


def is_azure() -> bool:
    """True when an Azure ML tracking URI is configured."""
    return bool(os.getenv("AZURE_ML_TRACKING_URI"))
