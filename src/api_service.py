"""FastAPI serving layer for the predictive-maintenance model.

This wraps the registered MLflow model behind a documented REST API, matching
the course material on REST/FastAPI/OpenAPI (week 2) and API security
(week 3). Requests must carry a valid API key in the ``X-API-Key`` header.
Interactive OpenAPI docs are served at ``/docs``.

Run locally:
    PYTHONPATH=src API_KEY=secret-key .venv/bin/uvicorn api_service:app --port 8000
"""

from __future__ import annotations

import json
import os

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import REGISTERED_MODEL_NAME, REPORTS_DIR, get_tracking_uri

API_KEY = os.getenv("API_KEY", "predmaint-demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def resolve_model_uri() -> str:
    """Use the registered version recorded by register_model.py if available."""
    override = os.getenv("MODEL_URI")
    if override:
        return override
    summary = REPORTS_DIR / "registered_model.json"
    if summary.exists():
        version = json.loads(summary.read_text())["version"]
        return f"models:/{REGISTERED_MODEL_NAME}/{version}"
    return f"models:/{REGISTERED_MODEL_NAME}/Production"


class MachineReading(BaseModel):
    """One machine's sensor reading. Aliases match the training feature names."""

    air_temperature: float = Field(..., alias="Air temperature [K]", example=302.3)
    process_temperature: float = Field(..., alias="Process temperature [K]", example=311.5)
    rotational_speed: int = Field(..., alias="Rotational speed [rpm]", example=1380)
    torque: float = Field(..., alias="Torque [Nm]", example=62.4)
    tool_wear: int = Field(..., alias="Tool wear [min]", example=210)
    type: str = Field(..., alias="Type", example="L")

    class Config:
        populate_by_name = True


class Prediction(BaseModel):
    failure_predicted: int
    failure_probability: float


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings (MLflow model).",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    app.state.model = mlflow.sklearn.load_model(resolve_model_uri())


def require_api_key(key: str = Security(api_key_header)) -> None:
    """Reject requests without the correct API key (week 3: API keys)."""
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict(reading: MachineReading) -> Prediction:
    row = pd.DataFrame([reading.model_dump(by_alias=True)])
    proba = float(app.state.model.predict_proba(row)[0, 1])
    return Prediction(failure_predicted=int(proba >= 0.5), failure_probability=proba)
