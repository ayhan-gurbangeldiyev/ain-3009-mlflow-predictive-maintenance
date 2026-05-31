"""Deploy the registered model to an Azure ML Managed Online Endpoint.

This performs a no-code deployment: because the model is logged in the MLflow
flavor, Azure ML automatically builds the scoring environment and inference
script. After deployment the scoring URI and primary key are written back to
``.env`` so ``src/serve_test.py`` can call the live endpoint.

Run ``python azure/deploy_endpoint.py`` after the model has been registered.
Remember to run with ``--delete`` once the demo is finished to stop billing.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "predmaint-endpoint")
DEPLOYMENT_NAME = "blue"
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "PredictiveMaintenanceModel")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "Standard_DS2_v2")


def get_client() -> MLClient:
    """Build an MLClient from the .env credentials."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE"],
    )


def latest_version(client: MLClient, name: str) -> str:
    """Return the highest registered version number for a model."""
    versions = [int(m.version) for m in client.models.list(name=name)]
    if not versions:
        raise RuntimeError(f"Model '{name}' is not registered yet.")
    return str(max(versions))


def append_env(scoring_uri: str, key: str) -> None:
    """Append the scoring URI and key to .env (without duplicating)."""
    lines = [
        line
        for line in (ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else [])
        if not line.startswith(("SCORING_URI=", "SCORING_KEY="))
    ]
    lines += [f"SCORING_URI={scoring_uri}", f"SCORING_KEY={key}"]
    ENV_PATH.write_text("\n".join(lines) + "\n")


def deploy() -> None:
    client = get_client()
    version = latest_version(client, MODEL_NAME)
    print(f"Deploying {MODEL_NAME}:{version} to endpoint '{ENDPOINT_NAME}'...")

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Predictive maintenance failure classifier",
        auth_mode="key",
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=f"azureml:{MODEL_NAME}:{version}",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    )
    client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% of traffic to the new deployment.
    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    time.sleep(5)

    scoring_uri = client.online_endpoints.get(ENDPOINT_NAME).scoring_uri
    keys = client.online_endpoints.get_keys(ENDPOINT_NAME)
    append_env(scoring_uri, keys.primary_key)

    print(f"Endpoint live. Scoring URI: {scoring_uri}")
    print("Scoring URI and key appended to .env")


def delete() -> None:
    client = get_client()
    print(f"Deleting endpoint '{ENDPOINT_NAME}' to stop billing...")
    client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
    print("Endpoint deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the managed online endpoint.")
    parser.add_argument(
        "--delete", action="store_true", help="Delete the endpoint instead of creating"
    )
    args = parser.parse_args()
    delete() if args.delete else deploy()
