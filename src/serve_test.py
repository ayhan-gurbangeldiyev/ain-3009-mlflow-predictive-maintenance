"""Send a sample request to the deployed Azure ML online endpoint.

Reads ``SCORING_URI`` and ``SCORING_KEY`` from ``.env`` (written by
``azure/deploy_endpoint.py``) and posts one machine reading in the
``input_data`` format expected by Azure ML MLflow deployments.
"""

from __future__ import annotations

import argparse
import json
import os

import requests
from dotenv import load_dotenv

from config import ENV_PATH

load_dotenv(ENV_PATH)

# A single representative machine reading (matches the training feature names).
SAMPLE = {
    "Air temperature [K]": 302.3,
    "Process temperature [K]": 311.5,
    "Rotational speed [rpm]": 1380,
    "Torque [Nm]": 62.4,
    "Tool wear [min]": 210,
    "Type": "L",
}


def build_payload() -> dict:
    """Build the split-oriented payload that Azure ML MLflow scoring expects."""
    return {
        "input_data": {
            "columns": list(SAMPLE.keys()),
            "index": [0],
            "data": [list(SAMPLE.values())],
        }
    }


def main(url: str | None, key: str | None) -> None:
    url = url or os.getenv("SCORING_URI")
    key = key or os.getenv("SCORING_KEY")
    if not url or not key:
        raise SystemExit("SCORING_URI/SCORING_KEY missing. Deploy the endpoint first.")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    response = requests.post(url, headers=headers, data=json.dumps(build_payload()))
    response.raise_for_status()

    prediction = response.json()
    print("Request sample:", json.dumps(SAMPLE, indent=2))
    print("Prediction (1 = failure predicted):", prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the online endpoint.")
    parser.add_argument("--url", default=None, help="Scoring URI override")
    parser.add_argument("--key", default=None, help="Scoring key override")
    args = parser.parse_args()
    main(args.url, args.key)
