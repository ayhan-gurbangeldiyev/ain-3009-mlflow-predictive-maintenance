# Azure Evidence

Collected on 2026-05-31 with Azure CLI.

## Workspace

- Resource group: `rg-mlops-predmaint`
- Workspace: `mlw-predmaint`
- Location: `swedencentral`
- MLflow tracking URI scheme: `azureml://...`

## Model Registry

- Registered model: `PredictiveMaintenanceModel`
- Version: `1`
- Created at: `2026-05-31T09:01:37.783806+00:00`

## Managed Online Endpoint

- Endpoint name: `predmaint-endpoint`
- Provisioning state: `Succeeded`

Endpoint smoke-test files:

- `azure/sample_request.json` contains the non-secret sample payload.
- `src/serve_test.py` reads `SCORING_URI` and `SCORING_KEY` from the ignored
  local `.env` file and posts the same payload to `/score`.

During packaging, the endpoint credentials were fetched into the ignored local
`.env`, but direct scoring from this environment returned `ConnectionResetError`
both through `serve_test.py` and `az ml online-endpoint invoke`. The Azure
resource itself still reports `Succeeded`, so the remaining demo check is to run
the same command from a network path that can reach the endpoint or to show the
endpoint status in Azure ML Studio.

## Studio Screenshots To Include In Presentation

Add Azure ML Studio screenshots here if the presentation requires visual proof:

- `01_experiments_runs.png`: Experiment run list for `predictive-maintenance`
- `02_best_run_metrics.png`: Best run metrics and artifacts
- `03_model_registry.png`: `PredictiveMaintenanceModel` version 1
- `04_endpoint.png`: `predmaint-endpoint` deployment status

Do not include endpoint keys or `.env` contents in screenshots.
