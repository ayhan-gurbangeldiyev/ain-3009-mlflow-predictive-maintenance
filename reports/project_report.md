# Project Report — Predictive Maintenance ML Lifecycle on Azure ML + MLflow

**Course:** AIN-3009 Delivering AI Applications with MLOps
**Author:** Ayhan Gurbangeldiyev (2020053)
**Repository:** end-to-end MLflow lifecycle managed through Azure Machine Learning

## 1. Problem and Domain

Industrial equipment failure causes unplanned downtime that is expensive and
disruptive. **Predictive maintenance** uses sensor telemetry to predict failures
before they occur, so maintenance can be scheduled proactively. This project
frames the task as **binary classification**: given a machine's current sensor
readings, predict whether a failure is occurring.

The domain was deliberately chosen because it is the canonical MLOps case study —
it produces a continuous stream of incoming sensor data, models decay as
machinery ages, and predictions must be served in (near) real time. These are
exactly the conditions that make lifecycle management with MLflow worthwhile.

## 2. Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI), 10,000 rows.
- **Target:** `Machine failure` (binary). Failure rate ≈ **3.4%** → strong class
  imbalance, which drives the choice of metrics and `class_weight`.
- **Features:** air temperature, process temperature, rotational speed, torque,
  tool wear (numeric) and machine `Type` ∈ {L, M, H} (categorical).
- **Leakage control:** the per-mode failure flags (TWF, HDF, PWF, OSF, RNF) are
  dropped — they jointly define the target and would leak the label.

Preprocessing is a scikit-learn `ColumnTransformer`: `StandardScaler` for numeric
features and `OneHotEncoder` for `Type`, bundled with each estimator in a single
`Pipeline` so the exact same transformation is serialized with the model.

## 3. Tools and Architecture

| Concern | Tool |
|---|---|
| Experiment tracking | MLflow → **Azure ML workspace** (remote tracking server) |
| Modelling | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Hyperparameter tuning | **Optuna** (TPE sampler), every trial logged to MLflow |
| Model registry | Azure ML Model Registry (Staging → Production) |
| Deployment | Azure ML **Managed Online Endpoint** (no-code MLflow deployment) |
| Monitoring | MLflow metric logging with drift simulation |

The Azure ML workspace **is** an MLflow tracking server. Setting
`mlflow.set_tracking_uri(<workspace MLflow URI>)` routes all runs, parameters,
metrics, artifacts, and registered models into Azure ML Studio with no other code
change. Training and tuning run on the local machine (no compute cost); only the
online endpoint incurs charges, and it is torn down after the demo. If the Azure
URI is absent, the code transparently falls back to local SQLite, keeping the
project fully runnable offline.

## 4. Experiment Tracking and Baseline Models

Three baseline models were trained, each in its own MLflow run with parameters,
six metrics, a confusion-matrix artifact, the classification report, and the
serialized pipeline (with an inferred signature and input example).

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.907 | 0.382 | 0.242 | 0.824 | 0.142 |
| Random Forest | 0.963 | 0.775 | 0.600 | 0.441 | 0.938 |
| Gradient Boosting | **0.970** | **0.801** | **0.756** | 0.662 | 0.882 |

Because of the 3.4% imbalance, **accuracy is misleading** (a trivial model scores
~96.6%). The project therefore selects on **ROC-AUC** and reports **PR-AUC**,
recall, and F1, which reflect real failure-detection ability.

## 5. Hyperparameter Tuning (Optuna)

An Optuna study optimized the Random Forest over `n_estimators`, `max_depth`,
`min_samples_split`, `min_samples_leaf`, and `max_features`, maximizing 3-fold
cross-validated ROC-AUC. **Every trial is a nested MLflow run**; the best refit
is logged as the parent run with full test metrics.

- Best CV ROC-AUC: **0.974**
- Best configuration test ROC-AUC: **0.970**, F1 **0.712**, recall **0.765**

Tuning materially improved recall over the baseline Random Forest (0.44 → 0.77)
while keeping ROC-AUC competitive with Gradient Boosting — important because
recall is what catches real failures.

## 6. Model Registry

`register_model.py` selects the best run by ROC-AUC, registers it as
`PredictiveMaintenanceModel`, and transitions the version through **Staging** and
then **Production**. The promotion logic is portable: it uses MLflow stages where
available and falls back to registry aliases/tags on Azure ML, so the same script
works locally and in the cloud. The registration summary is persisted to
`reports/registered_model.json`.

## 7. Deployment

The production model is deployed to an **Azure ML Managed Online Endpoint** via a
no-code deployment — because the model is logged in the MLflow flavor, Azure ML
auto-builds the scoring environment and inference script. The endpoint exposes a
REST `/score` route; `serve_test.py` posts a sample machine reading in the
`input_data` format and receives a failure prediction.

The Azure deployment evidence collected for the submission is:

- Workspace `mlw-predmaint` in `swedencentral`
- Resource group `rg-mlops-predmaint`
- Registered model `PredictiveMaintenanceModel`, version `1`
- Managed online endpoint `predmaint-endpoint`, provisioning state `Succeeded`

Endpoint scoring URI and key are written only to the ignored local `.env` file,
which is not included in the submission archive. This keeps credentials out of
the deliverable while still allowing the endpoint to be tested locally before the
demo. The endpoint should be deleted after presentation if it is no longer needed
to avoid unnecessary Azure credit usage.

## 8. Performance Monitoring

`monitor_model.py` simulates production traffic by scoring six successive batches
with progressively stronger sensor drift injected into the numeric features. Each
batch's metrics are logged to MLflow with a `step` index, producing a degradation
curve in Azure ML Studio.

| Batch | ROC-AUC | F1 | Predicted failure rate |
|---|---|---|---|
| 0 | 0.914 | 0.640 | 4.2% |
| 2 | 0.799 | 0.218 | 14.4% |
| 5 | 0.766 | 0.092 | 36.9% |

As drift increases, ROC-AUC falls and the predicted failure rate inflates far
above the true rate — precisely the signal that would trigger automated
retraining in a production MLOps loop.

## 9. Insights and Reflection

- **Lifecycle, not just a model.** The model is a small part; tracking, registry,
  serving, and monitoring are what make it operable. MLflow + Azure ML provide all
  four with minimal glue code.
- **Metrics must match the business.** Under heavy imbalance, ROC-AUC/PR-AUC and
  recall tell the truth that accuracy hides.
- **The remote tracking server is the key abstraction.** Pointing MLflow at the
  Azure ML URI turned a local experiment into a cloud-managed, shareable,
  deployable asset without rewriting the training code.
- **Monitoring closes the loop.** Drift simulation makes the case for retraining
  triggers concrete and measurable.

## 10. How to Reproduce

See `README.md` for the full command sequence (workspace provisioning, training,
tuning, registration, deployment, endpoint test, monitoring, teardown).
