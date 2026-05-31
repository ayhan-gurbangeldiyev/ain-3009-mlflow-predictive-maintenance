# Speaker Script — ~5 minutes

**Slide 1 — Title & Goal (30s).**
"Hi, I'm Ayhan. This project builds an end-to-end ML lifecycle for predictive
maintenance using MLflow and Azure Machine Learning. The goal is to predict
machine failure from sensor readings, then manage the model through tracking,
tuning, registry, deployment, and monitoring."

**Slide 2 — Dataset & Framing (45s).**
"I used the AI4I 2020 predictive-maintenance dataset with 10,000 machine
telemetry rows. The target is `Machine failure`, but only 3.4% of rows are
positive, so accuracy is misleading. I removed the per-failure-mode columns
because they leak the label, and I evaluate mainly with ROC-AUC, PR-AUC, recall,
and F1."

**Slide 3 — Architecture (45s).**
"Azure ML is the remote MLflow server. The code runs locally for training and
tuning to avoid unnecessary compute cost, but `mlflow.set_tracking_uri` points to
the Azure ML workspace. That means experiments, artifacts, registered models, and
the managed endpoint are all visible in Azure ML Studio."

**Slide 4 — Experiments & Tuning (60s).**
"I trained logistic regression, random forest, and gradient boosting baselines.
Each run logs parameters, six metrics, a confusion matrix, a classification
report, and the serialized scikit-learn pipeline. Then I tuned a random forest
with Optuna for 20 trials, and every trial is a nested MLflow run. The best
configuration reached about 0.974 cross-validated ROC-AUC and improved recall
from 0.44 to around 0.77."

**Slide 5 — Registry & Deployment (60s).**
"The best run is selected by ROC-AUC and registered as
`PredictiveMaintenanceModel` version 1. The script promotes it through Staging
and Production, giving a clean lifecycle state instead of relying on a local file
path. The production model is deployed to an Azure ML Managed Online Endpoint
named `predmaint-endpoint`, which was verified as successfully provisioned."

**Slide 6 — Monitoring (45s).**
"For monitoring, I simulate six incoming batches with progressively stronger
sensor drift. The model's ROC-AUC and F1 drop while the predicted failure rate
inflates. Those trends are logged back to MLflow with a batch step, showing the
kind of signal that would trigger retraining in production."

**Slide 7 — Takeaways (35s).**
"The main takeaway is that MLOps is the lifecycle, not just the model. MLflow
tracks the experiments, Azure ML makes the tracking and registry shareable, the
endpoint demonstrates serving, and monitoring closes the loop. Azure credit was
used where it matters most for the course: tracking, registry, and managed
deployment, while local training kept cost low."
