# Speaker Script — ~5 minutes

**Slide 1 — Cover / Identity (20s).**
"Hi, I'm Ayhan Gurbangeldiyev. This is my AIN-3009 MLOps term project for Dr.
Gökşin Bakır. The project is Predictive Maintenance: an end-to-end ML lifecycle
managed with MLflow and Azure Machine Learning."

**Slide 2 — Project Objective (35s).**
"The assignment asks for more than a trained model. We need to demonstrate
experiment tracking, model training and tuning, deployment, monitoring, and model
registry lifecycle. I use this slide as the checklist for the whole presentation:
each following section maps to one of these requirements."

**Slide 3 — Dataset & Problem Framing (40s).**
"I used the AI4I 2020 predictive-maintenance dataset with 10,000 telemetry
records. The target is binary machine failure. The important issue is class
imbalance: only 3.4 percent of records are failures, so accuracy alone is
misleading. I also removed the failure-mode columns because they leak the label."

**Slide 4 — Data Preparation & EDA (40s).**
"For preprocessing, numeric sensor features are scaled and machine type is
one-hot encoded. This preprocessing is bundled with the estimator inside a
scikit-learn Pipeline, so serving uses the same transformations as training. The
EDA plots show that failure and non-failure cases separate in some sensor
patterns, especially torque, speed, temperature, and tool wear."

**Slide 5 — Architecture (40s).**
"The architecture is local training with cloud lifecycle management. Training and
tuning run locally to control cost. MLflow points to the Azure ML workspace, so
runs, metrics, artifacts, registered models, and endpoint state are managed in
Azure. This is where the Azure credit was used: tracking, registry, and managed
endpoint deployment."

**Slide 6 — Training, Tracking & Tuning (50s).**
"I trained three baseline models: logistic regression, random forest, and
gradient boosting. Each run logs parameters, metrics, reports, confusion matrix,
and the model artifact to MLflow. Then I tuned a random forest with Optuna for 20
trials. The best cross-validated ROC-AUC is about 0.974, and recall improves from
0.44 to around 0.77."

**Slide 7 — Registry, Deployment & Pipeline Automation (45s).**
"The best run is registered as `PredictiveMaintenanceModel` version 1 and
promoted through Staging and Production. The model is deployed through an Azure
ML Managed Online Endpoint named `predmaint-endpoint`, verified as Succeeded. I
also added an Airflow DAG to orchestrate the pipeline: validate, train, tune,
register, and monitor."

**Slide 8 — Monitoring & Conclusion (35s).**
"Finally, monitoring closes the loop. I simulate incoming batches with increasing
sensor drift and log the batch metrics back to MLflow. ROC-AUC drops while the
predicted failure rate rises, which is exactly the kind of signal that would
trigger retraining. So the final result is not just a notebook model; it is an
end-to-end MLflow and Azure ML lifecycle demonstration."
