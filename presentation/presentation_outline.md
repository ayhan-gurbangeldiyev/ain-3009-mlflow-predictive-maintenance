# Presentation Outline — 5 minutes, 7 slides

1. **Title & Goal** — Predictive Maintenance: end-to-end ML lifecycle on Azure ML + MLflow. Predict machine failure from sensor readings.
2. **Dataset & Framing** — AI4I 2020, 10,000 rows, 3.4% failure rate, leakage columns dropped, ROC-AUC/PR-AUC/recall over accuracy.
3. **Architecture** — Local training/tuning logs to Azure ML as the MLflow tracking server; Azure ML also hosts the registry and managed endpoint.
4. **Experiments & Tuning** — Baselines plus Optuna random-forest tuning; every run/trial logged with params, metrics, artifacts, and model.
5. **Registry & Deployment** — Best model registered as `PredictiveMaintenanceModel` v1, promoted Staging to Production, deployed to `predmaint-endpoint`.
6. **Monitoring** — Six drifting batches logged back to MLflow; model quality falls as sensor drift grows, creating a retraining signal.
7. **Takeaways** — Lifecycle management matters more than a single model; Azure credit was used for tracking, registry, and endpoint while training stayed local to control cost.
