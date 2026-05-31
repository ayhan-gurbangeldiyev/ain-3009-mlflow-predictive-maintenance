# Presentation Outline — 5 minutes, 8 slides

1. **Cover / Identity** — Predictive Maintenance: End-to-End ML Lifecycle on Azure ML + MLflow. Course: AIN-3009 Delivering AI Applications with MLOps. Instructor: Dr. Gökşin Bakır. Student: Ayhan Gurbangeldiyev, 2020053.
2. **Project Objective** — Map the assignment requirements: experiment tracking, training/tuning, deployment, monitoring, and model registry.
3. **Dataset & Problem Framing** — AI4I 2020, 10,000 telemetry records, binary `Machine failure`, 3.4% failure rate, leakage columns removed.
4. **Data Preparation & EDA** — StandardScaler, OneHotEncoder, scikit-learn Pipeline, feature distributions, correlation heatmap.
5. **Architecture** — Local training/tuning logs to Azure ML as the remote MLflow tracking server; Azure ML hosts registry and managed endpoint.
6. **Training, Tracking & Tuning** — Baseline models, MLflow logging, Optuna 20 trials, best CV ROC-AUC 0.974, recall 0.44 to 0.77.
7. **Registry, Deployment & Pipeline Automation** — `PredictiveMaintenanceModel` v1, Staging to Production, `predmaint-endpoint` Succeeded, Airflow DAG.
8. **Monitoring & Conclusion** — Drift simulation, metric degradation, retraining signal, final course-objective checklist.
