# English Prompt for Presentation Visual Placement

## Research Basis

Short technical presentations work best with 5-8 slides, one clear message per
slide, large readable visuals, and minimal text. For a 5-minute talk, use roughly
one slide per minute, or 5-7 slides when content is dense. Since this project has
technical charts and dashboard evidence, 8 slides is acceptable if each slide has
one main message and the visuals are not crowded.

References:

- HubSpot: https://blog.hubspot.com/marketing/5-minute-presentation
- Adobe Express: https://www.adobe.com/uk/express/learn/blog/how-many-slides-for-a-presentation
- MIT Lightning Talk Guidelines: https://urtc.mit.edu/lightning_talk_presentation_2025.pdf
- SlideModel: https://slidemodel.com/5-minutes-presentation/

## Prompt To Give The Extension

```text
You are editing an existing technical PowerPoint presentation for an MLOps course project. Do not change the slide order, core content, metrics, project claims, student/instructor information, or the existing story. Your task is to improve the visual placement of screenshots and figures so the deck looks clean, academic, and presentation-ready.

Presentation topic:
Predictive Maintenance - End-to-End ML Lifecycle on Azure ML + MLflow

Course and identity:
Course: AIN-3009 Delivering AI Applications with MLOps
Instructor: Dr. Gökşin Bakır
Student: Ayhan Gurbangeldiyev
Student No: 2020053

Main story:
This is not only a model-training project. It demonstrates the full ML lifecycle:
dataset selection, preprocessing, experiment tracking, model training, hyperparameter tuning, model registry, deployment evidence, Airflow pipeline orchestration, and monitoring/drift analysis.

Keep the presentation at 8 slides. Use the screenshots and figures from the `screenshots/` folder. There are 16 visual evidence files. Place them in the following slide order and priority.

Slide 1 - Cover / Identity
Purpose: Introduce the project.
Do not overcrowd this slide.
Use no dashboard screenshot here unless there is already a small lifecycle graphic.
If adding a visual, use a simple lifecycle strip only:
Dataset -> Training -> Tuning -> Registry -> Deployment -> Monitoring

Slide 2 - Project Objective / Assignment Requirements
Purpose: Show what the instructor asked for.
Use a clean checklist layout with the 5 required objectives:
Experiment Tracking, Training & Tuning, Deployment, Monitoring, Model Registry.
Optional small visual: `05_airflow_dags_home.png` as a small proof thumbnail on the right or bottom-right.
Do not make the screenshot dominant on this slide.

Slide 3 - Dataset & Problem Framing
Purpose: Explain the dataset and why the task is difficult.
Use these visuals:
1. `08_fig_class_imbalance.png` as the main visual.
2. `09_fig_machine_types.png` as a smaller supporting visual.
Main message: AI4I 2020 has 10,000 records and only 3.4% failure rate, so accuracy alone is misleading.

Slide 4 - Data Preparation & EDA
Purpose: Show preprocessing and feature understanding.
Use these visuals:
1. `10_fig_feature_distributions.png` as the main wide visual.
2. `11_fig_correlation_heatmap.png` as a smaller secondary visual.
Main message: sensor patterns such as torque, rotational speed, temperature, and tool wear support predictive maintenance modeling.
Keep labels readable. Do not shrink the feature distribution figure too much.

Slide 5 - MLflow + Azure ML Architecture / Experiment Tracking
Purpose: Show that MLflow tracking and Azure ML lifecycle management were used.
Use these visuals:
1. `01_mlflow_experiments_runs.png` as the main dashboard screenshot.
2. `02_mlflow_best_run.png` as a secondary screenshot.
Main message: MLflow logs experiments, parameters, metrics, artifacts, and best-run evidence.
If there is an architecture diagram already on the slide, keep it and place the screenshots as proof panels.

Slide 6 - Model Training, Metrics & Hyperparameter Tuning
Purpose: Show model comparison and tuning results.
Use these visuals:
1. `12_fig_model_comparison_metrics.png` as the main chart.
2. `13_fig_optuna_tuning_metrics.png` as the secondary chart.
Optional small supporting thumbnails if space allows:
3. `14_fig_confusion_matrix_best.png`
4. `15_fig_roc_curve_best_model.png`
Main message: baseline models were compared and Optuna tuning improved the selected model. Highlight ROC-AUC, PR-AUC, F1, recall, and best CV ROC-AUC around 0.974.
Do not put all four visuals at the same size. Model comparison and Optuna history are primary.

Slide 7 - Registry, Deployment & Airflow Pipeline
Purpose: Show model lifecycle management and pipeline automation.
Use these visuals:
1. `03_mlflow_model_registry.png`
2. `03_mlflow_model_version.png`
3. `07_airflow_predmaint_graph.png`
Optional supporting visual:
4. `06_airflow_predmaint_grid.png`
Main message: the model is registered as `PredictiveMaintenanceModel`, promoted through lifecycle stages, and the Airflow DAG shows the pipeline:
validate_data -> train_models -> tune_model -> register_model -> monitor_model.
Make `07_airflow_predmaint_graph.png` prominent because it shows all Airflow steps in green success state.
Do not emphasize failed/old run history. The graph success screenshot is the clean evidence.

Slide 8 - Monitoring, Drift & Conclusion
Purpose: Close the lifecycle loop.
Use these visuals:
1. `16_fig_drift_monitoring_metrics.png` as the main chart.
2. `04_mlflow_monitoring_run.png` as a secondary proof screenshot.
Main message: incoming batch drift was simulated, metrics were logged, ROC-AUC/F1 degraded, and predicted failure rate increased, showing a retraining signal.
End with a small checklist:
Tracking: completed
Training/Tuning: completed
Registry: completed
Deployment: completed
Monitoring: completed

Design rules:
- Preserve all existing slide titles and core text unless minor wording is needed for readability.
- Do not invent new results, screenshots, metrics, or Azure claims.
- Do not remove required course/student/instructor information.
- Keep each slide focused on one message.
- Use large, readable visuals. Avoid placing more than 2 main visuals at equal weight on a slide.
- Use consistent colors: dark navy/charcoal for text, blue for MLflow/Azure, green for success/completed states, red only for failure/drift warning.
- Use subtle captions under screenshots, for example: "MLflow experiment runs", "Registered model version", "Airflow DAG success view", "Monitoring drift metrics".
- Avoid decorative clutter, heavy animations, and large blocks of text.
- Keep the deck suitable for a 5-minute presentation. If extra visuals do not fit cleanly, make them smaller proof thumbnails rather than changing the slide order.
- Export-ready result should work as both PowerPoint and PDF.
```

## Recommended Visual Priority

Essential visuals:

- `08_fig_class_imbalance.png`
- `12_fig_model_comparison_metrics.png`
- `13_fig_optuna_tuning_metrics.png`
- `03_mlflow_model_registry.png`
- `03_mlflow_model_version.png`
- `07_airflow_predmaint_graph.png`
- `16_fig_drift_monitoring_metrics.png`
- `04_mlflow_monitoring_run.png`

Supporting visuals:

- `09_fig_machine_types.png`
- `10_fig_feature_distributions.png`
- `11_fig_correlation_heatmap.png`
- `14_fig_confusion_matrix_best.png`
- `15_fig_roc_curve_best_model.png`
- `01_mlflow_experiments_runs.png`
- `02_mlflow_best_run.png`
- `05_airflow_dags_home.png`
- `06_airflow_predmaint_grid.png`

## Assumption

The deck remains 8 slides. The extension should only improve layout, hierarchy,
and screenshot placement; it should not rewrite the project or change the
narrative.
