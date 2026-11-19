
# TP03 — MLflow Experiment Tracking: Summary

## 1. Experiment Setup

- Project: Telco Customer Churn
- Experiment: `telco-churn-tp03`
- MLflow version: 3.18.0
- Tracking server: http://127.0.0.1:5000
- Backend: SQLite (`mlflow.db`)
- Artifact storage: `mlartifacts/`

## 2. Model Comparison

All models used the same six features and stratified
60/20/20 split, with random seed 42.

Positive class: Churn = Yes (1).
Selection metric: Validation Average Precision.

| Model | Validation AP | Recall | F1 |
|---|---:|---:|---:|
| Random Forest | 0.6579 | 0.5187 | 0.5826 |
| Logistic Regression | 0.6475 | 0.5294 | 0.5893 |
| Decision Tree | 0.6033 | 0.6016 | 0.6156 |
| Majority Baseline | 0.2654 | 0.0000 | 0.0000 |

### Run IDs

- Random Forest: `e61f74b0850f4701ba48f64f76836fc0`
- Logistic Regression: `9a0ea52e568b4b7da72806f17584e8bc`
- Decision Tree: `23a978a8bd074ad786532dffd750c5a6`
- Majority Baseline: `bceb2108824c486a9e486bcbd9521862`

### Selection Decision

Random Forest was selected because it achieved the highest
validation Average Precision (0.6579).

Logistic Regression obtained slightly higher accuracy,
while Decision Tree obtained higher recall and F1.

Model selection was based on validation only.
The test set was not used.

### Screenshot

Add the MLflow experiment comparison screenshot here.

## 3. Selected Run Evidence

Selected model: Random Forest

Source run ID:
`e61f74b0850f4701ba48f64f76836fc0`

Original dataset SHA-256:
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Git commit:
Copy the actual `git_commit` tag from the selected MLflow run.

Working tree status:
`git_dirty=True` for the initial tracking run.

MLflow recorded:
- Model hyperparameters
- Validation AP, accuracy, precision, recall and F1
- Training and validation dataset metadata
- Confusion matrix and provenance JSON
- Source dataset hash and Git identity

### Screenshot

Add the selected-run details and dataset-input screenshots here.

## 4. Model Registry

Registered model: `telco-churn`

Registered version: `1`

Alias: `candidate`

Logged model URI:
`models:/m-a66eee7f4e4f49efa736b15770ebf8da`

Registry inference URI:
`models:/telco-churn@candidate`

The registered model contains the complete fitted preprocessing
and Random Forest classifier.

### Screenshot

Add the Model Registry screenshot showing version 1 and
the candidate alias here.

## 5. Fresh-Process Inference

Command:

    uv run python scripts/mlflow_predict.py

Predictions:

    [1, 0, 0, 1, 0]

Results:
- Original and registered class predictions matched.
- Prediction probabilities matched within tolerance.
- No retraining was performed.

## 6. Server Restart Test

The server was stopped and restarted with the same SQLite
database and artifact storage locations.

After restart:
- The registered model remained accessible.
- The candidate alias resolved successfully.
- Predictions were identical to the original results.

## 7. Storage and Reproducibility

The SQLite database preserves experiment records, parameters,
metrics, tags, dataset metadata and registry information.

The artifact store preserves logged files and models.

If only the database survives, the model files cannot be
recovered from it alone.

If only the artifact files survive, the experiment metadata,
registry associations and aliases are missing.

A colleague on another machine also needs the original dataset,
compatible Python dependencies and access to the tracking
database and model artifacts.

The local MLflow server is not accessible through another
computer's localhost address.
