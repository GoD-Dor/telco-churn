# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predict customer churn using machine learning

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         telco_churn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── telco_churn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes telco_churn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------



---

# TP02 — Reusable Churn Prediction Pipeline

## Objective

The objective of TP02 is to build a reproducible machine learning
workflow for predicting customer churn using the IBM Telco dataset.

The project compares multiple classifiers using the same data split
and preprocessing steps, then saves the selected fitted pipeline
for reuse.

## Dataset

The dataset contains 7,043 customers.

Target variable:
- Churn = Yes (1): customer leaves
- Churn = No (0): customer stays

Input features:

Numeric:
- tenure
- MonthlyCharges
- TotalCharges

Categorical:
- Contract
- InternetService
- PaymentMethod

The original CSV is stored locally at `data/raw/telco.csv`
and is excluded from Git.

## Environment Setup

Install uv and run the following commands from the project root:

    uv sync --locked

## Training

Run the training workflow:

    uv run python scripts/first_model.py

The script:
1. Loads the original dataset.
2. Creates stratified train/validation/test splits (60/20/20).
3. Trains a baseline and three classifiers.
4. Applies preprocessing through a scikit-learn Pipeline.
5. Evaluates candidates on validation data.
6. Selects the best model using Average Precision.
7. Saves the fitted pipeline and metadata.
8. Runs basic reliability checks.

The test set is reserved and is not used for model selection.

## Preprocessing

Numeric features:
- Median imputation for missing values
- StandardScaler for feature scaling

Categorical features:
- Most-frequent imputation
- OneHotEncoder with handle_unknown="ignore"

Preprocessing statistics are learned from training data only.

## Model Comparison

| Model | Average Precision | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.6579 | 0.8027 | 0.5187 | 0.5826 |
| Logistic Regression | 0.6475 | 0.8041 | 0.5294 | 0.5893 |
| Decision Tree | 0.6033 | 0.8006 | 0.6016 | 0.6156 |
| Majority Baseline | 0.2654 | 0.7346 | 0.0000 | 0.0000 |

Random Forest was selected because it achieved the highest
validation Average Precision.

## Model Inference

After running the training script:

    uv run python scripts/predict.py

This script loads the saved pipeline from:

    models/selected_pipeline.joblib

It predicts on five saved validation examples and verifies that
predictions and probabilities match those produced before saving.

No retraining is performed during inference.

## Reproducibility

- Random seed: 42
- Training customers: 4,225
- Validation customers: 1,409
- Test customers: 1,409
- Positive class: Churn = Yes (1)
- Classification threshold: 0.5
- Selection metric: Average Precision

Dataset SHA-256:

    16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

The environment is defined in `pyproject.toml` and `uv.lock`.

## Saved Outputs

- `reports/model_comparison.json`: validation comparison results
- `models/selected_pipeline_metadata.json`: model configuration and metadata
- `models/selected_pipeline.joblib`: fitted pipeline (generated locally)
- `reports/split_allocations.json`: customer split allocations (local only)

## Limitations

The model was evaluated on a single validation split.
Its predicted probabilities have not been calibrated.
Handling unknown categories does not guarantee accurate predictions
for unseen customer populations.

Future improvements could include cross-validation,
hyperparameter optimization, probability calibration,
and experiment tracking.
