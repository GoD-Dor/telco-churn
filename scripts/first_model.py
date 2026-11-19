
from pathlib import Path
import pandas as pd

# Locate the dataset relative to the project root
project_root = Path(__file__).resolve().parent.parent
source = project_root / "data" / "raw" / "telco.csv"

# Load the original dataset without modifying it
df = pd.read_csv(source)

# Inspect its structure
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nChurn distribution:")
print(df["Churn"].value_counts())



from sklearn.model_selection import train_test_split

# First split: 60% training, 40% remaining
train_df, remaining_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["Churn"],
    random_state=42
)

# Second split: 20% validation, 20% test
valid_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    stratify=remaining_df["Churn"],
    random_state=42
)

print("\nDataset splits:")
print("Training:", train_df.shape)
print("Validation:", valid_df.shape)
print("Test:", test_df.shape)


# Verify that customers do not overlap between splits
train_ids = set(train_df["customerID"])
valid_ids = set(valid_df["customerID"])
test_ids = set(test_df["customerID"])

assert train_ids.isdisjoint(valid_ids)
assert train_ids.isdisjoint(test_ids)
assert valid_ids.isdisjoint(test_ids)

print("\nCustomer ID overlap check: PASSED")


from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Select the two features required for the first model
features = ["tenure", "MonthlyCharges"]

X_train = train_df[features]
X_valid = valid_df[features]

# Convert Churn into binary labels
# No = 0, Yes = 1
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on validation data
y_pred = model.predict(X_valid)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("Confusion matrix [0=No, 1=Yes]:")
print(confusion_matrix(y_valid, y_pred, labels=[0, 1]))



# Train majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_valid)

print("\n=== Majority-Class Baseline ===")
print("Accuracy:", accuracy_score(y_valid, baseline_pred))
print("Confusion matrix [0=No, 1=Yes]:")
print(confusion_matrix(y_valid, baseline_pred, labels=[0, 1]))




# Inspect TotalCharges before preprocessing
charges = df["TotalCharges"].astype("string")

blank_values = charges.str.strip().eq("").sum()
missing_values = charges.isna().sum()

numeric_values = pd.to_numeric(charges, errors="coerce")
conversion_failures = (
    numeric_values.isna()
    & charges.notna()
    & charges.str.strip().ne("")
).sum()

print("\n=== TotalCharges Inspection ===")
print("Data type:", df["TotalCharges"].dtype)
print("Blank/whitespace values:", blank_values)
print("Missing values:", missing_values)
print("Nonblank conversion failures:", conversion_failures)



# Define the six features required for TP02
numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

all_features = numeric_features + categorical_features

# Create copies of our existing splits
# This keeps the original dataset unchanged
X_train_full = train_df[all_features].copy()
X_valid_full = valid_df[all_features].copy()

# Convert TotalCharges to numeric in both splits
for X in [X_train_full, X_valid_full]:
    # Convert whitespace-only values to missing
    X["TotalCharges"] = X["TotalCharges"].str.strip()
    X["TotalCharges"] = X["TotalCharges"].replace("", float("nan"))

    # Reject malformed nonblank numeric values
    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="raise"
    )

print("\n=== Full Feature Preparation ===")
print("Training shape:", X_train_full.shape)
print("Validation shape:", X_valid_full.shape)

print("\nTraining feature types:")
print(X_train_full.dtypes)

print("\nMissing TotalCharges (training):")
print(X_train_full["TotalCharges"].isna().sum())


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the appropriate transformations to each column
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Complete pipeline: preprocessing + model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Fit ONLY on training data
full_pipeline.fit(X_train_full, y_train)

# Predict validation customers (without refitting)
y_pred_full = full_pipeline.predict(X_valid_full)

print("\n=== Complete Pipeline ===")
print("Accuracy:", accuracy_score(y_valid, y_pred_full))
print("Confusion matrix [0=No, 1=Yes]:")
print(confusion_matrix(y_valid, y_pred_full, labels=[0, 1]))



from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# Create a fresh pipeline for every classifier
def build_pipeline(classifier):
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(classifier))
    ])

# Four models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),

    "Majority Baseline": DummyClassifier(
        strategy="most_frequent"
    )
}

print("\n=== Model Candidates ===")
for name, classifier in models.items():
    print(name, ":", classifier.__class__.__name__)


import time
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# Primary selection metric: Average Precision
# Churn = Yes (1) is the positive class.
# Fixed classification threshold: 0.5

results = []
fitted_models = {}

for name, classifier in models.items():

    # Fresh pipeline for each candidate
    candidate = build_pipeline(classifier)

    # Measure training duration
    start = time.perf_counter()
    candidate.fit(X_train_full, y_train)
    fit_time = time.perf_counter() - start

    # Positive-class probabilities
    positive_index = list(candidate.classes_).index(1)
    y_scores = candidate.predict_proba(X_valid_full)[:, positive_index]

    # Fixed threshold for binary predictions
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluation metrics
    metrics = {
        "model": name,
        "average_precision": float(
            average_precision_score(y_valid, y_scores)
        ),
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "precision": float(
            precision_score(y_valid, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_valid, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_valid, y_pred, zero_division=0)),
        "fit_seconds": round(fit_time, 4),
        "confusion_matrix": confusion_matrix(
            y_valid, y_pred, labels=[0, 1]
        ).tolist()
    }

    results.append(metrics)
    fitted_models[name] = candidate

# Sort models by Average Precision (highest first)
results.sort(
    key=lambda result: result["average_precision"],
    reverse=True
)

print("\n=== MODEL COMPARISON ===")

for result in results:
    print(f"\n{result['model']}")
    print(f"Average Precision: {result['average_precision']:.4f}")
    print(f"Accuracy:          {result['accuracy']:.4f}")
    print(f"Precision:         {result['precision']:.4f}")
    print(f"Recall:            {result['recall']:.4f}")
    print(f"F1:                {result['f1']:.4f}")
    print(f"Fit time:          {result['fit_seconds']:.4f} s")
    print(f"Confusion matrix:  {result['confusion_matrix']}")

# Save machine-readable results
report_path = project_root / "reports" / "model_comparison.json"
report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

# Select only among learned classifiers
learned_results = [
    result for result in results
    if result["model"] != "Majority Baseline"
]

best_result = max(
    learned_results,
    key=lambda result: result["average_precision"]
)

best_model_name = best_result["model"]
selected_pipeline = fitted_models[best_model_name]

print("\n=== SELECTED MODEL ===")
print("Model:", best_model_name)
print("Average Precision:", best_result["average_precision"])
print("Results saved to:", report_path)



import joblib
import hashlib
import sys
import sklearn

# Create output directories
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

evidence_dir = project_root / "reports"
evidence_dir.mkdir(parents=True, exist_ok=True)

# Save the complete fitted pipeline
model_path = models_dir / "selected_pipeline.joblib"
joblib.dump(selected_pipeline, model_path)

# Record metadata for reproducibility
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

metadata = {
    "selected_model": best_model_name,
    "selection_metric": "average_precision",
    "validation_average_precision": best_result["average_precision"],
    "validation_metrics": best_result,
    "positive_class": "Churn=Yes (1)",
    "prediction_threshold": 0.5,
    "random_seed": 42,
    "features": {
        "numeric": numeric_features,
        "categorical": categorical_features
    },
    "preprocessing": {
        "numeric_imputer": "median",
        "numeric_scaler": "StandardScaler",
        "categorical_imputer": "most_frequent",
        "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')"
    },
    "source_sha256": source_hash,
    "source_path": str(source.relative_to(project_root)),
    "model_params": selected_pipeline.named_steps[
        "classifier"
    ].get_params(),
    "environment": {
        "python": sys.version,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__
    }
}

metadata_path = models_dir / "selected_pipeline_metadata.json"

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n=== MODEL SAVED ===")
print("Model:", model_path)
print("Metadata:", metadata_path)
print("Source SHA-256:", source_hash)


# Save a small inference batch and reference predictions
evidence_dir = project_root / "reports"
sample_path = evidence_dir / "inference_sample.json"
expected_path = evidence_dir / "expected_predictions.json"

sample = X_valid_full.head(5).copy()

sample.to_json(
    sample_path,
    orient="records",
    indent=2
)

expected = {
    "predictions": selected_pipeline.predict(sample).tolist(),
    "probabilities": selected_pipeline.predict_proba(sample)[
        :, list(selected_pipeline.classes_).index(1)
    ].tolist()
}

with expected_path.open("w", encoding="utf-8") as file:
    json.dump(expected, file, indent=2)

print("\n=== INFERENCE EVIDENCE SAVED ===")
print("Sample:", sample_path)
print("Expected predictions:", expected_path)



import numpy as np

print("\n=== PREPROCESSING RELIABILITY ===")

# Access fitted numeric preprocessing
numeric_pipe = selected_pipeline.named_steps[
    "preprocessor"
].named_transformers_["num"]

imputer = numeric_pipe.named_steps["imputer"]

# Medians learned by the pipeline
learned_medians = imputer.statistics_.copy()

# Independently calculate training-only medians
expected_medians = X_train_full[numeric_features].median().to_numpy()

print("Learned medians:", learned_medians)
print("Training medians:", expected_medians)

# Check that only training statistics were learned
assert np.allclose(learned_medians, expected_medians)
print("Training-only imputation: PASSED")

# Verify prediction does not modify statistics
before_prediction = imputer.statistics_.copy()

selected_pipeline.predict(X_valid_full.head(5))

after_prediction = imputer.statistics_.copy()

assert np.array_equal(before_prediction, after_prediction)
print("Prediction does not change imputation statistics: PASSED")


print("\n=== INPUT RELIABILITY TESTS ===")

# Use one validation record as a starting point
sample = X_valid_full.head(1).copy()

# TEST 1: Missing numeric value
missing_numeric = sample.copy()
missing_numeric["TotalCharges"] = np.nan

prediction = selected_pipeline.predict(missing_numeric)
print("Missing TotalCharges: PASSED")
print("Prediction:", prediction.tolist())

# TEST 2: Previously unseen category
unknown_category = sample.copy()
unknown_category["Contract"] = "Unknown Contract Type"

prediction = selected_pipeline.predict(unknown_category)
print("Unknown category: PASSED")
print("Prediction:", prediction.tolist())

# TEST 3: Missing required feature
missing_feature = sample.drop(columns=["tenure"])

try:
    selected_pipeline.predict(missing_feature)
except (ValueError, KeyError) as error:
    print("Missing required feature: PASSED")
    print("Error type:", type(error).__name__)
else:
    raise AssertionError(
        "FAILED: Model accepted a missing required feature"
    )


# Save exact customer allocations for reproducibility

split_record = {
    "random_seed": 42,
    "split_method": "stratified 60/20/20",
    "stratification_target": "Churn",
    "train_customer_ids": train_df["customerID"].tolist(),
    "validation_customer_ids": valid_df["customerID"].tolist(),
    "test_customer_ids": test_df["customerID"].tolist()
}

split_path = project_root / "reports" / "split_allocations.json"

with split_path.open("w", encoding="utf-8") as file:
    json.dump(split_record, file, indent=2)

print("\n=== SPLIT REPRODUCIBILITY ===")
print("Training IDs:", len(split_record["train_customer_ids"]))
print("Validation IDs:", len(split_record["validation_customer_ids"]))
print("Test IDs:", len(split_record["test_customer_ids"]))
print("Saved to:", split_path)
