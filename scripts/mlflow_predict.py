
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Connect to local tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model from registry using its alias
model_uri = "models:/telco-churn@candidate"
loaded_pipeline = mlflow.sklearn.load_model(model_uri)

# Load the same inference examples from TP02
project_root = Path(__file__).resolve().parent.parent

sample_path = project_root / "reports" / "inference_sample.json"
expected_path = project_root / "reports" / "expected_predictions.json"

sample = pd.read_json(sample_path, orient="records")

# Apply the same stateless parsing as during training
sample["TotalCharges"] = pd.to_numeric(
    sample["TotalCharges"].astype("string").str.strip().replace("", np.nan),
    errors="raise"
)

# No training or refitting
predictions = loaded_pipeline.predict(sample)

positive_index = list(loaded_pipeline.classes_).index(1)
probabilities = loaded_pipeline.predict_proba(sample)[:, positive_index]

# Compare with predictions from the original fitted pipeline
with expected_path.open("r", encoding="utf-8") as file:
    expected = json.load(file)

assert predictions.tolist() == expected["predictions"]
assert np.allclose(
    probabilities,
    expected["probabilities"],
    rtol=1e-7,
    atol=1e-9
)

print("\n=== MLFLOW REGISTRY INFERENCE ===")
print("Model URI:", model_uri)
print("Predictions:", predictions.tolist())
print("Churn probabilities:", probabilities.tolist())
print("Original vs registered model: MATCH")
print("Inference without retraining: PASSED")
