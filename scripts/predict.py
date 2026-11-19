
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "models" / "selected_pipeline.joblib"
sample_path = project_root / "reports" / "inference_sample.json"
expected_path = project_root / "reports" / "expected_predictions.json"

# Only load artifacts you created or otherwise trust.
model = joblib.load(model_path)

# Read original, unencoded feature columns
sample = pd.read_json(sample_path, orient="records")

# Same input parsing policy used during training
sample["TotalCharges"] = pd.to_numeric(
    sample["TotalCharges"].astype("string").str.strip().replace("", np.nan),
    errors="raise"
)

# Predict WITHOUT calling fit()
predictions = model.predict(sample)
probabilities = model.predict_proba(sample)[
    :, list(model.classes_).index(1)
]

with expected_path.open("r", encoding="utf-8") as file:
    expected = json.load(file)

# Compare loaded pipeline with original pipeline
assert predictions.tolist() == expected["predictions"]
assert np.allclose(
    probabilities,
    expected["probabilities"],
    rtol=1e-7,
    atol=1e-9
)

print("\n=== RELOAD VERIFICATION ===")
print("Predictions:", predictions.tolist())
print("Churn probabilities:", probabilities.tolist())
print("Original vs loaded predictions: MATCH")
print("No training performed: PASSED")
