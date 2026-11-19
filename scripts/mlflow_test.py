
import mlflow

# Connect to our local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create the experiment if it doesn't exist
mlflow.set_experiment("telco-churn-tp03")

# Create our first test run
with mlflow.start_run(run_name="first-test-run") as run:
    mlflow.log_param("model_type", "test")
    mlflow.log_param("random_state", 42)

    print("Experiment ID:", run.info.experiment_id)
    print("Run ID:", run.info.run_id)

print("Test run completed successfully.")
