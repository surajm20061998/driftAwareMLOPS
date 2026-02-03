import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

MODEL_NAME = "DriftAwareDemoModel"

def main():
    # IMPORTANT: your MLflow UI is on host 5050
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment("drift-aware-demo")

    # Synthetic dataset (we’ll add drift later)
    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("val_auc", auc)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Validation AUC: {auc:.4f}")

        # Register model in MLflow Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    # Move latest version to Staging
    client = MlflowClient()
    version = registered.version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered {MODEL_NAME} v{version} → Staging")

if __name__ == "__main__":
    main()
