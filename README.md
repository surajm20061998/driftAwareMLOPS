🧠 Drift-Aware MLOps System (Prod + Canary)

This repository implements a drift-aware machine learning deployment system with:
	•	Offline drift detection (PSI)
	•	Automated retraining
	•	MLflow Model Registry
	•	Production + Canary inference services
	•	Postgres inference logging
	•	MinIO (S3) artifact storage
	•	Manual promotion workflow
	•	Observability hooks (Prometheus-ready)

The system continuously monitors feature drift and enables safe, controlled rollout of new models via canary deployments.

⸻

🏗️ Architecture Overview

                        ┌────────────┐
                        │  MinIO     │
                        │  (S3)      │
                        │ artifacts  │
                        └─────┬──────┘
                              │
        ┌───────────────┐     │      ┌────────────────┐
        │ Drift Job     │─────┼─────▶│ MLflow Server  │
        │ (PSI)         │     │      │ Model Registry │
        └─────┬─────────┘     │      └───────┬────────┘
              │               │              │
              │ drift=true    │              │
              ▼               │              ▼
     ┌─────────────────┐     │     ┌──────────────────┐
     │ Retraining      │─────┘     │ Inference Prod   │
     │ (local / CI)    │           │ MODEL_STAGE=Prod │
     └─────┬───────────┘           └───────┬──────────┘
           │                               │
           │ new version (Staging)         │
           ▼                               ▼
 ┌──────────────────┐           ┌──────────────────┐
 │ Inference Canary │           │ Postgres          │
 │ MODEL_STAGE=Stg │──────────▶│ inference_logs    │
 └──────────────────┘           └──────────────────┘


⸻

🚀 Components

1️⃣ Drift Detection (drift/drift_job.py)
	•	Pulls recent inference data from Postgres
	•	Computes Population Stability Index (PSI)
	•	Compares against baseline distribution
	•	Uploads drift report to MinIO
	•	Exits with:
	•	0 → no drift
	•	2 → drift detected (used by shell script)

⸻

2️⃣ Automated Retraining (drift/retrain_if_drifted.sh)
	•	Runs drift job
	•	If drift detected:
	•	Triggers retraining
	•	Logs model to MLflow
	•	Registers new version in Staging
	•	Designed to run locally or in CI

⸻

3️⃣ Model Registry (MLflow)
	•	Centralized model tracking
	•	Uses stages (Production / Staging / Archived)
	•	Models stored in MinIO (s3://mlflow)
	•	Promotion handled via MLflow API

⸻

4️⃣ Inference Services (FastAPI)

Service	Port	Model Stage	Deployment
inference	8000	Production	legacy
inference_prod	8001	Production	prod
inference_canary	8002	Staging	canary

Each service:
	•	Loads model from MLflow
	•	Serves /predict
	•	Logs inference metadata to Postgres
	•	Exposes /metrics for Prometheus

⸻

5️⃣ Inference Logging (Postgres)

Table: inference_logs

CREATE TABLE inference_logs (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT now(),
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  latency_ms DOUBLE PRECISION NOT NULL,
  features JSONB NOT NULL,
  prediction DOUBLE PRECISION NOT NULL,
  deployment TEXT NOT NULL
);

This enables:
	•	Prod vs canary comparison
	•	Latency analysis
	•	Drift analysis per deployment
	•	Rollback decisions

⸻

🧪 Example Usage

Health Checks

curl http://localhost:8001/health   # prod
curl http://localhost:8002/health   # canary

Prediction

curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'


⸻

📊 Compare Prod vs Canary

SELECT deployment,
       COUNT(*) AS n,
       AVG(latency_ms) AS avg_latency,
       AVG(prediction) AS avg_pred
FROM inference_logs
GROUP BY deployment;


⸻

🚦 Canary Promotion (Manual)

Once canary behavior is acceptable:

curl -X POST http://localhost:5050/api/2.0/mlflow/model-versions/transition-stage \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DriftAwareDemoModel",
    "version": "3",
    "stage": "Production",
    "archive_existing_versions": true
  }'

Then restart prod:

docker restart mlops_inference_prod


⸻

🔍 Observability
	•	Prometheus metrics exposed at /metrics
	•	Latency histogram
	•	Request counters
	•	Model load success/failure

⸻

🧱 Stack
	•	ML Framework: scikit-learn
	•	Serving: FastAPI + Uvicorn
	•	Registry: MLflow
	•	Storage: MinIO (S3)
	•	DB: Postgres 16
	•	Orchestration: Docker Compose
	•	Monitoring: Prometheus-ready

⸻

