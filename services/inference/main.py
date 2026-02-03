import os
import time
import json
from typing import List, Optional

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

import psycopg2
from psycopg2.extras import Json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response


# -------------------------
# Config
# -------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlops_mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "DriftAwareDemoModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # we'll load this stage

# Postgres (compose network host = container name)
PG_HOST = os.getenv("PG_HOST", "mlops_postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "mlflow")
PG_USER = os.getenv("PG_USER", "mlflow")
PG_PASSWORD = os.getenv("PG_PASSWORD", "mlflow")
DEPLOYMENT = os.getenv("DEPLOYMENT", "unknown")

# S3/MinIO for artifacts (client-side download)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio12345")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://mlops_minio:9000")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# -------------------------
# Metrics
# -------------------------
REQS = Counter("inference_requests_total", "Total inference requests", ["status"])
LAT = Histogram("inference_latency_seconds", "Inference latency in seconds")
LOADS = Counter("model_loads_total", "Model load events", ["result"])


# -------------------------
# Request/Response models
# -------------------------
class PredictRequest(BaseModel):
    # We expect the same 20 features as your synthetic training data for now
    features: List[float] = Field(..., min_items=20, max_items=20)


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str


app = FastAPI(title="Drift-Aware Inference Service", version="0.1")


# -------------------------
# Model loading helpers
# -------------------------
_model = None
_model_version = None

def get_db_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )

def resolve_model_version(name: str, stage: str) -> str:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # Using stages for now since your registry is stage-based; we’ll migrate later if needed.
    latest = client.get_latest_versions(name, stages=[stage])
    if not latest:
        raise RuntimeError(f"No versions found for {name} in stage {stage}")
    return str(latest[0].version)

def load_model():
    global _model, _model_version
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # These env vars are required when MLflow downloads artifacts from MinIO
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

        version = resolve_model_version(MODEL_NAME, MODEL_STAGE)
        uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"  # stage URI
        _model = mlflow.sklearn.load_model(uri)
        _model_version = version
        LOADS.labels("success").inc()
    except Exception:
        LOADS.labels("failure").inc()
        raise

@app.on_event("startup")
def on_startup():
    # Ensure DB table exists (idempotent)
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
              id BIGSERIAL PRIMARY KEY,
              ts TIMESTAMPTZ NOT NULL DEFAULT now(),
              model_name TEXT NOT NULL,
              model_version TEXT NOT NULL,
              latency_ms DOUBLE PRECISION NOT NULL,
              features JSONB NOT NULL,
              prediction DOUBLE PRECISION NOT NULL
            );
            """)
            cur.execute("ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS deployment TEXT NOT NULL DEFAULT 'unknown';")
            conn.commit()

    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE, "version": _model_version}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global _model, _model_version
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    x = np.array(req.features, dtype=np.float32).reshape(1, -1)

    start = time.time()
    try:
        # RandomForestClassifier supports predict_proba
        proba = float(_model.predict_proba(x)[0, 1])
        status = "ok"
    except Exception as e:
        REQS.labels("error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = time.time() - start

    # Log to Postgres
    latency_ms = elapsed * 1000.0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO inference_logs (model_name, model_version, latency_ms, features, prediction, deployment)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (MODEL_NAME, str(_model_version), latency_ms, Json(req.features), float(proba), DEPLOYMENT),
            )
            conn.commit()

    REQS.labels(status).inc()
    LAT.observe(elapsed)

    return PredictResponse(
        prediction=proba,
        model_name=MODEL_NAME,
        model_version=str(_model_version),
    )

@app.post("/reload-model")
def reload_model():
    # Manual reload endpoint (we’ll automate later)
    load_model()
    return {"reloaded": True, "model": MODEL_NAME, "stage": MODEL_STAGE, "version": _model_version}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
