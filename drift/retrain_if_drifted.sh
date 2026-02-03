#!/usr/bin/env bash
set -euo pipefail

# --- run drift job in docker and capture its exit code (without script exiting) ---
set +e
docker run --rm --network infra_default \
  -e PG_HOST=mlops_postgres -e PG_PORT=5432 -e PG_DB=mlflow -e PG_USER=mlflow -e PG_PASSWORD=mlflow \
  -e AWS_ACCESS_KEY_ID=minio -e AWS_SECRET_ACCESS_KEY=minio12345 -e S3_ENDPOINT_URL=http://minio:9000 \
  -e DRIFT_BUCKET=mlflow -e DRIFT_WINDOW_ROWS=${DRIFT_WINDOW_ROWS:-300} -e PSI_THRESHOLD=${PSI_THRESHOLD:-0.2} \
  -v "$PWD:/work" -w /work \
  python:3.12-slim bash -lc \
  "pip -q install psycopg2-binary boto3 numpy >/dev/null; python drift/drift_job.py"
code=$?
set -e

echo "drift_job exit code: $code"

if [[ "$code" != "2" ]]; then
  echo "No drift trigger; skipping retrain."
  exit 0
fi

echo "Drift detected → retraining locally..."
python3 training/train.py

echo "Retrain done."
