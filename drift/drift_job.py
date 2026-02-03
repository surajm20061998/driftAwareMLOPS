import os
import json
import math
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

import boto3

# --- DB ---
PG_HOST = os.getenv("PG_HOST", "mlops_postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "mlflow")
PG_USER = os.getenv("PG_USER", "mlflow")
PG_PASSWORD = os.getenv("PG_PASSWORD", "mlflow")

# --- MinIO/S3 ---
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")).strip()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio12345")
BUCKET = os.getenv("DRIFT_BUCKET", "mlflow")

REFERENCE_KEY = os.getenv("REFERENCE_KEY", "drift/reference.json")
REPORT_KEY = os.getenv("REPORT_KEY", "drift/reports/latest.json")

WINDOW = int(os.getenv("DRIFT_WINDOW_ROWS", "500"))     # last N requests
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))  # >0.2 often considered notable drift

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )

def load_reference():
    s3 = s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key=REFERENCE_KEY)
    return json.loads(obj["Body"].read().decode("utf-8"))

def fetch_recent_features():
    with psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT features
                FROM inference_logs
                ORDER BY id DESC
                LIMIT %s
                """,
                (WINDOW,),
            )
            rows = cur.fetchall()

    if not rows:
        return None

    # features stored as JSON list of 20 floats
    X = np.array([r["features"] for r in rows], dtype=np.float64)
    return X

def psi(expected_bins, expected_counts, actual_counts, eps=1e-6):
    # PSI = sum((a - e) * ln(a/e))
    e = np.array(expected_counts, dtype=np.float64) + eps
    a = np.array(actual_counts, dtype=np.float64) + eps
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))

def compute_psi_per_feature(X, ref):
    n_features = X.shape[1]
    results = {}
    for i in range(n_features):
        name = f"f{i}"
        bins = ref["features"][name]["bins"]
        # reference expected distribution: uniform quantile buckets (~10% each)
        expected_counts = np.ones(len(bins) - 1)

        actual_counts, _ = np.histogram(X[:, i], bins=bins)
        results[name] = psi(bins, expected_counts, actual_counts)
    return results

def upload_report(report):
    s3 = s3_client()
    s3.put_object(Bucket=BUCKET, Key=REPORT_KEY, Body=json.dumps(report, indent=2).encode("utf-8"))
    print(f"✅ Uploaded drift report to s3://{BUCKET}/{REPORT_KEY}")

def main():
    ref = load_reference()
    X = fetch_recent_features()
    if X is None:
        print("No inference logs yet; nothing to analyze.")
        return 0

    psi_scores = compute_psi_per_feature(X, ref)
    overall = float(np.mean(list(psi_scores.values())))

    report = {
        "window_rows": int(X.shape[0]),
        "psi_threshold": PSI_THRESHOLD,
        "overall_psi": overall,
        "per_feature_psi": psi_scores,
        "drift_detected": bool(overall >= PSI_THRESHOLD),
    }

    print(json.dumps(report, indent=2))
    upload_report(report)

    # Exit code can be used by a scheduler to trigger retraining
    return 2 if report["drift_detected"] else 0

if __name__ == "__main__":
    code = main()
    print(f"EXITING_WITH_CODE={code}")
    raise SystemExit(code)