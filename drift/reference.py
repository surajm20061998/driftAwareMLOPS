import os
import json
import numpy as np
import pandas as pd
import boto3

# --- MinIO/S3 config ---
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://mlops_minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio12345")
BUCKET = os.getenv("DRIFT_BUCKET", "mlflow")

REFERENCE_KEY = os.getenv("REFERENCE_KEY", "drift/reference.json")

def make_reference(n_features: int = 20, n_rows: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    # baseline: standard normal per feature (matches your synthetic setup)
    X = rng.normal(0, 1, size=(n_rows, n_features))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

    ref = {"features": {}}
    for c in df.columns:
        arr = df[c].to_numpy()
        ref["features"][c] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) + 1e-12),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            # bin edges used for PSI
            "bins": list(np.quantile(arr, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])),
        }

    return ref

def upload_json(obj, bucket: str, key: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL.strip(),
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=2).encode("utf-8"))
    print(f"✅ Uploaded reference to s3://{bucket}/{key}")

if __name__ == "__main__":
    ref = make_reference()
    upload_json(ref, BUCKET, REFERENCE_KEY)