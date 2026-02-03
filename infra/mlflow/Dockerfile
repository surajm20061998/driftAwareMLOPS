FROM ghcr.io/mlflow/mlflow:v2.14.2

# install curl for healthchecks (debian/ubuntu base in most mlflow images)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Postgres driver + S3 deps
RUN pip install --no-cache-dir psycopg2-binary boto3