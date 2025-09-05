#!/usr/bin/env bash
set -euo pipefail

<<<<<<< HEAD

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://minio:9000}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-admin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-adminadmin}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

MODEL_URI="${MODEL_URI:-models:/mntrading_router/Production}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5001}"
WORKERS="${WORKERS:-1}"


export MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT="${MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT:-600}"

echo "[serve_router] MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "[serve_router] MODEL_URI=${MODEL_URI}"
echo "[serve_router] PORT=${PORT}, WORKERS=${WORKERS}, REQ_TIMEOUT=${MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT}"


for i in $(seq 1 60); do
  if curl -fsS "${MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list" >/dev/null; then
    echo "[serve_router] MLflow is reachable"
    break
  fi
  echo "[serve_router] waiting for MLflow... (${i}/60)"
  sleep 2
done


python - <<'PY'
import os, sys
from mlflow.tracking import MlflowClient
uri = os.environ.get("MODEL_URI","models:/mntrading_router/Production")
if uri.startswith("models:/"):
    name = uri.split("models:/",1)[1].split("/",1)[0]
    c = MlflowClient()
    try:
        c.get_registered_model(name)
        print("[serve_router] MODEL_OK:", name)
    except Exception as e:
        print("[serve_router] MODEL_MISSING:", name, e)
PY


exec mlflow models serve \
  -m "${MODEL_URI}" \
  --env-manager local \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${WORKERS}"
=======
MODEL_NAME="${MODEL_NAME:-mntrading_router}"
MODEL_STAGE="${MODEL_STAGE:-Production}"
TRACKING="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
PORT="${PORT:-5001}"

echo "[serve_router] waiting for model ${MODEL_NAME} @ ${MODEL_STAGE} in ${TRACKING} ..."
until python - <<'PY'
import os, mlflow
from mlflow.tracking import MlflowClient
name = os.getenv("MODEL_NAME","mntrading_router")
client = MlflowClient()
mvs = client.search_model_versions(f"name='{name}'")
print("found", len(list(mvs)))
raise SystemExit(0 if any(v.current_stage=="Production" for v in mvs) else 1)
PY
do
  echo "[serve_router] no Production version yet; sleep 10s..."
  sleep 10
done

exec mlflow models serve -m "models:/${MODEL_NAME}/${MODEL_STAGE}" --host 0.0.0.0 --port "${PORT}" --env-manager local
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
