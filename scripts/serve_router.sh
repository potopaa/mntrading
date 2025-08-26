#!/usr/bin/env bash
set -euo pipefail

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
