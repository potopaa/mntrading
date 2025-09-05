#!/usr/bin/env bash
# scripts/serve_router.sh
# Start MLflow model serving for router, resolving URI automatically.

set -euo pipefail

: "${MLFLOW_TRACKING_URI:?MLFLOW_TRACKING_URI is required}"
: "${MLFLOW_S3_ENDPOINT_URL:?MLFLOW_S3_ENDPOINT_URL is required}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"

ROUTER_NAME="${ROUTER_NAME:-mntrading_router}"
ROUTER_STAGE="${ROUTER_STAGE:-Production}"
ROUTER_EXPERIMENT="${ROUTER_EXPERIMENT:-mntrading}"

URI="$(python /app/scripts/resolve_router_uri.py || true)"
if [ -z "${URI}" ]; then
  echo "[serve] failed to resolve router URI; check registry or build router first" >&2
  exit 1
fi

echo "[serve] using URI: ${URI}"
exec mlflow models serve \
  --model-uri "${URI}" \
  --host 0.0.0.0 --port 5001 \
  --env-manager local
