# MNTrading — end-to-end MLOps pipeline (MinIO + MLflow + FastAPI + Streamlit + Airflow)

## Overview
This repository contains a time-series pipeline for pairs/spread features and a baseline ML model (RandomForest, LightGBM).
The project demonstrates:
- Data ingestion & feature building (rolling OLS β/α, spread, z-score, lags, labels)
- Dataset building & training with time-series CV (gap, max_train_size)
- MLflow tracking with artifacts stored in S3/MinIO
- Model selection (champion/challenger) and promotion map
- Backtest & reporting
- Serving via **FastAPI** (minimal endpoints)
- **Streamlit** demo UI
- **Airflow** DAG to build dataset and upload it to MinIO

> Windows note: run everything under **WSL2** or Linux containers. Airflow does not support native Windows execution.

## Architecture
```
screen → ingest → features → dataset → train → backtest → select → promote → inference → aggregate → report
                 \________________ rolling OLS on pairs (β_t, α_t) _________________/
```

Key components:
- `features/spread.py` — rolling OLS (β/α via Cov/Var + α = E[y] − βE[x]), spread & z-score features
- `features/labels.py` — labels and dataset builder
- `models/train.py` — LogisticRegression, RandomForest, LightGBM + time-series CV; MLflow logging
- `backtest/runner.py`, `models/select.py` — backtest & model selection
- `api/server.py` — minimal FastAPI wrapper (healthcheck + short/full cycle triggers)
- `ui/streamlit_app.py` — one-click demo UI for short/full pipelines, artifacts and plots
- `scripts/upload_to_minio.py` — upload datasets (or any dir) to MinIO
- `dags/mntrading_dataset_upload.py` — Airflow DAG: build dataset → upload to MinIO

## Requirements
- Docker + Docker Compose
- WSL2 (Windows) or a Linux/macOS host
- Python 3.10–3.12 for local dev (containers are preferred)
- `.env` with MinIO/MLflow credentials

Example `.env`:
```
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=adminadmin
MINIO_BUCKET=mlflow
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

MLFLOW_PORT=5000
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=adminadmin
AWS_DEFAULT_REGION=us-east-1

STREAMLIT_PORT=8501
```

## Quickstart (Docker)
1) Build and start infra + app:
```bash
docker compose build --no-cache
docker compose up -d minio setup-minio mlflow api app streamlit
```

2) Open:
- MinIO console: http://localhost:9001 (login from `.env`)
- MLflow UI:     http://localhost:5000
- Streamlit UI:  http://localhost:8501
- FastAPI docs:  http://localhost:8000/docs

3) Run a **short cycle** from Streamlit (recommended) or via CLI:
```bash
# Example from container:
docker compose exec app bash -lc "python /app/main.py --mode screen"
docker compose exec app bash -lc "python /app/main.py --mode ingest"
docker compose exec app bash -lc "python /app/main.py --mode features"
docker compose exec app bash -lc "python /app/main.py --mode dataset --label-type z_threshold --z-th 1.5 --lag-features 10 --horizon 3"
docker compose exec app bash -lc "python /app/main.py --mode train"
docker compose exec app bash -lc "python /app/main.py --mode backtest"
docker compose exec app bash -lc "python /app/main.py --mode select"
```

4) (Optional) Full cycle:
```bash
docker compose exec app bash -lc "python /app/main.py --mode promote"
docker compose exec app bash -lc "python /app/main.py --mode inference"
docker compose exec app bash -lc "python /app/main.py --mode aggregate"
docker compose exec app bash -lc "python /app/main.py --mode report"
```

5) Datasets to MinIO:
```bash
docker compose exec app bash -lc "python /app/scripts/upload_to_minio.py --src /app/data/datasets --prefix datasets/"
```

## Airflow
- Place DAGs under the directory mounted to `/opt/airflow/dags` in the `airflow` container.
- Start Airflow:
```bash
docker compose up -d airflow
docker compose exec airflow bash -lc "airflow dags list"
docker compose exec airflow bash -lc "airflow dags trigger mntrading_dataset_upload_daily"
```
> Do **not** run DAG files directly with `python`. Let the Airflow **scheduler** pick them up.

## Models
We use **RandomForest** and **LightGBM** as baseline estimators (no XGBoost by default).
If needed, XGBoost can be added later behind a feature flag.

## Data & Artifacts
- Local: `data/*` (raw, features, datasets, models, backtest_results, signals, portfolio, report)
- MLflow artifacts: S3/MinIO bucket (`experiments/…`)
- You can upload datasets/features to MinIO using `scripts/upload_to_minio.py`.

## API (FastAPI)
- `GET /health`
- `POST /run/short_cycle` — triggers short pipeline
- `GET /run/short_cycle_now` — alias for short pipeline (simple demo)
- `GET /artifacts` — optional listing of produced artifacts

## Streamlit UI
- Buttons to run **Short** and **Full** pipelines
- Live logs of steps
- Tables/plots for datasets, backtest summary, and equity curves
- Links to MLflow runs

## Known issues
- **Windows**: Airflow must run in WSL2 or Linux containers; do not run DAGs with `python`.
- **CRLF**: ensure `*.py` are LF (`.gitattributes` or `git config core.autocrlf input`).
