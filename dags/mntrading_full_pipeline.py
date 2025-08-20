# dags/mntrading_full_pipeline.py
# All comments are in English by request.

from __future__ import annotations
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

# NOTE: Airflow on Windows is not supported; run this DAG in Linux/containers.
# This DAG assumes the 'app' service has the project at /app.

default_args = {
    "owner": "mntrading",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Tunables via env
SINCE_1H = os.getenv("MNTR_SINCE_1H", "2025-01-01T00:00:00Z")
SINCE_5M = os.getenv("MNTR_SINCE_5M", "2025-01-01T00:00:00Z")
TOP = os.getenv("MNTR_TOP", "200")
PROBA = os.getenv("MNTR_PROBA", "0.55")

with DAG(
    dag_id="mntrading_full_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,   # trigger manually, or set a cron
    catchup=False,
    default_args=default_args,
    tags=["mntrading"],
) as dag:

    # -------- Core pipeline --------
    ingest_1h = BashOperator(
        task_id="ingest_1h",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/main.py --mode ingest --symbols-auto --exchange binance --quote USDT "
            f"--top {TOP} --timeframe 1h --since-utc {SINCE_1H} --limit 1000 --max-candles 0\""
        ),
    )

    screen_and_ingest_5m = BashOperator(
        task_id="screen_and_ingest_5m",
        bash_command=(
            "docker compose exec app bash -lc "
            f"\"python /app/main.py --mode screen --since-utc-5m {SINCE_5M} --limit-5m 1000\""
        ),
    )

    features = BashOperator(
        task_id="features",
        bash_command=(
            # Disable eager MinIO upload for speed; we sync later explicitly
            "docker compose exec app bash -lc "
            "\"MLFLOW_S3_ENDPOINT_URL= MINIO_BUCKET= "
            "python /app/main.py --mode features --symbols '/app/data/pairs/screened_pairs_*.json' "
            "--beta-window 1000 --z-window 300\""
        ),
    )

    dataset = BashOperator(
        task_id="dataset",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"DATASET_DEBUG=1 DATASET_MIN_ROWS=50 "
            "python /app/main.py --mode dataset --label-type z_threshold --zscore-threshold 1.2 "
            "--lag-features 10 --horizon 3\""
        ),
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/main.py --mode train --use-dataset --n-splits 5 --gap 24 "
            f"--early-stopping-rounds 50 --proba-threshold {PROBA}\""
        ),
    )

    backtest = BashOperator(
        task_id="backtest",
        bash_command=(
            "docker compose exec app bash -lc "
            f"\"python /app/main.py --mode backtest --signals-from auto --proba-threshold {PROBA} --fee-rate 0.0005\""
        ),
    )

    select = BashOperator(
        task_id="select",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/main.py --mode select --summary-path /app/data/backtest_results/_summary.json "
            "--registry-out /app/data/models/registry.json --sharpe-min 0.0 --maxdd-max 1.0 --top-k 20\""
        ),
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/main.py --mode promote --registry-in /app/data/models/registry.json "
            "--production-map-out /app/data/models/production_map.json\""
        ),
    )

    inference = BashOperator(
        task_id="inference",
        bash_command=(
            "docker compose exec app bash -lc "
            f"\"python /app/main.py --mode inference --registry-in /app/data/models/registry.json --signals-from model --proba-threshold {PROBA} --n-last 1 --update\""
        ),
    )

    aggregate = BashOperator(
        task_id="aggregate",
        bash_command=(
            "docker compose exec app bash -lc "
            f"\"python /app/main.py --mode aggregate --min-proba {PROBA} --top-k 20\""
        ),
    )

    report = BashOperator(
        task_id="report",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/main.py --mode report --summary-path /app/data/backtest_results/_summary.json "
            "--orders-json /app/data/portfolio/latest_orders.json --report-out /app/data/portfolio/_latest_report.md\""
        ),
    )

    # -------- Dedicated MinIO sync tasks --------
    sync_features_datasets = BashOperator(
        task_id="sync_features_datasets",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/scripts/upload_to_minio.py "
            "/app/data/features /app/data/datasets\""
        ),
        trigger_rule="all_done",  # attempt to sync whatever exists
    )

    sync_models_backtest_portfolio = BashOperator(
        task_id="sync_models_backtest_portfolio",
        bash_command=(
            "docker compose exec app bash -lc "
            "\"python /app/scripts/upload_to_minio.py "
            "/app/data/models /app/data/backtest_results /app/data/portfolio\""
        ),
        trigger_rule="all_done",
    )

    # -------- Dependencies --------
    ingest_1h >> screen_and_ingest_5m >> features >> dataset >> train >> backtest >> select >> promote >> inference >> aggregate >> report
    dataset >> sync_features_datasets
    report >> sync_models_backtest_portfolio
