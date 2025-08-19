# -*- coding: utf-8 -*-
"""
Daily dataset build + upload to MinIO.

Steps:
  1) Build datasets from features manifest.
  2) Upload datasets/ to MinIO (same env vars as MLflow client).

Note:
- This DAG must be run by Airflow scheduler inside Linux/WSL2 or Docker container.
- Do not execute this file directly with 'python'.
"""

from __future__ import annotations
import os
from datetime import datetime

# Safe import for different Airflow versions
try:
    from airflow import DAG
except Exception as e:
    # This will fail under native Windows or if Airflow is not installed.
    raise RuntimeError(
        "Airflow must run inside Linux/WSL2 or Docker. "
        "Do not run this file with 'python'. Use Airflow scheduler."
    ) from e

# BashOperator location in Airflow 2.x
try:
    from airflow.operators.bash import BashOperator
except Exception:
    # Fallback for very old Airflow (<2.0)
    from airflow.operators.bash_operator import BashOperator  # type: ignore

default_args = {
    "owner": "airflow",
    "retries": 0,
}

with DAG(
    dag_id="mntrading_dataset_upload_daily",
    default_args=default_args,
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mntrading", "datasets", "minio"],
    description="Build datasets and upload to MinIO",
) as dag:

    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=(
            "python /app/main.py "
            "--mode dataset "
            "--pairs-manifest /app/data/features/pairs/_manifest.json "
            "--label-type z_threshold "
            "--z-th 1.5 "
            "--lag-features 10 "
            "--horizon 3"
        ),
        env=os.environ.copy(),
    )

    upload_to_minio = BashOperator(
        task_id="upload_to_minio",
        bash_command=(
            "python /app/scripts/upload_to_minio.py "
            "--src /app/data/datasets "
            "--prefix datasets/"
        ),
        env=os.environ.copy(),
    )

    build_dataset >> upload_to_minio
