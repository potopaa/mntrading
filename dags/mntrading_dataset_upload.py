# dags/mntrading_dataset_upload.py
# -*- coding: utf-8 -*-
"""
Airflow DAG: build dataset then upload to MinIO.
Run under Linux/WSL2. Airflow is not supported natively on Windows.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="mntrading_dataset_upload",
    default_args=default_args,
    description="Build dataset and upload to MinIO",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mntrading"],
) as dag:

    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=(
            "python /app/main.py --mode dataset "
            "--label-type z_threshold --zscore-threshold 1.5 --lag-features 10 --horizon 3"
        ),
    )

    upload_to_minio = BashOperator(
        task_id="upload_to_minio",
        bash_command=(
            "python /app/scripts/upload_to_minio.py --src /app/data/datasets --prefix datasets/"
        ),
        env={
            "MLFLOW_S3_ENDPOINT_URL": "{{ var.value.get('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000') }}",
            "AWS_ACCESS_KEY_ID": "{{ var.value.get('AWS_ACCESS_KEY_ID', 'admin') }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.get('AWS_SECRET_ACCESS_KEY', 'adminadmin') }}",
            "AWS_DEFAULT_REGION": "{{ var.value.get('AWS_DEFAULT_REGION', 'us-east-1') }}",
            "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'mlflow') }}",
        },
    )

    build_dataset >> upload_to_minio
