# scripts/upload_to_minio.py
# -*- coding: utf-8 -*-
"""
Upload a local folder recursively to MinIO (S3-compatible) bucket.
Environment:
  - MLFLOW_S3_ENDPOINT_URL = http://minio:9000
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  - MINIO_BUCKET = bucket name (default: mlflow)
Usage:
  python scripts/upload_to_minio.py --src /app/data/datasets --prefix datasets/
All comments are in English.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)

def upload_dir(s3, bucket: str, src: Path, prefix: str = ""):
    src = src.resolve()
    for p in src.rglob("*"):
        if p.is_file():
            key = f"{prefix}{p.relative_to(src).as_posix()}"
            s3.upload_file(str(p), bucket, key)

def main():
    ap = argparse.ArgumentParser(description="Upload folder to MinIO (S3)")
    ap.add_argument("--src", required=True, help="Local folder to upload recursively")
    ap.add_argument("--prefix", default="", help="Prefix inside bucket, e.g. datasets/")
    args = ap.parse_args()

    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "admin")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "adminadmin")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket = os.getenv("MINIO_BUCKET", "mlflow")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=region,
    )

    ensure_bucket(s3, bucket)
    upload_dir(s3, bucket, Path(args.src), args.prefix)
    print(f"Uploaded {args.src} to s3://{bucket}/{args.prefix}")

if __name__ == "__main__":
    main()
