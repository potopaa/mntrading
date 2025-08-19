#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upload local datasets (or any directory) to MinIO/S3.

Default behavior:
- Source: data/datasets
- Bucket: env MINIO_BUCKET
- Endpoint: env MLFLOW_S3_ENDPOINT_URL
- Credentials: env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
- Region: env AWS_DEFAULT_REGION (default: us-east-1)
- Prefix on S3: "datasets/" (can be changed via --prefix)

Usage examples:
  python scripts/upload_to_minio.py
  python scripts/upload_to_minio.py --src data/datasets --prefix datasets/
  python scripts/upload_to_minio.py --src data/features --prefix features/
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import boto3
from botocore.client import Config

def _norm_key(*parts: str) -> str:
    """Join path parts into POSIX-style S3 key."""
    return "/".join(p.strip("/").replace("\\", "/") for p in parts if p is not None and p != "")

def _walk_files(root: Path):
    """Yield absolute file paths under root (recursively)."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield Path(dirpath) / fn

def upload_dir(
    src_dir: Path,
    bucket: str,
    prefix: str,
    endpoint_url: str,
    region: str,
    access_key: str,
    secret_key: str,
    overwrite: bool = True,
) -> int:
    """Upload all files from src_dir to s3://bucket/prefix/.. . Return number of uploaded files."""
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"max_attempts": 5, "mode": "standard"})
    )

    uploaded = 0
    src_dir = src_dir.resolve()
    for path in _walk_files(src_dir):
        rel = path.relative_to(src_dir).as_posix()
        key = _norm_key(prefix, rel)
        # Overwrite policy is explicit (S3 PutObject is idempotent for same key)
        print(f"[upload] {path} -> s3://{bucket}/{key}")
        s3.upload_file(str(path), bucket, key)
        uploaded += 1

    print(f"[done] uploaded {uploaded} files from {src_dir} to s3://{bucket}/{prefix}")
    return uploaded

def main():
    parser = argparse.ArgumentParser(description="Upload a local directory to MinIO/S3")
    parser.add_argument("--src", type=str, default="data/datasets", help="Local source directory")
    parser.add_argument("--bucket", type=str, default=os.getenv("MINIO_BUCKET", ""),
                        help="Target S3 bucket (default: env MINIO_BUCKET)")
    parser.add_argument("--prefix", type=str, default="datasets/", help="Key prefix in bucket")
    parser.add_argument("--endpoint-url", type=str,
                        default=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
                        help="S3/MinIO endpoint URL")
    parser.add_argument("--region", type=str, default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                        help="AWS region (default: us-east-1)")
    parser.add_argument("--access-key-id", type=str, default=os.getenv("AWS_ACCESS_KEY_ID", ""),
                        help="Access key id (default: env AWS_ACCESS_KEY_ID)")
    parser.add_argument("--secret-access-key", type=str, default=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                        help="Secret access key (default: env AWS_SECRET_ACCESS_KEY)")
    parser.add_argument("--no-overwrite", action="store_true", help="(Reserved) Do not overwrite existing keys")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    bucket = args.bucket.strip()
    if not bucket:
        raise SystemExit("Bucket name is required (set MINIO_BUCKET or pass --bucket)")

    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    upload_dir(
        src_dir=src,
        bucket=bucket,
        prefix=prefix,
        endpoint_url=args.endpoint_url,
        region=args.region,
        access_key=args.access_key_id,
        secret_key=args.secret_access_key,
        overwrite=not args.no_overwrite,
    )

if __name__ == "__main__":
    main()
