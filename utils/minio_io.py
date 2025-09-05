from __future__ import annotations
from typing import Iterable, Optional, List
from pathlib import Path
import os
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None
    ClientError = Exception


class MinioSink:

    def __init__(self, endpoint: str, key: str, secret: str, region: str, bucket: str, prefix: str = ""):
        if boto3 is None:
            raise RuntimeError("boto3 is not installed in this environment")
        self.endpoint = endpoint
        self.bucket = bucket
        self.prefix = (prefix or "").lstrip("/")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region,
        )
        self._ensure_bucket()

    @classmethod
    def from_env(cls, enabled: bool = True) -> Optional["MinioSink"]:
        if not enabled:
            return None
        endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        key = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        bucket = os.getenv("MINIO_BUCKET")
        prefix = os.getenv("MINIO_PREFIX", "mntrading/")
        if not endpoint or not key or not secret or not bucket:
            return None
        return cls(endpoint=endpoint, key=key, secret=secret, region=region, bucket=bucket, prefix=prefix)

    # ---------------- internal ----------------
    def _ensure_bucket(self) -> None:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.s3.create_bucket(Bucket=self.bucket)

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{key}"
        return key

    def upload_file(self, local_path: str | Path, key: str) -> None:
        p = Path(local_path)
        if not p.exists() or not p.is_file():
            return
        self.s3.upload_file(str(p), self.bucket, self._full_key(key))

    def upload_dir(self, src: str | Path, prefix: str = "") -> None:
        root = Path(src).resolve()
        if not root.exists() or not root.is_dir():
            return
        base = self._full_key(prefix)
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                key = f"{base.rstrip('/')}/{rel}"
                self.s3.upload_file(str(p), self.bucket, key)

    def download_file(self, key: str, local_path: str | Path) -> None:
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self._full_key(key), str(p))

    def download_dir(self, prefix: str, local_dir: str | Path) -> int:
        files = self._list_objects(prefix)
        root = Path(local_dir).resolve()
        n = 0
        for key in files:
            rel = key[len(prefix.lstrip("/")) :].lstrip("/")
            if rel == "":
                continue
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._full_key(key), str(out))
            n += 1
        return n

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError:
            return False

    def list_prefix(self, prefix: str) -> List[str]:
        return list(self._list_objects(prefix))

    # ---------------- helpers ----------------
    def _list_objects(self, prefix: str) -> Iterable[str]:
        full_prefix = self._full_key(prefix)
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key_full = obj["Key"]
                if self.prefix and key_full.startswith(self.prefix):
                    yield key_full[len(self.prefix):]
                else:
                    yield key_full
