"""Cloud upload helpers for raw artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


def _split_cloud_uri(uri: str) -> Tuple[str, str, str]:
    parsed = urlparse((uri or "").strip())
    scheme = (parsed.scheme or "").lower()
    bucket = parsed.netloc
    prefix = (parsed.path or "").lstrip("/")
    if not scheme or not bucket:
        raise ValueError(f"Invalid cloud URI: {uri!r}")
    return scheme, bucket, prefix


class CloudUploader:
    def __init__(self, root_uri: str) -> None:
        self.root_uri = (root_uri or "").strip().rstrip("/")
        self.enabled = bool(self.root_uri)
        self.scheme = ""
        self.bucket = ""
        self.prefix = ""
        self._client = None

        if not self.enabled:
            return

        scheme, bucket, prefix = _split_cloud_uri(self.root_uri)
        if scheme not in {"s3", "gs"}:
            raise ValueError("Unsupported cloud URI scheme. Use s3:// or gs://")

        self.scheme = scheme
        self.bucket = bucket
        self.prefix = prefix

    def _ensure_client(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        if self.scheme == "s3":
            import boto3  # type: ignore

            self._client = boto3.client("s3")
            return self._client
        if self.scheme == "gs":
            from google.cloud import storage  # type: ignore

            self._client = storage.Client()
            return self._client
        raise ValueError(f"Unsupported scheme: {self.scheme}")

    def uri_for(self, relative_key: str) -> str:
        if not self.enabled:
            return ""
        key = "/".join([p.strip("/") for p in [self.prefix, relative_key] if p and p.strip("/")])
        return f"{self.scheme}://{self.bucket}/{key}"

    def upload_file(self, local_path: Path, relative_key: str) -> str:
        if not self.enabled:
            return ""
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Upload source not found: {local}")

        key = "/".join([p.strip("/") for p in [self.prefix, relative_key] if p and p.strip("/")])
        client = self._ensure_client()
        if self.scheme == "s3":
            client.upload_file(str(local), self.bucket, key)
        elif self.scheme == "gs":
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(key)
            blob.upload_from_filename(str(local))
        return f"{self.scheme}://{self.bucket}/{key}"

    def download_file(self, relative_key: str, local_path: Path) -> str:
        if not self.enabled:
            return ""
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        key = "/".join([p.strip("/") for p in [self.prefix, relative_key] if p and p.strip("/")])
        client = self._ensure_client()
        if self.scheme == "s3":
            client.download_file(self.bucket, key, str(local))
        elif self.scheme == "gs":
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(key)
            blob.download_to_filename(str(local))
        return f"{self.scheme}://{self.bucket}/{key}"


def maybe_delete_local(path: Path, delete_after_upload: bool) -> None:
    if delete_after_upload and Path(path).exists():
        Path(path).unlink()
