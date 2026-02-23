"""S3 artifact helpers for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class S3ArtifactStore:
    """Thin wrapper around boto3 S3 operations used by stage scripts."""

    def __init__(self, bucket: str, region: str = "", prefix: str = "") -> None:
        self.bucket = (bucket or "").strip()
        if not self.bucket:
            raise ValueError("S3 bucket is required")
        self.prefix = (prefix or "").strip().strip("/")
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise RuntimeError("boto3 is required for S3ArtifactStore") from exc
        session = boto3.session.Session(region_name=(region or None))
        self.client = session.client("s3")

    def key(self, relative_key: str) -> str:
        rel = (relative_key or "").strip().strip("/")
        if not self.prefix:
            return rel
        if not rel:
            return self.prefix
        return f"{self.prefix}/{rel}"

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            resp = getattr(exc, "response", {}) or {}
            code = str(resp.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def download_file(self, key: str, local_path: Path) -> None:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))

    def upload_file(self, local_path: Path, key: str) -> None:
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Local file does not exist: {local}")
        self.client.upload_file(str(local), self.bucket, key)

    def download_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def upload_bytes(self, content: bytes, key: str, content_type: str = "") -> None:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key, "Body": content}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def upload_json(self, payload: Any, key: str) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.upload_bytes(data, key, content_type="application/json")

    def download_json(self, key: str, default: Optional[Any] = None) -> Any:
        if not self.exists(key):
            return default
        raw = self.download_bytes(key)
        return json.loads(raw.decode("utf-8"))

    def restore_state_if_exists(self, s3_key: str, local_path: Path) -> bool:
        if not s3_key:
            return False
        if not self.exists(s3_key):
            return False
        self.download_file(s3_key, local_path)
        return True

    def persist_state(self, local_path: Path, s3_key: str) -> bool:
        if not s3_key:
            return False
        local = Path(local_path)
        if not local.exists():
            return False
        self.upload_file(local, s3_key)
        return True

    def list_keys(self, prefix: str) -> list[str]:
        """List object keys under prefix."""
        keys: list[str] = []
        token: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key")
                if key:
                    keys.append(str(key))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys
