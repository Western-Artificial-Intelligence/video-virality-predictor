"""Pinecone client wrapper with retry/backoff and metadata sanitization."""

from __future__ import annotations

import json
import time
from typing import Any, Dict


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_full_metadata(
    row: Dict[str, Any],
    mandatory: Dict[str, Any],
    max_value_len: int = 1000,
    max_kv_pairs: int = 200,
) -> Dict[str, Any]:
    """Build full-row metadata payload with truncation guardrails."""
    out: Dict[str, Any] = {}

    for key, value in (row or {}).items():
        if len(out) >= max_kv_pairs:
            break
        k = str(key)
        v = _normalize_scalar(value)
        if isinstance(v, str) and len(v) > max_value_len:
            v = v[:max_value_len]
        out[k] = v

    for key, value in (mandatory or {}).items():
        v = _normalize_scalar(value)
        if isinstance(v, str) and len(v) > max_value_len:
            v = v[:max_value_len]
        out[str(key)] = v

    return out


class PineconeVectorClient:
    def __init__(self, api_key: str, index_name: str, timeout_s: int = 30) -> None:
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if not index_name:
            raise ValueError("Pinecone index_name is required")
        try:
            from pinecone import Pinecone  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pinecone package is required. Install with: pip install pinecone"
            ) from exc

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        self.timeout_s = timeout_s

    def upsert(
        self,
        vector_id: str,
        values: list[float],
        metadata: Dict[str, Any],
        retry: int = 4,
        retry_delay: float = 1.0,
    ) -> None:
        last_error = ""
        for attempt in range(1, retry + 2):
            try:
                self.index.upsert(
                    vectors=[{"id": vector_id, "values": values, "metadata": metadata}],
                )
                return
            except Exception as exc:
                last_error = str(exc)
                if attempt <= retry:
                    time.sleep(retry_delay * attempt)
        raise RuntimeError(f"Pinecone upsert failed: {last_error}")
