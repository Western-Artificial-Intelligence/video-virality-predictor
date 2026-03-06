from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    repo_root: Path
    model_s3_bucket: str
    model_s3_region: str
    model_snapshot_prefix: str
    model_cache_dir: Path
    asr_backend: str
    asr_model: str
    max_upload_mb: int
    text_dim: int
    auto_load_models_on_startup: bool
    cors_allow_origins: tuple[str, ...]


def _split_csv(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ("*",)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_settings() -> AppSettings:
    repo_root = Path(__file__).resolve().parents[3]
    cache_dir_raw = os.getenv(
        "VIRALITY_MODEL_CACHE_DIR",
        str(repo_root / "state" / "virality_webapp" / "model_cache"),
    )
    auto_load_raw = os.getenv("VIRALITY_WEBAPP_SKIP_STARTUP_LOAD", "0").strip().lower()
    auto_load = auto_load_raw not in {"1", "true", "yes", "y"}

    model_bucket = os.getenv("MODEL_S3_BUCKET", "").strip() or os.getenv("S3_BUCKET", "").strip()

    return AppSettings(
        repo_root=repo_root,
        model_s3_bucket=model_bucket,
        model_s3_region=os.getenv("MODEL_S3_REGION", "").strip() or os.getenv("AWS_REGION", "").strip(),
        model_snapshot_prefix=os.getenv("MODEL_SNAPSHOT_PREFIX", "clipfarm/models/snapshots").strip("/"),
        model_cache_dir=Path(cache_dir_raw).expanduser(),
        asr_backend=os.getenv("VIRALITY_ASR_BACKEND", "auto").strip(),
        asr_model=os.getenv("VIRALITY_ASR_MODEL", "small").strip(),
        max_upload_mb=max(1, int(os.getenv("VIRALITY_MAX_UPLOAD_MB", "512") or "512")),
        text_dim=max(1, int(os.getenv("VIRALITY_TEXT_DIM", "768") or "768")),
        auto_load_models_on_startup=auto_load,
        cors_allow_origins=_split_csv(os.getenv("VIRALITY_CORS_ALLOW_ORIGINS", "*")),
    )
