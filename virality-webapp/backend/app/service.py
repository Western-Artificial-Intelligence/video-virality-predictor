from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import AdapterInput
from .constants import MODE_SPECS, Horizon, Mode, is_valid_mode
from .fusion import fuse_vectors
from .metadata import canonicalize_metadata
from .pipeline import OnlineFeaturePipeline
from .registry import ModelRegistry, ModelRegistryLoader
from .settings import AppSettings


def _log_to_raw(y_log: float) -> float:
    clipped = float(np.clip(y_log, -20.0, 30.0))
    return float(max(np.expm1(clipped), 0.0))


ROBUST_RANGE_MIN_SPREAD_LOG = 0.08
ROBUST_RANGE_PAD_SCALE = 0.5
ROBUST_RANGE_PAD_BIAS_LOG = 0.10


def _robust_padded_range(outputs: list[dict[str, Any]]) -> dict[str, float]:
    logs = sorted(float(row["prediction_log"]) for row in outputs)
    if not logs:
        raise ValueError("Cannot compute robust range for empty outputs")

    if len(logs) >= 4:
        core = logs[1:-1]
    elif len(logs) == 3:
        core = logs[1:2]
    else:
        core = logs

    core_low = float(min(core))
    core_high = float(max(core))
    spread = float(max(core_high - core_low, ROBUST_RANGE_MIN_SPREAD_LOG))
    pad = float(ROBUST_RANGE_PAD_SCALE * spread + ROBUST_RANGE_PAD_BIAS_LOG)

    padded_low_log = core_low - pad
    padded_high_log = core_high + pad
    padded_low_raw = _log_to_raw(padded_low_log)
    padded_high_raw = max(padded_low_raw, _log_to_raw(padded_high_log))

    return {
        "min_raw": float(padded_low_raw),
        "max_raw": float(padded_high_raw),
        "min_log": float(padded_low_log),
        "max_log": float(padded_high_log),
        "core_min_log": float(core_low),
        "core_max_log": float(core_high),
        "observed_min_log": float(min(logs)),
        "observed_max_log": float(max(logs)),
    }


class PredictionService:
    def __init__(
        self,
        settings: AppSettings,
        registry_loader: ModelRegistryLoader | None = None,
        feature_pipeline: OnlineFeaturePipeline | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings
        self._registry_loader = registry_loader or ModelRegistryLoader(settings)
        self._feature_pipeline = feature_pipeline or OnlineFeaturePipeline(settings)
        self._registry = registry
        self._lock = threading.Lock()

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            raise RuntimeError("Model registry not loaded")
        return self._registry

    def load_models(self) -> None:
        with self._lock:
            if self._registry is not None:
                return
            self._registry = self._registry_loader.load()

    def ensure_loaded(self) -> None:
        if self._registry is None:
            self.load_models()

    def get_schema_payload(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "modes": list(MODE_SPECS.keys()),
            "limits": {
                "max_upload_mb": self.settings.max_upload_mb,
                "allowed_extensions": [".mp4"],
            },
            "fields": self.registry.fields_payload(),
        }

    def predict(self, video_path: Path, mode: str, metadata: dict[str, Any]) -> dict[str, Any]:
        if not is_valid_mode(mode):
            raise ValueError(f"Unsupported mode: {mode}")

        typed_mode: Mode = mode  # type: ignore[assignment]
        self.ensure_loaded()

        canonical_metadata = canonicalize_metadata(metadata, self.registry.fields)
        pipeline_result = self._feature_pipeline.run(video_path=video_path, metadata=canonical_metadata)

        all_adapters = []
        for horizon in (7, 30):
            all_adapters.extend(self.registry.adapters_for(typed_mode, horizon))

        required_strategies = sorted({a.spec.strategy for a in all_adapters if a.needs_fused_vector})
        fused_by_strategy = {
            strategy: fuse_vectors(
                strategy=strategy,
                video_vec=pipeline_result.video_vec,
                audio_vec=pipeline_result.audio_vec,
                text_vec=pipeline_result.text_vec,
                text_present=pipeline_result.text_present,
                append_mask=True,
            )
            for strategy in required_strategies
        }

        payload = AdapterInput(
            metadata=canonical_metadata,
            fused_by_strategy=fused_by_strategy,
            video_vec=pipeline_result.video_vec,
            audio_vec=pipeline_result.audio_vec,
            text_vec=pipeline_result.text_vec,
            text_present=pipeline_result.text_present,
        )

        horizon_outputs: dict[Horizon, list[dict[str, Any]]] = {7: [], 30: []}
        for horizon in (7, 30):
            adapters = self.registry.adapters_for(typed_mode, horizon)
            for adapter in adapters:
                y_log = float(adapter.predict_log(payload))
                y_raw = _log_to_raw(y_log)
                horizon_outputs[horizon].append(
                    {
                        "prediction_log": y_log,
                        "prediction_raw": y_raw,
                        **adapter.provenance,
                    }
                )

        transcript_meta = pipeline_result.transcript_meta or {}
        transcript_payload = {
            "text_present": int(pipeline_result.text_present),
            "source": transcript_meta.get("source", ""),
            "model": transcript_meta.get("model", ""),
            "language": transcript_meta.get("language", ""),
            "error": pipeline_result.transcript_error,
        }

        provenance_payload = {
            "7d": [
                {
                    "model": p["model"],
                    "strategy": p["strategy"],
                    "run_id": p["run_id"],
                    "horizon_days": p["horizon_days"],
                }
                for p in horizon_outputs[7]
            ],
            "30d": [
                {
                    "model": p["model"],
                    "strategy": p["strategy"],
                    "run_id": p["run_id"],
                    "horizon_days": p["horizon_days"],
                }
                for p in horizon_outputs[30]
            ],
        }

        if typed_mode == "fast":
            return {
                "mode": typed_mode,
                "predictions_7d": horizon_outputs[7][0],
                "predictions_30d": horizon_outputs[30][0],
                "artifact_provenance": provenance_payload,
                "transcript": transcript_payload,
            }

        range_7d = _robust_padded_range(horizon_outputs[7])
        range_30d = _robust_padded_range(horizon_outputs[30])

        return {
            "mode": typed_mode,
            "predictions_7d": horizon_outputs[7],
            "predictions_30d": horizon_outputs[30],
            "range_7d": range_7d,
            "range_30d": range_30d,
            "artifact_provenance": provenance_payload,
            "transcript": transcript_payload,
        }
