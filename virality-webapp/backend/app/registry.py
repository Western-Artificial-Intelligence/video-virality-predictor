from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from .adapters import (
    ConcatMLPAdapter,
    GatedFusionMLPAdapter,
    GBDTAdapter,
    ModelAdapter,
    RidgeAdapter,
    TorchProjector,
    metadata_fields_for_bundle,
    metadata_fields_for_sklearn,
)
from .constants import MODE_SPECS, Horizon, ModelSpec, Mode
from .metadata import FieldDescriptor, merge_field_maps
from .settings import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Super_Predict.train_suite_from_horizon import (  # noqa: E402
    ConcatMLP,
    GatedFusionMLP,
    ProjectorRegressor,
)


@dataclass
class ModelRegistry:
    adapters_by_mode: dict[Mode, dict[Horizon, list[ModelAdapter]]]
    fields: dict[str, FieldDescriptor]

    def adapters_for(self, mode: Mode, horizon_days: Horizon) -> list[ModelAdapter]:
        return list(self.adapters_by_mode[mode][horizon_days])

    def fields_payload(self) -> list[dict[str, Any]]:
        return [desc.to_dict() for desc in self.fields.values()]


class ModelRegistryLoader:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> ModelRegistry:
        bucket = self.settings.model_s3_bucket
        if not bucket:
            raise ValueError("MODEL_S3_BUCKET (or S3_BUCKET) is required to load model artifacts")

        s3 = S3ArtifactStore(bucket=bucket, region=self.settings.model_s3_region)

        adapters_by_mode: dict[Mode, dict[Horizon, list[ModelAdapter]]] = {
            "fast": {7: [], 30: []},
            "full": {7: [], 30: []},
        }

        loaded_specs: dict[tuple[str, str, str, int], ModelAdapter] = {}

        for mode, horizon_map in MODE_SPECS.items():
            for horizon_days, specs in horizon_map.items():
                for spec in specs:
                    key = (spec.model_family, spec.strategy, spec.run_id, spec.horizon_days)
                    if key not in loaded_specs:
                        loaded_specs[key] = self._load_adapter(s3=s3, spec=spec)
                    adapters_by_mode[mode][horizon_days].append(loaded_specs[key])

        all_field_maps = []
        for mode in adapters_by_mode.values():
            for adapters in mode.values():
                for adapter in adapters:
                    all_field_maps.append(adapter.metadata_fields())

        merged_fields = merge_field_maps(all_field_maps)
        return ModelRegistry(adapters_by_mode=adapters_by_mode, fields=merged_fields)

    def _load_adapter(self, s3: S3ArtifactStore, spec: ModelSpec) -> ModelAdapter:
        local_root = self._local_root(spec)
        s3_root = self._s3_root(spec)

        required_files = ["config_used.json"]
        if spec.model_family == "gbdt":
            required_files += ["models/gbdt.joblib", "models/gbdt_projector.pt"]
        elif spec.model_family == "ridge":
            required_files += ["models/ridge.joblib"]
        elif spec.model_family == "concat_mlp":
            required_files += ["models/concat_mlp.pt"]
        elif spec.model_family == "gated_fusion_mlp":
            required_files += ["models/gated_fusion_mlp.pt"]
        else:
            raise ValueError(f"Unsupported model family: {spec.model_family}")

        for rel in required_files:
            self._download_if_missing(s3=s3, s3_key=f"{s3_root}/{rel}", local_path=local_root / rel)

        if spec.model_family == "gbdt":
            pipe = joblib.load(local_root / "models" / "gbdt.joblib")
            projector_ckpt = torch.load(local_root / "models" / "gbdt_projector.pt", map_location="cpu")
            projector = ProjectorRegressor(
                input_dim=int(projector_ckpt["input_dim"]),
                hidden_dim=256,
                projector_dim=int(projector_ckpt["projector_dim"]),
                dropout=0.1,
            )
            projector.load_state_dict(projector_ckpt["state_dict"])
            return GBDTAdapter(
                spec=spec,
                pipe=pipe,
                projector=TorchProjector(projector),
                _fields=metadata_fields_for_sklearn(pipe),
            )

        if spec.model_family == "ridge":
            pipe = joblib.load(local_root / "models" / "ridge.joblib")
            return RidgeAdapter(spec=spec, pipe=pipe, _fields=metadata_fields_for_sklearn(pipe))

        if spec.model_family == "concat_mlp":
            ckpt = torch.load(local_root / "models" / "concat_mlp.pt", map_location="cpu")
            model_cfg = ckpt["model_config"]
            model = ConcatMLP(
                fused_dim=int(model_cfg["fused_dim"]),
                numeric_dim=int(model_cfg["numeric_dim"]),
                cat_cardinalities=[int(v) for v in model_cfg["cat_cardinalities"]],
                hidden_dims=[1024, 512, 256],
                dropout=0.20,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            preprocess_bundle = ckpt["preprocess"]
            return ConcatMLPAdapter(
                spec=spec,
                model=model,
                preprocess_bundle=preprocess_bundle,
                _fields=metadata_fields_for_bundle(preprocess_bundle),
            )

        if spec.model_family == "gated_fusion_mlp":
            ckpt = torch.load(local_root / "models" / "gated_fusion_mlp.pt", map_location="cpu")
            model_cfg = ckpt["model_config"]
            model = GatedFusionMLP(
                video_dim=int(model_cfg["video_dim"]),
                audio_dim=int(model_cfg["audio_dim"]),
                text_dim=int(model_cfg["text_dim"]),
                numeric_dim=int(model_cfg["numeric_dim"]),
                cat_cardinalities=[int(v) for v in model_cfg["cat_cardinalities"]],
                tower_dim=256,
                gate_hidden=128,
                head_hidden=[256, 128],
                dropout=0.15,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            preprocess_bundle = ckpt["preprocess"]
            return GatedFusionMLPAdapter(
                spec=spec,
                model=model,
                preprocess_bundle=preprocess_bundle,
                _fields=metadata_fields_for_bundle(preprocess_bundle),
            )

        raise ValueError(f"Unsupported model family: {spec.model_family}")

    def _s3_root(self, spec: ModelSpec) -> str:
        return (
            f"{self.settings.model_snapshot_prefix}/"
            f"run_id={spec.run_id}/model={spec.model_family}/strategy={spec.strategy}/horizon={spec.horizon_days}"
        )

    def _local_root(self, spec: ModelSpec) -> Path:
        return (
            self.settings.model_cache_dir
            / f"run_id={spec.run_id}"
            / f"model={spec.model_family}"
            / f"strategy={spec.strategy}"
            / f"horizon={spec.horizon_days}"
        )

    @staticmethod
    def _download_if_missing(s3: S3ArtifactStore, s3_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            return
        s3.download_file(s3_key, local_path)
