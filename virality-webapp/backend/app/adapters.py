from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import torch

from .constants import ModelSpec
from .metadata import FieldDescriptor, KNOWN_BOOLEAN_FIELDS, parse_number


@dataclass
class AdapterInput:
    metadata: dict[str, Any]
    fused_by_strategy: dict[str, np.ndarray]
    video_vec: np.ndarray
    audio_vec: np.ndarray
    text_vec: np.ndarray
    text_present: int


class ModelAdapter(Protocol):
    spec: ModelSpec

    @property
    def needs_fused_vector(self) -> bool: ...

    def metadata_fields(self) -> dict[str, FieldDescriptor]: ...

    def predict_log(self, payload: AdapterInput) -> float: ...


@dataclass
class BaseAdapter:
    spec: ModelSpec

    @property
    def needs_fused_vector(self) -> bool:
        return False

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.spec.model_family,
            "strategy": self.spec.strategy,
            "run_id": self.spec.run_id,
            "horizon_days": self.spec.horizon_days,
        }


def _infer_field_type(name: str, numeric: bool) -> str:
    if name in KNOWN_BOOLEAN_FIELDS:
        return "boolean"
    if numeric:
        return "number"
    return "string"


def _extract_sklearn_metadata_fields(pipe: Any) -> dict[str, FieldDescriptor]:
    fields: dict[str, FieldDescriptor] = {}

    pre = pipe.named_steps.get("pre") if hasattr(pipe, "named_steps") else None
    if pre is None:
        return fields

    transformers = getattr(pre, "transformers_", None) or getattr(pre, "transformers", [])

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for name, _transformer, cols in transformers:
        col_list = [str(c) for c in cols]
        if name == "num":
            numeric_cols.extend(col_list)
        elif name == "cat":
            categorical_cols.extend(col_list)

    numeric_cols = [c for c in numeric_cols if not c.startswith("fused_") and not c.startswith("proj_")]

    cat_options: dict[str, list[str] | None] = {c: None for c in categorical_cols}
    cat_pipeline = getattr(pre, "named_transformers_", {}).get("cat") if hasattr(pre, "named_transformers_") else None
    if cat_pipeline is not None and hasattr(cat_pipeline, "named_steps"):
        ohe = cat_pipeline.named_steps.get("onehot")
        if ohe is not None and hasattr(ohe, "categories_"):
            for idx, col in enumerate(categorical_cols):
                if idx >= len(ohe.categories_):
                    continue
                options = [str(v) for v in ohe.categories_[idx] if str(v) != "<NA>"]
                unique = sorted(set(options))
                cat_options[col] = unique if 0 < len(unique) <= 100 else None

    for col in sorted(set(numeric_cols)):
        fields[col] = FieldDescriptor(name=col, type=_infer_field_type(col, numeric=True), required=False)

    for col in sorted(set(categorical_cols)):
        fields[col] = FieldDescriptor(
            name=col,
            type=_infer_field_type(col, numeric=False),
            required=False,
            options=cat_options.get(col),
        )

    return fields


def _extract_bundle_metadata_fields(preprocess_bundle: dict[str, Any]) -> dict[str, FieldDescriptor]:
    fields: dict[str, FieldDescriptor] = {}

    numeric_cols = [str(c) for c in preprocess_bundle.get("numeric_cols", [])]
    categorical_cols = [str(c) for c in preprocess_bundle.get("categorical_cols", [])]
    cat_mappings = preprocess_bundle.get("cat_mappings", {}) or {}

    for col in numeric_cols:
        fields[col] = FieldDescriptor(name=col, type=_infer_field_type(col, numeric=True), required=False)

    for col in categorical_cols:
        options_raw = list((cat_mappings.get(col) or {}).keys())
        options = sorted(set(str(v) for v in options_raw if str(v) != "<NA>"))
        fields[col] = FieldDescriptor(
            name=col,
            type=_infer_field_type(col, numeric=False),
            required=False,
            options=options if 0 < len(options) <= 100 else None,
        )

    return fields


def _bundle_to_arrays(preprocess_bundle: dict[str, Any], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    numeric_cols = [str(c) for c in preprocess_bundle.get("numeric_cols", [])]
    categorical_cols = [str(c) for c in preprocess_bundle.get("categorical_cols", [])]

    num_median = preprocess_bundle.get("num_median", {}) or {}
    num_mean = preprocess_bundle.get("num_mean", {}) or {}
    num_std = preprocess_bundle.get("num_std", {}) or {}
    cat_mappings = preprocess_bundle.get("cat_mappings", {}) or {}

    num_values: list[float] = []
    for col in numeric_cols:
        parsed = parse_number(metadata.get(col))
        median = float(num_median.get(col, 0.0))
        mean = float(num_mean.get(col, 0.0))
        std = float(num_std.get(col, 1.0))
        if std <= 1e-8:
            std = 1.0
        value = median if parsed is None else parsed
        num_values.append((float(value) - mean) / std)

    cat_values: list[int] = []
    for col in categorical_cols:
        mapping = cat_mappings.get(col, {}) or {}
        raw_val = metadata.get(col)
        normalized = "<NA>" if raw_val is None else str(raw_val)
        cat_values.append(int(mapping.get(normalized, 0)))

    num_arr = np.asarray(num_values, dtype=np.float32).reshape(1, -1) if num_values else np.zeros((1, 0), dtype=np.float32)
    cat_arr = np.asarray(cat_values, dtype=np.int64).reshape(1, -1) if cat_values else np.zeros((1, 0), dtype=np.int64)
    return num_arr, cat_arr


@dataclass
class RidgeAdapter(BaseAdapter):
    pipe: Any
    _fields: dict[str, FieldDescriptor]

    @property
    def needs_fused_vector(self) -> bool:
        return True

    def metadata_fields(self) -> dict[str, FieldDescriptor]:
        return self._fields

    def predict_log(self, payload: AdapterInput) -> float:
        fused = np.asarray(payload.fused_by_strategy[self.spec.strategy], dtype=np.float32).reshape(-1)
        row = {k: payload.metadata.get(k) for k in self._fields.keys()}
        for idx, value in enumerate(fused):
            row[f"fused_{idx}"] = float(value)
        X = pd.DataFrame([row])
        return float(self.pipe.predict(X)[0])


@dataclass
class GBDTAdapter(BaseAdapter):
    pipe: Any
    projector: Callable[[np.ndarray], np.ndarray]
    _fields: dict[str, FieldDescriptor]

    @property
    def needs_fused_vector(self) -> bool:
        return True

    def metadata_fields(self) -> dict[str, FieldDescriptor]:
        return self._fields

    def predict_log(self, payload: AdapterInput) -> float:
        fused = np.asarray(payload.fused_by_strategy[self.spec.strategy], dtype=np.float32).reshape(-1)
        projected = np.asarray(self.projector(fused), dtype=np.float32).reshape(-1)
        row = {k: payload.metadata.get(k) for k in self._fields.keys()}
        for idx, value in enumerate(projected):
            row[f"proj_{idx}"] = float(value)
        X = pd.DataFrame([row])
        return float(self.pipe.predict(X)[0])


@dataclass
class ConcatMLPAdapter(BaseAdapter):
    model: torch.nn.Module
    preprocess_bundle: dict[str, Any]
    _fields: dict[str, FieldDescriptor]

    @property
    def needs_fused_vector(self) -> bool:
        return True

    def metadata_fields(self) -> dict[str, FieldDescriptor]:
        return self._fields

    def predict_log(self, payload: AdapterInput) -> float:
        fused = np.asarray(payload.fused_by_strategy[self.spec.strategy], dtype=np.float32).reshape(1, -1)
        num_arr, cat_arr = _bundle_to_arrays(self.preprocess_bundle, payload.metadata)

        fused_t = torch.from_numpy(fused).float()
        num_t = torch.from_numpy(num_arr).float()
        cat_t = torch.from_numpy(cat_arr).long()

        self.model.eval()
        with torch.no_grad():
            out = self.model(fused_t, num_t, cat_t)
        return float(out.detach().cpu().numpy().reshape(-1)[0])


@dataclass
class GatedFusionMLPAdapter(BaseAdapter):
    model: torch.nn.Module
    preprocess_bundle: dict[str, Any]
    _fields: dict[str, FieldDescriptor]

    def metadata_fields(self) -> dict[str, FieldDescriptor]:
        return self._fields

    def predict_log(self, payload: AdapterInput) -> float:
        num_arr, cat_arr = _bundle_to_arrays(self.preprocess_bundle, payload.metadata)

        video_t = torch.from_numpy(np.asarray(payload.video_vec, dtype=np.float32).reshape(1, -1)).float()
        audio_t = torch.from_numpy(np.asarray(payload.audio_vec, dtype=np.float32).reshape(1, -1)).float()
        text_t = torch.from_numpy(np.asarray(payload.text_vec, dtype=np.float32).reshape(1, -1)).float()
        num_t = torch.from_numpy(num_arr).float()
        cat_t = torch.from_numpy(cat_arr).long()
        text_present_t = torch.tensor([[float(int(payload.text_present))]], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            out = self.model(video_t, audio_t, text_t, num_t, cat_t, text_present_t)
        return float(out.detach().cpu().numpy().reshape(-1)[0])


@dataclass
class TorchProjector:
    model: torch.nn.Module

    def __call__(self, fused: np.ndarray) -> np.ndarray:
        vec = np.asarray(fused, dtype=np.float32).reshape(1, -1)
        inp = torch.from_numpy(vec).float()
        self.model.eval()
        with torch.no_grad():
            _pred, proj = self.model(inp)
        return proj.detach().cpu().numpy().reshape(-1).astype(np.float32)


def metadata_fields_for_sklearn(pipe: Any) -> dict[str, FieldDescriptor]:
    return _extract_sklearn_metadata_fields(pipe)


def metadata_fields_for_bundle(preprocess_bundle: dict[str, Any]) -> dict[str, FieldDescriptor]:
    return _extract_bundle_metadata_fields(preprocess_bundle)
