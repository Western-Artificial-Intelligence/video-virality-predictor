import numpy as np
import torch

from app.adapters import (
    AdapterInput,
    ConcatMLPAdapter,
    GatedFusionMLPAdapter,
    GBDTAdapter,
    RidgeAdapter,
)
from app.constants import ModelSpec
from app.metadata import FieldDescriptor


class DummyPipe:
    def __init__(self, value: float):
        self.value = value
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        return np.array([self.value], dtype=np.float32)


class DummyConcat(torch.nn.Module):
    def forward(self, fused, numeric, cat):
        batch = fused.shape[0]
        return torch.ones((batch,), dtype=torch.float32) * 1.25


class DummyGated(torch.nn.Module):
    def forward(self, video, audio, text, numeric, cat, text_present):
        batch = video.shape[0]
        return torch.ones((batch,), dtype=torch.float32) * 2.5


PAYLOAD = AdapterInput(
    metadata={"duration_seconds": "20", "channel_country": "US"},
    fused_by_strategy={
        "concat": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "sum_pool": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "max_pool": np.array([7.0, 8.0, 9.0], dtype=np.float32),
    },
    video_vec=np.array([1.0, 2.0], dtype=np.float32),
    audio_vec=np.array([3.0, 4.0], dtype=np.float32),
    text_vec=np.array([5.0, 6.0], dtype=np.float32),
    text_present=1,
)


def test_ridge_adapter_predicts_from_fused_vector():
    fields = {"duration_seconds": FieldDescriptor(name="duration_seconds", type="number")}
    adapter = RidgeAdapter(
        spec=ModelSpec(model_family="ridge", strategy="sum_pool", run_id="r", horizon_days=7),
        pipe=DummyPipe(0.7),
        _fields=fields,
    )

    pred = adapter.predict_log(PAYLOAD)

    assert np.isclose(pred, 0.7)
    assert "fused_0" in adapter.pipe.last_X.columns


def test_gbdt_adapter_projects_before_predicting():
    fields = {"duration_seconds": FieldDescriptor(name="duration_seconds", type="number")}
    adapter = GBDTAdapter(
        spec=ModelSpec(model_family="gbdt", strategy="concat", run_id="r", horizon_days=7),
        pipe=DummyPipe(1.4),
        projector=lambda fused: np.array([fused.sum(), 42.0], dtype=np.float32),
        _fields=fields,
    )

    pred = adapter.predict_log(PAYLOAD)

    assert np.isclose(pred, 1.4)
    assert "proj_0" in adapter.pipe.last_X.columns


def test_concat_mlp_adapter_predicts_with_bundle_transform():
    bundle = {
        "numeric_cols": ["duration_seconds"],
        "categorical_cols": ["channel_country"],
        "num_median": {"duration_seconds": 10.0},
        "num_mean": {"duration_seconds": 5.0},
        "num_std": {"duration_seconds": 5.0},
        "cat_mappings": {"channel_country": {"US": 1}},
    }
    fields = {
        "duration_seconds": FieldDescriptor(name="duration_seconds", type="number"),
        "channel_country": FieldDescriptor(name="channel_country", type="string"),
    }

    adapter = ConcatMLPAdapter(
        spec=ModelSpec(model_family="concat_mlp", strategy="max_pool", run_id="r", horizon_days=7),
        model=DummyConcat(),
        preprocess_bundle=bundle,
        _fields=fields,
    )

    pred = adapter.predict_log(PAYLOAD)

    assert pred == 1.25


def test_gated_mlp_adapter_predicts_with_modal_inputs():
    bundle = {
        "numeric_cols": ["duration_seconds"],
        "categorical_cols": ["channel_country"],
        "num_median": {"duration_seconds": 10.0},
        "num_mean": {"duration_seconds": 5.0},
        "num_std": {"duration_seconds": 5.0},
        "cat_mappings": {"channel_country": {"US": 1}},
    }
    fields = {
        "duration_seconds": FieldDescriptor(name="duration_seconds", type="number"),
        "channel_country": FieldDescriptor(name="channel_country", type="string"),
    }

    adapter = GatedFusionMLPAdapter(
        spec=ModelSpec(model_family="gated_fusion_mlp", strategy="concat", run_id="r", horizon_days=30),
        model=DummyGated(),
        preprocess_bundle=bundle,
        _fields=fields,
    )

    pred = adapter.predict_log(PAYLOAD)

    assert pred == 2.5
