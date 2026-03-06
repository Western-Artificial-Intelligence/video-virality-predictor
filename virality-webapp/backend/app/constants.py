from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Horizon = Literal[7, 30]
Mode = Literal["fast", "full"]
ModelFamily = Literal["gbdt", "ridge", "concat_mlp", "gated_fusion_mlp"]
FusionStrategy = Literal["concat", "sum_pool", "max_pool"]


@dataclass(frozen=True)
class ModelSpec:
    model_family: ModelFamily
    strategy: FusionStrategy
    run_id: str
    horizon_days: Horizon


FAST_RUN_GBDT = "colab-gbdt-20260306T070623Z"
FULL_RUN_CONCAT = "colab-concat_mlp-20260306T054620Z"
FULL_RUN_GATED = "colab-gated_fusion_mlp-20260306T054937Z"
FULL_RUN_RIDGE = "colab-ridge-20260306T070324Z"
FULL_RUN_GBDT = "colab-gbdt-20260306T070623Z"


MODE_SPECS: dict[Mode, dict[Horizon, list[ModelSpec]]] = {
    "fast": {
        7: [
            ModelSpec(
                model_family="gbdt",
                strategy="concat",
                run_id=FAST_RUN_GBDT,
                horizon_days=7,
            )
        ],
        30: [
            ModelSpec(
                model_family="gbdt",
                strategy="concat",
                run_id=FAST_RUN_GBDT,
                horizon_days=30,
            )
        ],
    },
    "full": {
        7: [
            ModelSpec(model_family="gbdt", strategy="concat", run_id=FULL_RUN_GBDT, horizon_days=7),
            ModelSpec(model_family="concat_mlp", strategy="max_pool", run_id=FULL_RUN_CONCAT, horizon_days=7),
            ModelSpec(model_family="gated_fusion_mlp", strategy="concat", run_id=FULL_RUN_GATED, horizon_days=7),
            ModelSpec(model_family="ridge", strategy="sum_pool", run_id=FULL_RUN_RIDGE, horizon_days=7),
        ],
        30: [
            ModelSpec(model_family="gbdt", strategy="concat", run_id=FULL_RUN_GBDT, horizon_days=30),
            ModelSpec(model_family="concat_mlp", strategy="max_pool", run_id=FULL_RUN_CONCAT, horizon_days=30),
            ModelSpec(model_family="gated_fusion_mlp", strategy="concat", run_id=FULL_RUN_GATED, horizon_days=30),
            ModelSpec(model_family="ridge", strategy="sum_pool", run_id=FULL_RUN_RIDGE, horizon_days=30),
        ],
    },
}


def is_valid_mode(value: str) -> bool:
    return value in MODE_SPECS
