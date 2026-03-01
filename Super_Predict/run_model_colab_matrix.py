"""Run one model family across (3 fusions x 2 horizons) for Colab execution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

MODEL_FAMILIES = ("concat_mlp", "gated_fusion_mlp", "ridge", "gbdt")
DEFAULT_STRATEGIES = ("concat", "sum_pool", "max_pool")
DEFAULT_HORIZONS = (7, 30)


def parse_csv_values(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single model family across fusion/horizon matrix")
    parser.add_argument("--model_family", required=True, choices=MODEL_FAMILIES)
    parser.add_argument("--metadata_csv", default="Data/raw/Metadata/shorts_metadata_horizon.csv")
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--snapshot_prefix", default="clipfarm/models/snapshots")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--strategies", default="concat,sum_pool,max_pool")
    parser.add_argument("--horizons", default="7,30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank_metric", default="rmse_log")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--projector_dim", type=int, default=128)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not args.s3_bucket:
        raise ValueError("--s3_bucket is required")

    strategies = parse_csv_values(args.strategies)
    horizons = [int(x) for x in parse_csv_values(args.horizons)]
    if not strategies:
        strategies = list(DEFAULT_STRATEGIES)
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)

    run_id = args.run_id.strip() or f"colab-{args.model_family}-{utc_stamp()}"
    python_bin = sys.executable
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "Super_Predict" / "train_suite_from_horizon.py"
    aggregate_script = repo_root / "Super_Predict" / "aggregate_train_suite_results.py"

    for strategy in strategies:
        for horizon in horizons:
            fused_manifest = f"clipfarm/fused/{strategy}/fused_manifest.parquet"
            cmd = [
                python_bin,
                str(train_script),
                "--metadata_csv",
                args.metadata_csv,
                "--s3_bucket",
                args.s3_bucket,
                "--s3_region",
                args.s3_region,
                "--fused_manifest_s3_key",
                fused_manifest,
                "--fusion_strategy",
                strategy,
                "--target_horizon_days",
                str(horizon),
                "--model_family",
                args.model_family,
                "--snapshot_prefix",
                args.snapshot_prefix,
                "--run_id",
                run_id,
                "--seed",
                str(args.seed),
                "--rank_metric",
                args.rank_metric,
                "--max_epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--projector_dim",
                str(args.projector_dim),
            ]
            run_cmd(cmd)

    agg_cmd = [
        python_bin,
        str(aggregate_script),
        "--s3_bucket",
        args.s3_bucket,
        "--s3_region",
        args.s3_region,
        "--snapshot_prefix",
        args.snapshot_prefix,
        "--run_id",
        run_id,
        "--model_families",
        args.model_family,
        "--strategies",
        ",".join(strategies),
        "--horizons",
        ",".join(str(h) for h in horizons),
        "--rank_metric",
        args.rank_metric,
    ]
    run_cmd(agg_cmd)

    print("Colab matrix run completed")
    print(f"run_id: {run_id}")
    print(f"model_family: {args.model_family}")
    print(f"strategies: {','.join(strategies)}")
    print(f"horizons: {','.join(str(h) for h in horizons)}")
    print(
        "comparison_s3_prefix: "
        f"{args.snapshot_prefix.strip('/')}/run_id={run_id}/comparison"
    )


if __name__ == "__main__":
    main()
