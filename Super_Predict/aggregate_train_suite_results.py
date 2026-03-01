"""Aggregate per-job training-suite metrics from S3 into run-level comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.s3_artifact_store import S3ArtifactStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_values(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate training-suite metrics by run_id")
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--snapshot_prefix", default="clipfarm/models/snapshots")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_families", default="all")
    parser.add_argument("--strategies", default="concat,sum_pool,max_pool")
    parser.add_argument("--horizons", default="7,30")
    parser.add_argument("--rank_metric", default="rmse_log")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def build_rows(
    s3: S3ArtifactStore,
    snapshot_prefix: str,
    run_id: str,
    model_families: list[str],
    strategies: list[str],
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing_jobs: list[dict[str, Any]] = []

    for model_family in model_families:
        for strategy in strategies:
            for horizon in horizons:
                if model_family == "all":
                    metrics_key = (
                        f"{snapshot_prefix.strip('/')}/run_id={run_id}/strategy={strategy}/horizon={horizon}/metrics_summary.json"
                    )
                else:
                    metrics_key = (
                        f"{snapshot_prefix.strip('/')}/run_id={run_id}/model={model_family}/"
                        f"strategy={strategy}/horizon={horizon}/metrics_summary.json"
                    )
                if not s3.exists(metrics_key):
                    missing_jobs.append(
                        {
                            "model_family": model_family,
                            "fusion_strategy": strategy,
                            "target_horizon_days": int(horizon),
                            "metrics_key": metrics_key,
                            "status": "missing",
                        }
                    )
                    continue

                payload = s3.download_json(metrics_key, default={}) or {}
                models = payload.get("models", {})
                if not isinstance(models, dict) or not models:
                    missing_jobs.append(
                        {
                            "model_family": model_family,
                            "fusion_strategy": strategy,
                            "target_horizon_days": int(horizon),
                            "metrics_key": metrics_key,
                            "status": "empty_models",
                        }
                    )
                    continue

                for model_name, per_split in models.items():
                    val = per_split.get("val", {}) if isinstance(per_split, dict) else {}
                    test = per_split.get("test", {}) if isinstance(per_split, dict) else {}
                    rows.append(
                        {
                            "model_family": str(payload.get("model_family") or model_family),
                            "fusion_strategy": strategy,
                            "target_horizon_days": int(horizon),
                            "model": str(model_name),
                            "val_rmse_log": _to_float(val.get("rmse_log")),
                            "val_mae_log": _to_float(val.get("mae_log")),
                            "val_r2_log": _to_float(val.get("r2_log")),
                            "val_rmse_raw": _to_float(val.get("rmse_raw")),
                            "val_mae_raw": _to_float(val.get("mae_raw")),
                            "test_rmse_log": _to_float(test.get("rmse_log")),
                            "test_mae_log": _to_float(test.get("mae_log")),
                            "test_r2_log": _to_float(test.get("r2_log")),
                            "test_rmse_raw": _to_float(test.get("rmse_raw")),
                            "test_mae_raw": _to_float(test.get("mae_raw")),
                            "metrics_key": metrics_key,
                            "status": "ok",
                        }
                    )

    return rows, missing_jobs


def main() -> None:
    args = parse_args()
    if not args.s3_bucket:
        raise ValueError("--s3_bucket is required")

    model_families = parse_csv_values(args.model_families)
    strategies = parse_csv_values(args.strategies)
    horizons = [int(x) for x in parse_csv_values(args.horizons)]
    if not model_families or not strategies or not horizons:
        raise ValueError("model_families, strategies and horizons must not be empty")

    rank_col = args.rank_metric if args.rank_metric.startswith("val_") else f"val_{args.rank_metric}"

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    rows, missing_jobs = build_rows(
        s3=s3,
        snapshot_prefix=args.snapshot_prefix,
        run_id=args.run_id,
        model_families=model_families,
        strategies=strategies,
        horizons=horizons,
    )

    df = pd.DataFrame(rows)
    if not df.empty and rank_col in df.columns:
        sort_cols = [rank_col, "model_family", "fusion_strategy", "target_horizon_days", "model"]
        sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    best_by_rank = None
    best_by_mae = None
    if not df.empty:
        if rank_col in df.columns and df[rank_col].notna().any():
            best_by_rank = df.loc[df[rank_col].idxmin()].to_dict()
        if "val_mae_log" in df.columns and df["val_mae_log"].notna().any():
            best_by_mae = df.loc[df["val_mae_log"].idxmin()].to_dict()

    report = {
        "generated_at": utc_now_iso(),
        "run_id": args.run_id,
        "snapshot_prefix": args.snapshot_prefix.strip("/"),
        "model_families": model_families,
        "rank_metric": rank_col,
        "rows_count": int(len(df)),
        "missing_jobs_count": int(len(missing_jobs)),
        "missing_jobs": missing_jobs,
        "best_by_rank_metric": best_by_rank,
        "best_by_mae_log": best_by_mae,
        "rows": df.to_dict(orient="records"),
    }

    with tempfile.TemporaryDirectory(prefix="train_compare_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = tmp_dir

        json_path = out_dir / "metrics_comparison.json"
        csv_path = out_dir / "metrics_comparison.csv"
        best_mae_path = out_dir / "best_by_mae_log.json"

        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        best_mae_payload = {
            "generated_at": utc_now_iso(),
            "run_id": args.run_id,
            "best_by_mae_log": best_by_mae,
        }
        best_mae_path.write_text(json.dumps(best_mae_payload, indent=2), encoding="utf-8")

        comparison_prefix = f"{args.snapshot_prefix.strip('/')}/run_id={args.run_id}/comparison"
        s3.upload_file(json_path, f"{comparison_prefix}/metrics_comparison.json")
        s3.upload_file(csv_path, f"{comparison_prefix}/metrics_comparison.csv")
        s3.upload_file(best_mae_path, f"{comparison_prefix}/best_by_mae_log.json")

        print("Training comparison summary")
        print(f"run_id: {args.run_id}")
        print(f"rank_metric: {rank_col}")
        print(f"rows_count: {len(df)}")
        print(f"missing_jobs_count: {len(missing_jobs)}")
        if best_by_rank:
            print(
                "best_by_rank_metric: "
                f"{best_by_rank.get('model_family')} {best_by_rank.get('fusion_strategy')} "
                f"h={best_by_rank.get('target_horizon_days')} "
                f"{best_by_rank.get('model')} ({rank_col}={best_by_rank.get(rank_col)})"
            )
        if best_by_mae:
            print(
                "best_by_mae_log: "
                f"{best_by_mae.get('model_family')} {best_by_mae.get('fusion_strategy')} "
                f"h={best_by_mae.get('target_horizon_days')} "
                f"{best_by_mae.get('model')} (val_mae_log={best_by_mae.get('val_mae_log')})"
            )
        print(f"comparison_json_s3_key: {comparison_prefix}/metrics_comparison.json")
        print(f"comparison_csv_s3_key: {comparison_prefix}/metrics_comparison.csv")
        print(f"best_by_mae_log_s3_key: {comparison_prefix}/best_by_mae_log.json")


if __name__ == "__main__":
    main()
