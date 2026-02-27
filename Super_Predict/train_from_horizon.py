"""Train virality model from horizon metadata + fused embeddings in S3."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Super_Predict.virality_model import ViralityConfig, train_virality_model  # noqa: E402

FEATURE_WHITELIST = [
    "channel_subscriber_count",
    "channel_video_count",
    "channel_view_count",
    "channel_age_days",
    "channel_country",
    "title_length",
    "description_length",
    "emoji_count",
    "has_hashtags",
    "has_shorts_hashtag",
    "has_clickbait_words",
    "hashtag_count",
    "duration_seconds",
    "default_language",
    "default_audio_language",
    "caption_available",
    "thumb_width",
    "thumb_height",
    "thumb_aspect_ratio",
    "is_vertical_thumb",
    "published_hour",
    "published_dayofweek",
]


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _safe_bool_to_int(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x: 1
        if str(x).strip().lower() in {"1", "true", "t", "yes", "y"}
        else (0 if str(x).strip().lower() in {"0", "false", "f", "no", "n"} else np.nan)
    )


def build_training_features(metadata_csv: Path, target_horizon_days: int, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "horizon_days" not in df.columns or "horizon_view_count" not in df.columns:
        raise ValueError("metadata CSV is missing horizon label columns")

    df = df[df["horizon_days"] == target_horizon_days].copy()
    if df.empty:
        raise ValueError(f"No rows for horizon_days={target_horizon_days}")

    if "captured_at" in df.columns:
        df["_captured_at_dt"] = _to_datetime(df["captured_at"])
        df = df.sort_values("_captured_at_dt")
    else:
        df["_captured_at_dt"] = pd.NaT
    df = df.drop_duplicates(subset=["video_id"], keep="last")

    if "published_at" in df.columns:
        pub_dt = _to_datetime(df["published_at"])
        df["published_hour"] = pub_dt.dt.hour.astype("Int64")
        df["published_dayofweek"] = pub_dt.dt.dayofweek.astype("Int64")
    else:
        df["published_hour"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["published_dayofweek"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    for col in ("has_hashtags", "has_shorts_hashtag", "has_clickbait_words", "caption_available", "is_vertical_thumb"):
        if col in df.columns:
            df[col] = _safe_bool_to_int(df[col])

    target = pd.to_numeric(df["horizon_view_count"], errors="coerce").fillna(0).clip(lower=0)
    df[target_col] = target.map(lambda x: float(math.log1p(x)))

    keep_cols = ["video_id", target_col] + [c for c in FEATURE_WHITELIST if c in df.columns]
    out = df[keep_cols].copy()
    return out


def _build_fused_pt_from_manifest(
    s3,
    manifest_df: pd.DataFrame,
    train_video_ids: list[str],
    local_fused_pt: Path,
    cache_dir: Path,
) -> tuple[int, int]:
    required = {"video_id", "fused_key", "shard_idx"}
    if not required.issubset(manifest_df.columns):
        missing = sorted(required - set(manifest_df.columns))
        raise ValueError(f"Fused manifest missing required columns for sharded load: {missing}")

    mf = manifest_df.copy()
    mf["video_id"] = mf["video_id"].astype(str)
    if "captured_at" in mf.columns:
        mf = mf.sort_values("captured_at")
    mf = mf.drop_duplicates(subset=["video_id"], keep="last")
    wanted = set(str(v) for v in train_video_ids)
    mf = mf[mf["video_id"].isin(wanted)].copy()
    if mf.empty:
        raise ValueError("No manifest rows matched training video IDs")

    shard_cache: dict[str, np.lib.npyio.NpzFile] = {}
    downloaded_shards = 0

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for row in mf.itertuples(index=False):
        key = str(row.fused_key)
        idx = int(row.shard_idx)

        if key not in shard_cache:
            local_shard = cache_dir / Path(key).name
            s3.download_file(key, local_shard)
            shard_cache[key] = np.load(local_shard, allow_pickle=True)
            downloaded_shards += 1

        shard = shard_cache[key]
        shard_vectors = shard["vectors"]
        if idx < 0 or idx >= shard_vectors.shape[0]:
            raise IndexError(f"Invalid shard_idx={idx} for key={key} with shard_n={shard_vectors.shape[0]}")

        vec = np.asarray(shard_vectors[idx], dtype=np.float32).reshape(-1)
        ids.append(str(row.video_id))
        vectors.append(vec)

    if not ids:
        raise ValueError("No vectors reconstructed from fused manifest")

    fused_tensor = torch.tensor(np.stack(vectors).astype(np.float32))
    torch.save({"ids": ids, "fused": fused_tensor}, local_fused_pt)
    return len(ids), downloaded_shards


def main() -> None:
    from Data.common.s3_artifact_store import S3ArtifactStore  # local import for testability

    parser = argparse.ArgumentParser(description="Train virality model from horizon metadata and fused embeddings")
    parser.add_argument("--metadata_csv", default="Data/raw/Metadata/shorts_metadata_horizon.csv")
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--fused_manifest_s3_key", default="clipfarm/fused/fused_manifest.parquet")
    parser.add_argument("--fused_pt_s3_key", default="clipfarm/fused/fused.pt")
    parser.add_argument("--model_out_prefix", default="clipfarm/models/virality/latest")
    parser.add_argument("--target_horizon_days", type=int, default=7)
    parser.add_argument("--fusion_strategy", default="", help="Optional fusion strategy label for reporting")
    args = parser.parse_args()

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    target_col = f"log_view_count_h{args.target_horizon_days}d"

    with tempfile.TemporaryDirectory(prefix="train_horizon_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_manifest = tmp / "fused_manifest.parquet"
        local_fused_pt = tmp / "fused.pt"
        local_model = tmp / "model.joblib"
        local_metrics = tmp / "metrics.json"
        local_feature_imp = tmp / "feature_importance.parquet"
        local_train_manifest = tmp / "train_manifest.json"

        if not s3.exists(args.fused_manifest_s3_key):
            raise FileNotFoundError(f"Missing fused manifest in S3: {args.fused_manifest_s3_key}")

        s3.download_file(args.fused_manifest_s3_key, local_manifest)

        fused_manifest = pd.read_parquet(local_manifest)
        if fused_manifest.empty:
            raise ValueError("Fused manifest is empty")

        features = build_training_features(Path(args.metadata_csv), args.target_horizon_days, target_col=target_col)
        features = features[features["video_id"].astype(str).isin(fused_manifest["video_id"].astype(str))].copy()
        if features.empty:
            raise ValueError("No training rows after joining with fused manifest")

        reconstructed_count, downloaded_shards = _build_fused_pt_from_manifest(
            s3=s3,
            manifest_df=fused_manifest,
            train_video_ids=features["video_id"].astype(str).tolist(),
            local_fused_pt=local_fused_pt,
            cache_dir=tmp / "shard_cache",
        )

        cfg = ViralityConfig(
            target_col=target_col,
            categorical_cols=[c for c in ("channel_country", "default_language", "default_audio_language") if c in features.columns],
        )
        out = train_virality_model(
            fused_pt_path=str(local_fused_pt),
            features_df=features,
            config=cfg,
        )

        joblib.dump(out["pipeline"], local_model)
        out["feature_importance"].to_parquet(local_feature_imp, index=False)

        metrics_payload = dict(out["metrics"])
        metrics_payload["target_horizon_days"] = int(args.target_horizon_days)
        metrics_payload["target_col"] = target_col
        metrics_payload["train_rows_after_join"] = int(len(features))
        if args.fusion_strategy:
            metrics_payload["fusion_strategy"] = args.fusion_strategy
        local_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        train_manifest_payload = {
            "target_horizon_days": int(args.target_horizon_days),
            "target_col": target_col,
            "fused_manifest_s3_key": args.fused_manifest_s3_key,
            "fused_pt_s3_key": "(rebuilt_from_sharded_manifest_locally_for_training)",
            "model_s3_key": f"{args.model_out_prefix.strip('/')}/model.joblib",
            "metrics_s3_key": f"{args.model_out_prefix.strip('/')}/metrics.json",
            "feature_importance_s3_key": f"{args.model_out_prefix.strip('/')}/feature_importance.parquet",
            "train_ids_sample": out["train_ids"][:50],
            "test_ids_sample": out["test_ids"][:50],
            "train_count": len(out["train_ids"]),
            "test_count": len(out["test_ids"]),
            "reconstructed_vectors_count": reconstructed_count,
            "downloaded_shards_count": downloaded_shards,
        }
        if args.fusion_strategy:
            train_manifest_payload["fusion_strategy"] = args.fusion_strategy
        local_train_manifest.write_text(json.dumps(train_manifest_payload, indent=2), encoding="utf-8")

        s3.upload_file(local_model, f"{args.model_out_prefix.strip('/')}/model.joblib")
        s3.upload_file(local_metrics, f"{args.model_out_prefix.strip('/')}/metrics.json")
        s3.upload_file(local_feature_imp, f"{args.model_out_prefix.strip('/')}/feature_importance.parquet")
        s3.upload_file(local_train_manifest, f"{args.model_out_prefix.strip('/')}/train_manifest.json")

        print("Training summary")
        print(f"target_horizon_days: {args.target_horizon_days}")
        print(f"target_col: {target_col}")
        print(f"train_rows_after_join: {len(features)}")
        print(f"reconstructed_vectors_count: {reconstructed_count}")
        print(f"downloaded_shards_count: {downloaded_shards}")
        print(f"model_s3_key: {args.model_out_prefix.strip('/')}/model.joblib")
        print(f"metrics_s3_key: {args.model_out_prefix.strip('/')}/metrics.json")
        print(f"feature_importance_s3_key: {args.model_out_prefix.strip('/')}/feature_importance.parquet")
        print(f"train_manifest_s3_key: {args.model_out_prefix.strip('/')}/train_manifest.json")


if __name__ == "__main__":
    main()
