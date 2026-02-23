"""Fuse S3 modality embeddings by video_id intersection and publish sharded fused outputs."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, DeltaItem, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Data.common.stage_state import StageStateDB, compute_stage_delta  # noqa: E402

DEFAULT_STATE_DB = REPO_ROOT / "state" / "fusion.sqlite"
DEFAULT_STATE_S3_KEY = "clipfarm/state/fusion.sqlite"
DEFAULT_EMB_PREFIX = "clipfarm/embeddings"
DEFAULT_FUSED_PREFIX = "clipfarm/fused"
DEFAULT_FUSED_SHARDS_PREFIX = "clipfarm/fused/shards"


@dataclass
class FusedRecord:
    item: DeltaItem
    fused_vec: np.ndarray
    video_key: str
    audio_key: str
    text_key: str
    fused_key: str = ""
    shard_idx: int = -1
    shard_n: int = 0


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _merge_manifest(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    for col in ("video_id", "source_hash", "captured_at"):
        if col not in merged.columns:
            merged[col] = ""
    merged["captured_at"] = merged["captured_at"].fillna("").astype(str)
    merged = merged.sort_values(["video_id", "source_hash", "captured_at", "fused_key"])
    merged = merged.drop_duplicates(subset=["video_id", "source_hash"], keep="last")
    return merged.reset_index(drop=True)


def _next_shard_index(s3: S3ArtifactStore, shard_partition_prefix: str) -> int:
    keys = s3.list_keys(shard_partition_prefix.rstrip("/") + "/")
    max_idx = -1
    for key in keys:
        m = re.search(r"part-(\d+)\.npz$", key)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _chunked(records: list[FusedRecord], shard_size: int) -> list[list[FusedRecord]]:
    if shard_size <= 0:
        return [records]
    return [records[i : i + shard_size] for i in range(0, len(records), shard_size)]


def _load_back_vector(s3: S3ArtifactStore, key: str, idx: int, cache_dir: Path, cache: dict[str, np.lib.npyio.NpzFile]) -> np.ndarray:
    if key not in cache:
        local = cache_dir / Path(key).name
        s3.download_file(key, local)
        cache[key] = np.load(local, allow_pickle=True)
    vectors = cache[key]["vectors"]
    return np.asarray(vectors[idx], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse delta embeddings from S3 into deterministic sharded vectors")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--emb_prefix", default=DEFAULT_EMB_PREFIX)
    parser.add_argument("--fused_prefix", default=DEFAULT_FUSED_PREFIX)
    parser.add_argument("--fused_shards_prefix", default=DEFAULT_FUSED_SHARDS_PREFIX)
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--state_s3_key", default=DEFAULT_STATE_S3_KEY)
    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--run_date", default="")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    run_date = (args.run_date or "").strip() or _today_utc_str()
    shard_partition_prefix = f"{args.fused_shards_prefix.strip('/')}/date={run_date}"
    manifest_key = f"{args.fused_prefix.strip('/')}/fused_manifest.parquet"

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    state_path = Path(args.state_db)
    s3.restore_state_if_exists(args.state_s3_key, state_path)
    state = StageStateDB(state_path)

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_stage_delta(items, state, terminal_statuses=("success",), max_items=(args.max_items or None))

        success = 0
        failed = 0
        missing_modality = 0
        added_manifest_rows = 0
        shard_files_written = 0
        expected_fused_dim = -1
        fused_candidates: list[FusedRecord] = []

        with tempfile.TemporaryDirectory(prefix="fuse_delta_") as tmp_dir:
            tmp = Path(tmp_dir)

            # Collect candidates first. Do not mark success yet.
            for item in delta:
                v_key = f"{args.emb_prefix.strip('/')}/video/{item.video_id}.npy"
                a_key = f"{args.emb_prefix.strip('/')}/audio/{item.video_id}.npy"
                t_key = f"{args.emb_prefix.strip('/')}/text/{item.video_id}.npy"

                if not (s3.exists(v_key) and s3.exists(a_key) and s3.exists(t_key)):
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="missing_modality",
                        error=f"requires_all:{v_key}|{a_key}|{t_key}",
                        artifact_key="",
                        vector_id=f"fused:{item.video_id}",
                    )
                    missing_modality += 1
                    continue

                local_v = tmp / f"{item.video_id}.video.npy"
                local_a = tmp / f"{item.video_id}.audio.npy"
                local_t = tmp / f"{item.video_id}.text.npy"
                try:
                    s3.download_file(v_key, local_v)
                    s3.download_file(a_key, local_a)
                    s3.download_file(t_key, local_t)

                    v = _load_npy(local_v)
                    a = _load_npy(local_a)
                    t = _load_npy(local_t)
                    fused = np.concatenate([v, a, t], axis=0).astype(np.float32)

                    # Validation: deterministic concat dim must equal sum of modality dims.
                    if fused.shape[0] != (v.shape[0] + a.shape[0] + t.shape[0]):
                        raise ValueError(
                            f"Fused dim mismatch for {item.video_id}: {fused.shape[0]} != {v.shape[0]}+{a.shape[0]}+{t.shape[0]}"
                        )
                    if expected_fused_dim < 0:
                        expected_fused_dim = int(fused.shape[0])
                    elif expected_fused_dim != int(fused.shape[0]):
                        raise ValueError(
                            f"Inconsistent fused_dim: {fused.shape[0]} != {expected_fused_dim} for {item.video_id}"
                        )

                    fused_candidates.append(
                        FusedRecord(
                            item=item,
                            fused_vec=fused,
                            video_key=v_key,
                            audio_key=a_key,
                            text_key=t_key,
                        )
                    )
                except Exception as exc:
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="fail",
                        error=str(exc),
                        artifact_key="",
                        vector_id=f"fused:{item.video_id}",
                    )
                    failed += 1
                finally:
                    for p in (local_v, local_a, local_t):
                        if p.exists():
                            p.unlink()

            if fused_candidates:
                start_idx = _next_shard_index(s3, shard_partition_prefix)
                chunks = _chunked(fused_candidates, args.shard_size)

                # 1) Write shard files first
                for chunk_idx, chunk in enumerate(chunks):
                    shard_part = start_idx + chunk_idx
                    shard_key = f"{shard_partition_prefix}/part-{shard_part:04d}.npz"
                    local_shard = tmp / f"part-{shard_part:04d}.npz"

                    ids = np.array([r.item.video_id for r in chunk])
                    vectors = np.stack([r.fused_vec for r in chunk]).astype(np.float32)
                    captured = np.array([r.item.captured_at for r in chunk])

                    np.savez(local_shard, video_ids=ids, vectors=vectors, captured_at=captured)
                    s3.upload_file(local_shard, shard_key)
                    shard_files_written += 1

                    for i, rec in enumerate(chunk):
                        rec.fused_key = shard_key
                        rec.shard_idx = i
                        rec.shard_n = len(chunk)

                # 2) Merge manifest rows for exactly those shard contents
                new_manifest_rows = []
                for rec in fused_candidates:
                    new_manifest_rows.append(
                        {
                            "video_id": rec.item.video_id,
                            "captured_at": rec.item.captured_at,
                            "source_hash": rec.item.source_hash,
                            "fused_dim": int(rec.fused_vec.shape[0]),
                            "fused_key": rec.fused_key,
                            "shard_idx": int(rec.shard_idx),
                            "shard_n": int(rec.shard_n),
                            "video_emb_key": rec.video_key,
                            "audio_emb_key": rec.audio_key,
                            "text_emb_key": rec.text_key,
                        }
                    )
                new_df = pd.DataFrame(new_manifest_rows)

                old_df = pd.DataFrame()
                local_manifest = tmp / "fused_manifest.parquet"
                if s3.exists(manifest_key):
                    s3.download_file(manifest_key, local_manifest)
                    old_df = pd.read_parquet(local_manifest)

                old_keys = set()
                if not old_df.empty and {"video_id", "source_hash"}.issubset(old_df.columns):
                    old_keys = {(str(r.video_id), str(r.source_hash)) for r in old_df.itertuples(index=False)}
                new_keys = {(str(r.video_id), str(r.source_hash)) for r in new_df.itertuples(index=False)}

                merged_df = _merge_manifest(old_df, new_df)
                merged_df.to_parquet(local_manifest, index=False)
                s3.upload_file(local_manifest, manifest_key)

                added_manifest_rows = len(new_keys - old_keys)

                # 3) Mark state success only after manifest is safely updated.
                for rec in fused_candidates:
                    state.upsert(
                        rec.item.video_id,
                        rec.item.source_hash,
                        status="success",
                        error="",
                        artifact_key=rec.fused_key,
                        vector_id=f"fused:{rec.item.video_id}",
                    )
                    success += 1

                # Lightweight validation: load back a few vectors by manifest pointers.
                cache: dict[str, np.lib.npyio.NpzFile] = {}
                for rec in fused_candidates[: min(3, len(fused_candidates))]:
                    check = _load_back_vector(s3, rec.fused_key, rec.shard_idx, tmp, cache)
                    if int(check.shape[0]) != int(rec.fused_vec.shape[0]):
                        raise ValueError(
                            f"Shard pointer validation failed for {rec.item.video_id}: {check.shape[0]} != {rec.fused_vec.shape[0]}"
                        )

        print("Fusion summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"fused_candidates: {len(fused_candidates)}")
        print(f"success: {success}")
        print(f"missing_modality: {missing_modality}")
        print(f"failed: {failed}")
        print(f"shard_files_written: {shard_files_written}")
        print(f"manifest_rows_added: {added_manifest_rows}")
        print(f"fused_dim: {expected_fused_dim if expected_fused_dim > 0 else 0}")
        print(f"manifest_s3_key: {manifest_key}")
        print(f"shards_prefix: {shard_partition_prefix}/")
        print(f"state_db: {args.state_db}")
        print(f"state_s3_key: {args.state_s3_key}")
    finally:
        state.close()
        s3.persist_state(state_path, args.state_s3_key)


if __name__ == "__main__":
    main()
