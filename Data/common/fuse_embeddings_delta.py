"""Fuse S3 modality embeddings with deterministic multi-strategy outputs."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, DeltaItem, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Data.common.stage_state import StageStateDB, StageStateRow  # noqa: E402

DEFAULT_STATE_DB = REPO_ROOT / "state" / "fusion.sqlite"
DEFAULT_STATE_S3_KEY_TEMPLATE = "clipfarm/state/fusion_{strategy}.sqlite"
DEFAULT_TEXT_STATE_S3_KEY = "clipfarm/state/text_downloader.sqlite"
DEFAULT_EMB_PREFIX = "clipfarm/embeddings"
DEFAULT_FUSED_PREFIX_BASE = "clipfarm/fused"
DEFAULT_STRATEGY = "concat"
DEFAULT_TEXT_DIM = 768
DEFAULT_MAX_FAIL_RETRIES = 3
DEFAULT_SAMPLE_IDS = 10
STRATEGIES = ("concat", "sum_pool", "max_pool")
TERMINAL_FUSION_STATUSES = {"success_full", "success_text_placeholder", "fail_terminal"}


@dataclass
class FusedRecord:
    item: DeltaItem
    fused_vec: np.ndarray
    fusion_status: str
    text_present: int
    text_source: str
    text_state_status: str
    text_missing_reason: str
    video_key: str
    audio_key: str
    text_key: str
    retry_count_at_write: int = 0
    fused_key: str = ""
    shard_idx: int = -1
    shard_n: int = 0


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_csv_values(raw: str) -> list[str]:
    return [v.strip() for v in (raw or "").split(",") if v.strip()]


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Embedding is empty: {path}")
    return arr


def _pad_to_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(int(dim), dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


def _fuse_vectors(
    strategy: str,
    video_vec: np.ndarray,
    audio_vec: np.ndarray,
    text_vec: np.ndarray,
    append_mask: bool,
    text_present: int,
) -> np.ndarray:
    if strategy == "concat":
        base = np.concatenate([video_vec, audio_vec, text_vec], axis=0).astype(np.float32)
    else:
        dim = max(video_vec.shape[0], audio_vec.shape[0], text_vec.shape[0])
        v = _pad_to_dim(video_vec, dim)
        a = _pad_to_dim(audio_vec, dim)
        t = _pad_to_dim(text_vec, dim)
        if strategy == "sum_pool":
            base = (v + a + t).astype(np.float32)
        elif strategy == "max_pool":
            base = np.maximum(np.maximum(v, a), t).astype(np.float32)
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    if append_mask:
        return np.concatenate([base, np.array([float(text_present)], dtype=np.float32)], axis=0).astype(np.float32)
    return base.astype(np.float32)


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


def _load_back_vector(
    s3: S3ArtifactStore,
    key: str,
    idx: int,
    cache_dir: Path,
    cache: dict[str, np.lib.npyio.NpzFile],
) -> np.ndarray:
    if key not in cache:
        local = cache_dir / Path(key).name
        s3.download_file(key, local)
        cache[key] = np.load(local, allow_pickle=True)
    vectors = cache[key]["vectors"]
    return np.asarray(vectors[idx], dtype=np.float32)


def _load_text_state_map(local_db: Path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(local_db)
    try:
        columns = {str(r[1]) for r in conn.execute("PRAGMA table_info(processed_items)")}
        if "status" not in columns:
            return {}
        if "source_hash" in columns:
            rows = conn.execute("SELECT video_id, status, source_hash FROM processed_items")
            return {str(v): (str(s or ""), str(h or "")) for v, s, h in rows}
        rows = conn.execute("SELECT video_id, status FROM processed_items")
        return {str(v): (str(s or ""), "") for v, s in rows}
    finally:
        conn.close()


def _restore_text_state_snapshot(
    s3: S3ArtifactStore,
    text_state_s3_key: str,
    tmp_dir: Path,
) -> tuple[bool, dict[str, tuple[str, str]], str]:
    key = (text_state_s3_key or "").strip()
    if not key:
        return False, {}, "empty_text_state_key"
    try:
        if not s3.exists(key):
            return False, {}, "text_state_not_found"
        local = tmp_dir / "text_downloader.sqlite"
        s3.download_file(key, local)
        return True, _load_text_state_map(local), ""
    except Exception as exc:
        return False, {}, f"text_state_load_failed:{exc}"


def _record_status(
    counters: dict[str, int],
    samples: dict[str, list[str]],
    status: str,
    video_id: str,
    sample_limit: int,
) -> None:
    counters[status] = counters.get(status, 0) + 1
    bucket = samples.setdefault(status, [])
    if len(bucket) < sample_limit:
        bucket.append(video_id)


def _resolve_state_paths(strategy: str, state_db_arg: str, state_s3_key_arg: str) -> tuple[Path, str]:
    state_path = Path(state_db_arg)
    if state_path.name == "fusion.sqlite":
        state_path = state_path.with_name(f"fusion_{strategy}.sqlite")

    state_s3_key = (state_s3_key_arg or "").strip()
    if not state_s3_key:
        state_s3_key = DEFAULT_STATE_S3_KEY_TEMPLATE.format(strategy=strategy)
    return state_path, state_s3_key


def _build_delta(
    items: list[DeltaItem],
    state_rows: dict[str, Optional[StageStateRow]],
    s3: S3ArtifactStore,
    emb_prefix: str,
    max_items: int,
) -> list[DeltaItem]:
    delta_by_video: dict[str, DeltaItem] = {}

    for item in items:
        existing = state_rows.get(item.video_id)
        if existing is None:
            delta_by_video[item.video_id] = item
            continue
        if existing.source_hash != item.source_hash or existing.status not in TERMINAL_FUSION_STATUSES:
            delta_by_video[item.video_id] = item

    for item in items:
        existing = state_rows.get(item.video_id)
        if (
            existing
            and existing.source_hash == item.source_hash
            and existing.status == "success_text_placeholder"
        ):
            text_key = f"{emb_prefix.strip('/')}/text/{item.video_id}.npy"
            if s3.exists(text_key):
                delta_by_video[item.video_id] = item

    delta = list(delta_by_video.values())
    if max_items and max_items > 0:
        delta = delta[:max_items]
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse delta embeddings from S3 using deterministic strategies")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--emb_prefix", default=DEFAULT_EMB_PREFIX)

    parser.add_argument("--fusion_strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY)
    parser.add_argument("--fused_prefix_base", default=DEFAULT_FUSED_PREFIX_BASE)

    # Backward compatibility for older callers.
    parser.add_argument("--fused_prefix", default="")
    parser.add_argument("--fused_shards_prefix", default="")

    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--state_s3_key", default="")
    parser.add_argument("--text_state_s3_key", default=DEFAULT_TEXT_STATE_S3_KEY)

    parser.add_argument("--text_dim", type=int, default=DEFAULT_TEXT_DIM)
    parser.add_argument("--append_text_presence_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--terminal_text_statuses",
        default="no_captions,fail_empty_transcript_terminal",
    )
    parser.add_argument("--max_fail_retries", type=int, default=DEFAULT_MAX_FAIL_RETRIES)
    parser.add_argument("--sample_ids_per_status", type=int, default=DEFAULT_SAMPLE_IDS)

    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--run_date", default="")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    strategy = args.fusion_strategy
    run_date = (args.run_date or "").strip() or _today_utc_str()

    if args.fused_prefix:
        fused_prefix = args.fused_prefix.strip("/")
    else:
        fused_prefix = f"{args.fused_prefix_base.strip('/')}/{strategy}"

    if args.fused_shards_prefix:
        shards_root = args.fused_shards_prefix.strip("/")
    else:
        shards_root = f"{fused_prefix}/shards"

    shard_partition_prefix = f"{shards_root}/date={run_date}"
    manifest_key = f"{fused_prefix}/fused_manifest.parquet"
    schema_key = f"{fused_prefix}/schema.json"

    state_path, state_s3_key = _resolve_state_paths(strategy, args.state_db, args.state_s3_key)

    terminal_text_statuses = set(_parse_csv_values(args.terminal_text_statuses))

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    s3.restore_state_if_exists(state_s3_key, state_path)
    state = StageStateDB(state_path)

    status_counters: dict[str, int] = {}
    status_samples: dict[str, list[str]] = {}

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        state_rows = {item.video_id: state.get(item.video_id) for item in items}
        delta = _build_delta(
            items=items,
            state_rows=state_rows,
            s3=s3,
            emb_prefix=args.emb_prefix,
            max_items=args.max_items,
        )

        shard_files_written = 0
        manifest_rows_added = 0
        manifest_rows_updated = 0

        with tempfile.TemporaryDirectory(prefix="fuse_delta_") as tmp_dir:
            tmp = Path(tmp_dir)

            text_state_available, text_state_map, text_state_error = _restore_text_state_snapshot(
                s3=s3,
                text_state_s3_key=args.text_state_s3_key,
                tmp_dir=tmp,
            )

            existing_schema = s3.download_json(schema_key, default={}) or {}
            schema_dims: dict[str, int | bool | str] = {}

            fused_candidates: list[FusedRecord] = []

            for item in delta:
                v_key = f"{args.emb_prefix.strip('/')}/video/{item.video_id}.npy"
                a_key = f"{args.emb_prefix.strip('/')}/audio/{item.video_id}.npy"
                t_key = f"{args.emb_prefix.strip('/')}/text/{item.video_id}.npy"

                if not (s3.exists(v_key) and s3.exists(a_key)):
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="missing_required_modality",
                        error=f"requires_video_audio:{v_key}|{a_key}",
                        artifact_key="",
                        vector_id=f"fused:{strategy}:{item.video_id}",
                        retry_count=0,
                    )
                    _record_status(
                        status_counters,
                        status_samples,
                        "missing_required_modality",
                        item.video_id,
                        args.sample_ids_per_status,
                    )
                    continue

                text_exists = s3.exists(t_key)
                text_present = 0
                text_source = "placeholder"
                text_state_status = ""
                text_missing_reason = ""

                if text_exists:
                    text_present = 1
                    text_source = "full"
                    if item.video_id in text_state_map:
                        text_state_status = text_state_map[item.video_id][0]
                else:
                    if not text_state_available:
                        state.upsert(
                            item.video_id,
                            item.source_hash,
                            status="missing_text_state_unavailable",
                            error=text_state_error,
                            artifact_key="",
                            vector_id=f"fused:{strategy}:{item.video_id}",
                            retry_count=0,
                        )
                        _record_status(
                            status_counters,
                            status_samples,
                            "missing_text_state_unavailable",
                            item.video_id,
                            args.sample_ids_per_status,
                        )
                        continue

                    text_state = text_state_map.get(item.video_id)
                    if not text_state:
                        state.upsert(
                            item.video_id,
                            item.source_hash,
                            status="missing_text_pending",
                            error="text_state_not_found",
                            artifact_key="",
                            vector_id=f"fused:{strategy}:{item.video_id}",
                            retry_count=0,
                        )
                        _record_status(
                            status_counters,
                            status_samples,
                            "missing_text_pending",
                            item.video_id,
                            args.sample_ids_per_status,
                        )
                        continue

                    text_state_status, text_state_hash = text_state
                    hash_ok = not text_state_hash or text_state_hash == item.source_hash
                    if text_state_status not in terminal_text_statuses or not hash_ok:
                        reason = (
                            f"text_status={text_state_status or 'unknown'}"
                            if hash_ok
                            else f"text_hash_mismatch:{text_state_hash}"
                        )
                        state.upsert(
                            item.video_id,
                            item.source_hash,
                            status="missing_text_pending",
                            error=reason,
                            artifact_key="",
                            vector_id=f"fused:{strategy}:{item.video_id}",
                            retry_count=0,
                        )
                        _record_status(
                            status_counters,
                            status_samples,
                            "missing_text_pending",
                            item.video_id,
                            args.sample_ids_per_status,
                        )
                        continue

                    text_missing_reason = f"terminal_text_status:{text_state_status}"

                local_v = tmp / f"{item.video_id}.video.npy"
                local_a = tmp / f"{item.video_id}.audio.npy"
                local_t = tmp / f"{item.video_id}.text.npy"

                try:
                    s3.download_file(v_key, local_v)
                    s3.download_file(a_key, local_a)
                    video_vec = _load_npy(local_v)
                    audio_vec = _load_npy(local_a)

                    if text_present:
                        s3.download_file(t_key, local_t)
                        text_vec = _load_npy(local_t)
                    else:
                        text_vec = np.zeros(int(args.text_dim), dtype=np.float32)

                    fused_vec = _fuse_vectors(
                        strategy=strategy,
                        video_vec=video_vec,
                        audio_vec=audio_vec,
                        text_vec=text_vec,
                        append_mask=bool(args.append_text_presence_mask),
                        text_present=text_present,
                    )

                    current_dims = {
                        "fusion_strategy": strategy,
                        "video_dim": int(video_vec.shape[0]),
                        "audio_dim": int(audio_vec.shape[0]),
                        "text_dim": int(text_vec.shape[0]),
                        "fused_dim": int(fused_vec.shape[0]),
                        "mask_appended": bool(args.append_text_presence_mask),
                    }

                    if existing_schema:
                        for key in ("fusion_strategy", "video_dim", "audio_dim", "text_dim", "fused_dim", "mask_appended"):
                            if key in existing_schema and existing_schema[key] != current_dims[key]:
                                raise ValueError(
                                    f"Schema mismatch for {item.video_id}: key={key} existing={existing_schema[key]} current={current_dims[key]}"
                                )

                    if schema_dims:
                        for key in ("video_dim", "audio_dim", "text_dim", "fused_dim", "mask_appended"):
                            if schema_dims[key] != current_dims[key]:
                                raise ValueError(
                                    f"Inconsistent dims in run for {item.video_id}: key={key} expected={schema_dims[key]} current={current_dims[key]}"
                                )
                    else:
                        schema_dims = dict(current_dims)

                    fusion_status = "success_full" if text_present else "success_text_placeholder"
                    fused_candidates.append(
                        FusedRecord(
                            item=item,
                            fused_vec=fused_vec,
                            fusion_status=fusion_status,
                            text_present=text_present,
                            text_source="full" if text_present else "placeholder",
                            text_state_status=text_state_status,
                            text_missing_reason=text_missing_reason,
                            video_key=v_key,
                            audio_key=a_key,
                            text_key=t_key,
                            retry_count_at_write=0,
                        )
                    )
                except Exception as exc:
                    fail_status = state.upsert_with_retry(
                        video_id=item.video_id,
                        source_hash=item.source_hash,
                        max_fail_retries=args.max_fail_retries,
                        error=str(exc),
                        artifact_key="",
                        vector_id=f"fused:{strategy}:{item.video_id}",
                        fail_status="fail",
                        terminal_status="fail_terminal",
                    )
                    _record_status(
                        status_counters,
                        status_samples,
                        fail_status,
                        item.video_id,
                        args.sample_ids_per_status,
                    )
                finally:
                    for p in (local_v, local_a, local_t):
                        if p.exists():
                            p.unlink()

            if fused_candidates:
                start_idx = _next_shard_index(s3, shard_partition_prefix)
                chunks = _chunked(fused_candidates, args.shard_size)

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

                new_manifest_rows = []
                for rec in fused_candidates:
                    new_manifest_rows.append(
                        {
                            "video_id": rec.item.video_id,
                            "captured_at": rec.item.captured_at,
                            "source_hash": rec.item.source_hash,
                            "fusion_strategy": strategy,
                            "fused_dim": int(rec.fused_vec.shape[0]),
                            "fused_key": rec.fused_key,
                            "shard_idx": int(rec.shard_idx),
                            "shard_n": int(rec.shard_n),
                            "video_emb_key": rec.video_key,
                            "audio_emb_key": rec.audio_key,
                            "text_emb_key": rec.text_key,
                            "text_present": int(rec.text_present),
                            "text_source": rec.text_source,
                            "text_state_status": rec.text_state_status,
                            "text_missing_reason": rec.text_missing_reason,
                            "fusion_status": rec.fusion_status,
                            "retry_count_at_write": int(rec.retry_count_at_write),
                        }
                    )
                new_df = pd.DataFrame(new_manifest_rows)

                old_df = pd.DataFrame()
                local_manifest = tmp / "fused_manifest.parquet"
                if s3.exists(manifest_key):
                    s3.download_file(manifest_key, local_manifest)
                    old_df = pd.read_parquet(local_manifest)

                old_map: dict[tuple[str, str], tuple[str, int, str]] = {}
                if not old_df.empty and {"video_id", "source_hash", "fused_key", "shard_idx", "fusion_status"}.issubset(old_df.columns):
                    for row in old_df.itertuples(index=False):
                        old_map[(str(row.video_id), str(row.source_hash))] = (
                            str(row.fused_key),
                            int(row.shard_idx),
                            str(row.fusion_status),
                        )

                new_map = {
                    (str(r.video_id), str(r.source_hash)): (str(r.fused_key), int(r.shard_idx), str(r.fusion_status))
                    for r in new_df.itertuples(index=False)
                }

                old_keys = set(old_map.keys())
                new_keys = set(new_map.keys())
                manifest_rows_added = len(new_keys - old_keys)
                manifest_rows_updated = sum(1 for k in (new_keys & old_keys) if old_map[k] != new_map[k])

                merged_df = _merge_manifest(old_df, new_df)
                merged_df.to_parquet(local_manifest, index=False)
                s3.upload_file(local_manifest, manifest_key)

                schema_payload = {
                    "fusion_strategy": strategy,
                    "video_dim": int(schema_dims["video_dim"]),
                    "audio_dim": int(schema_dims["audio_dim"]),
                    "text_dim": int(schema_dims["text_dim"]),
                    "fused_dim": int(schema_dims["fused_dim"]),
                    "mask_appended": bool(schema_dims["mask_appended"]),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                s3.upload_json(schema_payload, schema_key)

                for rec in fused_candidates:
                    state.upsert(
                        rec.item.video_id,
                        rec.item.source_hash,
                        status=rec.fusion_status,
                        error="",
                        artifact_key=rec.fused_key,
                        vector_id=f"fused:{strategy}:{rec.item.video_id}",
                        retry_count=0,
                    )
                    _record_status(
                        status_counters,
                        status_samples,
                        rec.fusion_status,
                        rec.item.video_id,
                        args.sample_ids_per_status,
                    )

                cache: dict[str, np.lib.npyio.NpzFile] = {}
                for rec in fused_candidates[: min(3, len(fused_candidates))]:
                    check = _load_back_vector(s3, rec.fused_key, rec.shard_idx, tmp, cache)
                    if int(check.shape[0]) != int(rec.fused_vec.shape[0]):
                        raise ValueError(
                            f"Shard pointer validation failed for {rec.item.video_id}: {check.shape[0]} != {rec.fused_vec.shape[0]}"
                        )

            print("Fusion summary")
            print(f"fusion_strategy: {strategy}")
            print(f"metadata_rows_deduped: {len(items)}")
            print(f"delta_items: {len(delta)}")
            print(f"fused_candidates: {len(fused_candidates) if 'fused_candidates' in locals() else 0}")
            print(f"shard_files_written: {shard_files_written}")
            print(f"manifest_rows_added: {manifest_rows_added}")
            print(f"manifest_rows_updated: {manifest_rows_updated}")
            if schema_dims:
                print(f"fused_dim: {schema_dims['fused_dim']}")

            ordered_statuses = [
                "success_full",
                "success_text_placeholder",
                "missing_required_modality",
                "missing_text_pending",
                "missing_text_state_unavailable",
                "fail",
                "fail_terminal",
            ]
            for status in ordered_statuses:
                print(f"{status}: {status_counters.get(status, 0)}")
                samples = status_samples.get(status, [])
                if samples:
                    print(f"{status}_sample_ids: {','.join(samples)}")

            print(f"manifest_s3_key: {manifest_key}")
            print(f"schema_s3_key: {schema_key}")
            print(f"shards_prefix: {shard_partition_prefix}/")
            print(f"state_db: {state_path}")
            print(f"state_s3_key: {state_s3_key}")
    finally:
        state.close()
        s3.persist_state(state_path, state_s3_key)


if __name__ == "__main__":
    main()
