"""Delta text embedding stage: S3 raw transcript json -> MiniLM embedding -> S3 + Pinecone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402
from Data.common.pinecone_client import PineconeVectorClient, build_full_metadata  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Data.common.stage_state import StageStateDB, compute_stage_delta  # noqa: E402

DEFAULT_STATE_DB = REPO_ROOT / "state" / "text_embedding.sqlite"
DEFAULT_STATE_S3_KEY = "clipfarm/state/text_embedding.sqlite"
DEFAULT_RAW_PREFIX = "clipfarm/raw"
DEFAULT_EMB_PREFIX = "clipfarm/embeddings"
DEFAULT_INDEX = "clipfarm-text"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def build_meta_text(row: dict) -> str:
    parts = []
    for col in ("title", "description", "query"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " [SEP] ".join(parts)


def load_transcript_from_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tx = payload.get("transcript")
    if isinstance(tx, str):
        return tx.strip()
    return ""


class TextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, meta_text: str, transcript_text: str) -> np.ndarray:
        meta = self.model.encode(
            [meta_text or ""],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        tx = self.model.encode(
            [transcript_text or ""],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        emb = np.concatenate([meta, tx], axis=0).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed text deltas from S3 and upsert to Pinecone")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--raw_prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--emb_prefix", default=DEFAULT_EMB_PREFIX)
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--state_s3_key", default=DEFAULT_STATE_S3_KEY)
    parser.add_argument("--pinecone_index", default=DEFAULT_INDEX)
    parser.add_argument("--pinecone_api_key", default=os.getenv("PINECONE_API_KEY", ""))
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    state_path = Path(args.state_db)
    s3.restore_state_if_exists(args.state_s3_key, state_path)
    state = StageStateDB(state_path)
    pinecone = PineconeVectorClient(api_key=args.pinecone_api_key, index_name=args.pinecone_index)
    embedder = TextEmbedder(args.model_name)

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_stage_delta(items, state, terminal_statuses=("success",), max_items=(args.max_items or None))

        success = 0
        failed = 0
        missing_raw = 0

        with tempfile.TemporaryDirectory(prefix="text_embed_") as tmp_dir:
            tmp = Path(tmp_dir)
            for item in delta:
                raw_key = f"{args.raw_prefix.strip('/')}/text/{item.video_id}.json"
                emb_key = f"{args.emb_prefix.strip('/')}/text/{item.video_id}.npy"
                vector_id = f"text:{item.video_id}"

                if not s3.exists(raw_key):
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="missing_raw_object",
                        error=f"s3_missing:{raw_key}",
                        artifact_key="",
                        vector_id=vector_id,
                    )
                    missing_raw += 1
                    continue

                local_json = tmp / f"{item.video_id}.json"
                local_npy = tmp / f"{item.video_id}.npy"
                try:
                    s3.download_file(raw_key, local_json)
                    meta_text = build_meta_text(item.row)
                    transcript_text = load_transcript_from_json(local_json)
                    emb = embedder.embed(meta_text, transcript_text)
                    np.save(local_npy, emb)
                    s3.upload_file(local_npy, emb_key)

                    metadata = build_full_metadata(
                        item.row,
                        mandatory={
                            "video_id": item.video_id,
                            "captured_at": item.captured_at,
                            "source_hash": item.source_hash,
                            "s3_key": emb_key,
                            "modality": "text",
                            "horizon_days": item.row.get("horizon_days", ""),
                        },
                    )
                    pinecone.upsert(vector_id=vector_id, values=emb.tolist(), metadata=metadata)

                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="success",
                        error="",
                        artifact_key=emb_key,
                        vector_id=vector_id,
                    )
                    success += 1
                except Exception as exc:
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="fail",
                        error=str(exc),
                        artifact_key="",
                        vector_id=vector_id,
                    )
                    failed += 1
                finally:
                    if local_json.exists():
                        local_json.unlink()
                    if local_npy.exists():
                        local_npy.unlink()

        print("Text embedding summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"missing_raw_object: {missing_raw}")
        print(f"failed: {failed}")
        print(f"state_db: {args.state_db}")
        print(f"state_s3_key: {args.state_s3_key}")
    finally:
        state.close()
        s3.persist_state(state_path, args.state_s3_key)


if __name__ == "__main__":
    main()
