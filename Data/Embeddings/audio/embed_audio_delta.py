"""Delta audio embedding stage: S3 raw wav -> Wav2Vec2 embedding -> S3."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Data.common.stage_state import StageStateDB, compute_stage_delta  # noqa: E402

DEFAULT_STATE_DB = REPO_ROOT / "state" / "audio_embedding.sqlite"
DEFAULT_STATE_S3_KEY = "clipfarm/state/audio_embedding.sqlite"
DEFAULT_RAW_PREFIX = "clipfarm/raw"
DEFAULT_EMB_PREFIX = "clipfarm/embeddings"
DEFAULT_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 90.0


class AudioEmbedder:
    def __init__(self, model_name: str) -> None:
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(
        self,
        wav_path: Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
    ) -> np.ndarray:
        # Guard against pathological/corrupt files causing huge allocations.
        duration = None
        if max_audio_seconds and max_audio_seconds > 0:
            duration = float(max_audio_seconds)
        audio, _ = librosa.load(str(wav_path), sr=sample_rate, duration=duration)
        if audio.size == 0:
            raise ValueError("Empty audio after load")
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed audio deltas from S3 and store embeddings on S3")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--raw_prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--emb_prefix", default=DEFAULT_EMB_PREFIX)
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--state_s3_key", default=DEFAULT_STATE_S3_KEY)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max_audio_seconds", type=float, default=DEFAULT_MAX_AUDIO_SECONDS)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    state_path = Path(args.state_db)
    s3.restore_state_if_exists(args.state_s3_key, state_path)
    state = StageStateDB(state_path)
    embedder = AudioEmbedder(args.model_name)

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_stage_delta(items, state, terminal_statuses=("success",), max_items=(args.max_items or None))

        success = 0
        failed = 0
        missing_raw = 0

        with tempfile.TemporaryDirectory(prefix="audio_embed_") as tmp_dir:
            tmp = Path(tmp_dir)
            for item in delta:
                raw_key = f"{args.raw_prefix.strip('/')}/audio/{item.video_id}.wav"
                emb_key = f"{args.emb_prefix.strip('/')}/audio/{item.video_id}.npy"

                if not s3.exists(raw_key):
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="missing_raw_object",
                        error=f"s3_missing:{raw_key}",
                        artifact_key="",
                        vector_id="",
                    )
                    missing_raw += 1
                    continue

                local_wav = tmp / f"{item.video_id}.wav"
                local_npy = tmp / f"{item.video_id}.npy"
                try:
                    s3.download_file(raw_key, local_wav)
                    emb = embedder.embed(
                        local_wav,
                        sample_rate=args.sample_rate,
                        max_audio_seconds=args.max_audio_seconds,
                    )
                    np.save(local_npy, emb)
                    s3.upload_file(local_npy, emb_key)

                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="success",
                        error="",
                        artifact_key=emb_key,
                        vector_id="",
                    )
                    success += 1
                except Exception as exc:
                    state.upsert(
                        item.video_id,
                        item.source_hash,
                        status="fail",
                        error=str(exc),
                        artifact_key="",
                        vector_id="",
                    )
                    failed += 1
                finally:
                    if local_wav.exists():
                        local_wav.unlink()
                    if local_npy.exists():
                        local_npy.unlink()

        print("Audio embedding summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"missing_raw_object: {missing_raw}")
        print(f"failed: {failed}")
        print(f"sample_rate: {args.sample_rate}")
        print(f"max_audio_seconds: {args.max_audio_seconds}")
        print(f"state_db: {args.state_db}")
        print(f"state_s3_key: {args.state_s3_key}")
    finally:
        state.close()
        s3.persist_state(state_path, args.state_s3_key)


if __name__ == "__main__":
    main()
