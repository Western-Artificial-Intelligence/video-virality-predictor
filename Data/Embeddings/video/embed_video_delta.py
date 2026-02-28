"""Delta video embedding stage: S3 raw mp4 -> VideoMAE embedding -> S3."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402
from Data.common.stage_state import StageStateDB, compute_stage_delta  # noqa: E402

DEFAULT_STATE_DB = REPO_ROOT / "state" / "video_embedding.sqlite"
DEFAULT_STATE_S3_KEY = "clipfarm/state/video_embedding.sqlite"
DEFAULT_RAW_PREFIX = "clipfarm/raw"
DEFAULT_EMB_PREFIX = "clipfarm/embeddings"
DEFAULT_MODEL = "MCG-NJU/videomae-base"


def sample_frames(video_path: Path, num_frames: int = 16) -> np.ndarray:
    """Sample video frames without loading the entire video into memory."""
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream found")
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)

        sampled: list[np.ndarray] = []
        if total_frames > 0:
            target_idx = np.linspace(
                0,
                max(0, total_frames - 1),
                num=min(num_frames, total_frames),
                dtype=int,
            )
            target_set = set(int(i) for i in target_idx.tolist())
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx in target_set:
                    sampled.append(frame.to_ndarray(format="rgb24"))
                    if len(sampled) >= len(target_set):
                        break
        else:
            # Some sources report unknown total frame count; fall back to first N frames.
            for frame in container.decode(video=0):
                sampled.append(frame.to_ndarray(format="rgb24"))
                if len(sampled) >= num_frames:
                    break

    if not sampled:
        raise ValueError("No decodable video frames")

    # Ensure fixed-size tensor for the model.
    if len(sampled) < num_frames:
        sampled.extend([sampled[-1]] * (num_frames - len(sampled)))
    return np.stack(sampled[:num_frames])


class VideoEmbedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = "cuda"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            self.device = "mps"
        else:
            self.device = "cpu"
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.expected_frames = int(getattr(self.model.config, "num_frames", 16))
        self.expected_size = int(getattr(self.model.config, "image_size", 224))

    def _normalize_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Normalize shape to model expectations and harden against odd frame sizes."""
        if pixel_values.ndim == 4:
            # (T,C,H,W) -> batch dimension
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 5:
            raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

        # Determine layout:
        # - (B,T,C,H,W) => channel dim at index 2
        # - (B,C,T,H,W) => channel dim at index 1
        if pixel_values.shape[2] == 3:  # (B,T,C,H,W)
            b, t, c, h, w = pixel_values.shape
            if t != self.expected_frames:
                idx = torch.linspace(0, max(t - 1, 0), self.expected_frames, dtype=torch.long)
                pixel_values = pixel_values.index_select(dim=1, index=idx)
                b, t, c, h, w = pixel_values.shape
            if h != self.expected_size or w != self.expected_size:
                flat = pixel_values.reshape(b * t, c, h, w)
                flat = F.interpolate(flat, size=(self.expected_size, self.expected_size), mode="bilinear", align_corners=False)
                pixel_values = flat.reshape(b, t, c, self.expected_size, self.expected_size)
            return pixel_values

        if pixel_values.shape[1] == 3:  # (B,C,T,H,W)
            b, c, t, h, w = pixel_values.shape
            if t != self.expected_frames:
                idx = torch.linspace(0, max(t - 1, 0), self.expected_frames, dtype=torch.long)
                pixel_values = pixel_values.index_select(dim=2, index=idx)
                b, c, t, h, w = pixel_values.shape
            if h != self.expected_size or w != self.expected_size:
                flat = pixel_values.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                flat = F.interpolate(flat, size=(self.expected_size, self.expected_size), mode="bilinear", align_corners=False)
                pixel_values = flat.reshape(b, t, c, self.expected_size, self.expected_size).permute(0, 2, 1, 3, 4)
            return pixel_values

        raise ValueError(f"Cannot infer pixel layout from shape: {tuple(pixel_values.shape)}")

    @torch.no_grad()
    def embed(self, video_path: Path, num_frames: int = 16) -> np.ndarray:
        frames = sample_frames(video_path, num_frames=num_frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        if "pixel_values" not in inputs:
            raise ValueError("Video processor did not return pixel_values")
        inputs["pixel_values"] = self._normalize_pixel_values(inputs["pixel_values"])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0][0]
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        return cls_emb.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed video deltas from S3 and store embeddings on S3")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--raw_prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--emb_prefix", default=DEFAULT_EMB_PREFIX)
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--state_s3_key", default=DEFAULT_STATE_S3_KEY)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    state_path = Path(args.state_db)
    s3.restore_state_if_exists(args.state_s3_key, state_path)
    state = StageStateDB(state_path)
    embedder = VideoEmbedder(args.model_name, device=args.device)

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_stage_delta(items, state, terminal_statuses=("success",), max_items=(args.max_items or None))

        success = 0
        failed = 0
        missing_raw = 0

        with tempfile.TemporaryDirectory(prefix="video_embed_") as tmp_dir:
            tmp = Path(tmp_dir)
            total = len(delta)
            for idx, item in enumerate(delta, start=1):
                print(f"[embed-video] {idx}/{total} {item.video_id}: start", flush=True)
                raw_key = f"{args.raw_prefix.strip('/')}/video/{item.video_id}.mp4"
                emb_key = f"{args.emb_prefix.strip('/')}/video/{item.video_id}.npy"

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
                    print(f"[embed-video] {idx}/{total} {item.video_id}: missing_raw_object", flush=True)
                    continue

                local_mp4 = tmp / f"{item.video_id}.mp4"
                local_npy = tmp / f"{item.video_id}.npy"
                try:
                    s3.download_file(raw_key, local_mp4)
                    emb = embedder.embed(local_mp4, num_frames=args.num_frames)
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
                    print(f"[embed-video] {idx}/{total} {item.video_id}: success", flush=True)
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
                    print(f"[embed-video] {idx}/{total} {item.video_id}: fail", flush=True)
                finally:
                    if local_mp4.exists():
                        local_mp4.unlink()
                    if local_npy.exists():
                        local_npy.unlink()

        print("Video embedding summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"missing_raw_object: {missing_raw}")
        print(f"failed: {failed}")
        print(f"device: {embedder.device}")
        print(f"num_frames: {args.num_frames}")
        print(f"state_db: {args.state_db}")
        print(f"state_s3_key: {args.state_s3_key}")
    finally:
        state.close()
        s3.persist_state(state_path, args.state_s3_key)


if __name__ == "__main__":
    main()
