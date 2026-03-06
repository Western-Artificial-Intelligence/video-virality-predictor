"""Build interpretation.csv from clustered video_ids with media-derived attributes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import av
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402

DEFAULT_CLUSTER_CSV = REPO_ROOT / "Unsup_Cluster" / "cluster_results.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "Interpretation" / "interpretation.csv"
DEFAULT_VIDEO_DIR = REPO_ROOT / "Data" / "raw" / "Video" / "raw_data"
DEFAULT_AUDIO_DIR = REPO_ROOT / "Data" / "raw" / "Audio" / "raw_data"


def build_video_url_map(metadata_csv: Path) -> dict[str, str]:
    items = load_latest_horizon_rows(csv_path=metadata_csv)
    return {item.video_id: item.video_url for item in items if item.video_id}


def _download_if_missing(
    *,
    video_id: str,
    local_path: Path,
    s3: Optional[S3ArtifactStore],
    s3_key: str,
) -> bool:
    if local_path.exists():
        return True
    if s3 is None:
        return False
    try:
        s3.download_file(s3_key, local_path)
        return local_path.exists()
    except Exception:
        return False


def pacing_metrics(video_path: Path, fps_sample: int = 2, diff_thresh: float = 25.0) -> dict[str, float]:
    if not video_path.exists():
        return {"motion_mean": 0.0, "cut_rate_per_min": 0.0}

    try:
        container = av.open(str(video_path))
    except Exception:
        return {"motion_mean": 0.0, "cut_rate_per_min": 0.0}

    try:
        if not container.streams.video:
            return {"motion_mean": 0.0, "cut_rate_per_min": 0.0}

        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 30.0
        step = max(1, int(src_fps / max(1, fps_sample)))

        prev = None
        diffs: list[float] = []
        cuts = 0
        frame_idx = 0
        total_frames = 0

        for frame in container.decode(video=0):
            total_frames += 1
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            gray = frame.to_ndarray(format="gray")
            if prev is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32))))
                diffs.append(diff)
                if diff > float(diff_thresh):
                    cuts += 1
            prev = gray
            frame_idx += 1

        duration_s = float(total_frames / src_fps) if src_fps > 0 else 0.0
        return {
            "motion_mean": float(np.mean(diffs)) if diffs else 0.0,
            "cut_rate_per_min": float(cuts / max(duration_s / 60.0, 1e-6)),
        }
    except Exception:
        return {"motion_mean": 0.0, "cut_rate_per_min": 0.0}
    finally:
        try:
            container.close()
        except Exception:
            pass


def audio_energy_metrics(audio_path: Path) -> dict[str, float]:
    if not audio_path.exists():
        return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0}

    try:
        container = av.open(str(audio_path))
    except Exception:
        return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0}

    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0}

        rms_vals: list[float] = []
        for frame in container.decode(audio=0):
            samples = frame.to_ndarray().astype(np.float32)
            flat = samples.reshape(-1)
            if flat.size == 0:
                continue
            rms_vals.append(float(np.sqrt(np.mean(flat**2))))

        if not rms_vals:
            return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0}

        return {
            "audio_rms_mean": float(np.mean(rms_vals)),
            "audio_rms_std": float(np.std(rms_vals)),
        }
    except Exception:
        return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0}
    finally:
        try:
            container.close()
        except Exception:
            pass


def visual_density(video_path: Path, fps_sample: int = 1, edge_thresh: float = 20.0) -> float:
    if not video_path.exists():
        return 0.0

    try:
        container = av.open(str(video_path))
    except Exception:
        return 0.0

    try:
        if not container.streams.video:
            return 0.0

        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 30.0
        step = max(1, int(src_fps / max(1, fps_sample)))

        densities: list[float] = []
        frame_idx = 0

        for frame in container.decode(video=0):
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            gray = frame.to_ndarray(format="gray").astype(np.float32)
            gy, gx = np.gradient(gray)
            mag = np.sqrt(gx**2 + gy**2)
            densities.append(float((mag > edge_thresh).mean()))
            frame_idx += 1

        return float(np.mean(densities)) if densities else 0.0
    except Exception:
        return 0.0
    finally:
        try:
            container.close()
        except Exception:
            pass


def _cluster_sort_key(row: pd.Series) -> tuple[int, str]:
    raw = str(row.get("cluster", "")).strip()
    if raw.isdigit():
        return int(raw), str(row.get("video_id", ""))
    return 10**9, str(row.get("video_id", ""))


def build_interpretation_rows(
    *,
    clusters: pd.DataFrame,
    video_id_to_url: dict[str, str],
    video_dir: Path,
    audio_dir: Path,
    s3: Optional[S3ArtifactStore],
    raw_prefix: str,
    fetch_missing: bool,
    fps_sample: int,
    diff_thresh: float,
    edge_thresh: float,
    checkpoint_csv: Optional[Path] = None,
    checkpoint_every: int = 25,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    checkpoint_buffer: list[dict] = []
    checkpoint_every = max(1, int(checkpoint_every))
    checkpoint_has_header = bool(
        checkpoint_csv is not None and checkpoint_csv.exists() and checkpoint_csv.stat().st_size > 0
    )

    stats = {
        "video_downloaded": 0,
        "audio_downloaded": 0,
        "video_missing": 0,
        "audio_missing": 0,
        "video_decode_failures": 0,
        "audio_decode_failures": 0,
    }

    total = len(clusters)
    for idx, (_, row) in enumerate(clusters.iterrows(), start=1):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue

        print(f"[interpret] {idx}/{total} {video_id}", flush=True)

        video_path = video_dir / f"{video_id}.mp4"
        audio_path = audio_dir / f"{video_id}.wav"

        video_available_before = video_path.exists()
        audio_available_before = audio_path.exists()

        if fetch_missing and not video_available_before:
            ok = _download_if_missing(
                video_id=video_id,
                local_path=video_path,
                s3=s3,
                s3_key=f"{raw_prefix.strip('/')}/video/{video_id}.mp4",
            )
            if ok:
                stats["video_downloaded"] += 1

        if fetch_missing and not audio_available_before:
            ok = _download_if_missing(
                video_id=video_id,
                local_path=audio_path,
                s3=s3,
                s3_key=f"{raw_prefix.strip('/')}/audio/{video_id}.wav",
            )
            if ok:
                stats["audio_downloaded"] += 1

        video_available = video_path.exists()
        audio_available = audio_path.exists()

        if not video_available:
            stats["video_missing"] += 1
        if not audio_available:
            stats["audio_missing"] += 1

        pace = pacing_metrics(video_path=video_path, fps_sample=fps_sample, diff_thresh=diff_thresh)
        if video_available and pace["motion_mean"] == 0.0 and pace["cut_rate_per_min"] == 0.0:
            stats["video_decode_failures"] += 1

        audio = audio_energy_metrics(audio_path=audio_path)
        if audio_available and audio["audio_rms_mean"] == 0.0 and audio["audio_rms_std"] == 0.0:
            stats["audio_decode_failures"] += 1

        density = visual_density(video_path=video_path, fps_sample=max(1, int(fps_sample)), edge_thresh=edge_thresh)

        out_row = {
            "video_id": video_id,
            "cluster": int(row.get("cluster", 0)) if str(row.get("cluster", "")).isdigit() else row.get("cluster", ""),
            "cluster_id": int(row.get("cluster_id", row.get("cluster", 0)))
            if str(row.get("cluster_id", row.get("cluster", ""))).isdigit()
            else row.get("cluster_id", row.get("cluster", "")),
            "fusion_strategy": str(row.get("fusion_strategy", "")).strip(),
            "k_selected": int(row.get("k_selected", 0)) if str(row.get("k_selected", "")).isdigit() else row.get("k_selected", ""),
            "url": video_id_to_url.get(video_id, ""),
            "video_available": int(video_available),
            "audio_available": int(audio_available),
            "motion_mean": float(pace["motion_mean"]),
            "cut_rate_per_min": float(pace["cut_rate_per_min"]),
            "audio_rms_mean": float(audio["audio_rms_mean"]),
            "audio_rms_std": float(audio["audio_rms_std"]),
            "visual_density": float(density),
        }
        rows.append(out_row)
        checkpoint_buffer.append(out_row)

        if checkpoint_csv is not None and len(checkpoint_buffer) >= checkpoint_every:
            pd.DataFrame(checkpoint_buffer).to_csv(
                checkpoint_csv,
                mode="a",
                index=False,
                quoting=csv.QUOTE_MINIMAL,
                header=not checkpoint_has_header,
            )
            checkpoint_has_header = True
            checkpoint_buffer.clear()

    if checkpoint_csv is not None and checkpoint_buffer:
        pd.DataFrame(checkpoint_buffer).to_csv(
            checkpoint_csv,
            mode="a",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            header=not checkpoint_has_header,
        )

    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interpretation.csv from clustered video IDs")
    parser.add_argument("--cluster_csv", default=str(DEFAULT_CLUSTER_CSV))
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--output_csv", default=str(DEFAULT_OUT_CSV))

    parser.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))

    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--raw_prefix", default="clipfarm/raw")
    parser.add_argument("--fetch_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_every", type=int, default=25)

    parser.add_argument("--fps_sample", type=int, default=2)
    parser.add_argument("--diff_thresh", type=float, default=25.0)
    parser.add_argument("--edge_thresh", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cluster_csv = Path(args.cluster_csv)
    metadata_csv = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)

    if not cluster_csv.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    clusters = pd.read_csv(cluster_csv)
    required = {"video_id", "cluster"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"Missing required columns in {cluster_csv}: {missing}")
    clusters["video_id"] = clusters["video_id"].astype(str)

    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    video_id_to_url = build_video_url_map(metadata_csv)

    existing_df = pd.DataFrame()
    if bool(args.resume) and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv, low_memory=False)
            if "video_id" in existing_df.columns:
                existing_df["video_id"] = existing_df["video_id"].astype(str)
                existing_df = existing_df.drop_duplicates(subset=["video_id"], keep="last")
                done_ids = set(existing_df["video_id"].tolist())
                before_n = int(len(clusters))
                clusters = clusters[~clusters["video_id"].isin(done_ids)].copy()
                print(
                    f"[interpret] resume enabled: skipping {before_n - len(clusters)} already-computed rows",
                    flush=True,
                )
            else:
                existing_df = pd.DataFrame()
        except Exception:
            existing_df = pd.DataFrame()
    elif output_csv.exists():
        output_csv.unlink()

    s3: Optional[S3ArtifactStore] = None
    if bool(args.fetch_missing):
        if args.s3_bucket:
            s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
        else:
            print("[interpret] fetch_missing enabled but no s3_bucket provided; proceeding local-only", flush=True)

    rows, stats = build_interpretation_rows(
        clusters=clusters,
        video_id_to_url=video_id_to_url,
        video_dir=video_dir,
        audio_dir=audio_dir,
        s3=s3,
        raw_prefix=args.raw_prefix,
        fetch_missing=bool(args.fetch_missing),
        fps_sample=int(args.fps_sample),
        diff_thresh=float(args.diff_thresh),
        edge_thresh=float(args.edge_thresh),
        checkpoint_csv=output_csv,
        checkpoint_every=int(args.checkpoint_every),
    )

    if existing_df.empty:
        out_df = pd.DataFrame(rows)
    elif rows:
        out_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
    else:
        out_df = existing_df.copy()

    if not out_df.empty:
        out_df["video_id"] = out_df["video_id"].astype(str)
        out_df = out_df.drop_duplicates(subset=["video_id"], keep="last")
        out_df = out_df.sort_values(by=["cluster", "video_id"]).reset_index(drop=True)

    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("interpretation summary")
    print(f"rows: {len(out_df)}")
    print(f"video_missing: {stats['video_missing']}")
    print(f"audio_missing: {stats['audio_missing']}")
    print(f"video_downloaded: {stats['video_downloaded']}")
    print(f"audio_downloaded: {stats['audio_downloaded']}")
    print(f"video_decode_failures: {stats['video_decode_failures']}")
    print(f"audio_decode_failures: {stats['audio_decode_failures']}")
    print(f"output_csv: {output_csv}")


if __name__ == "__main__":
    main()
