"""Extract audio for daily-delta videos from Data/raw/Video into Data/raw/Audio."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    ScriptStateDB,
    compute_delta,
    load_latest_horizon_rows,
)

DEFAULT_VIDEO_DIR = REPO_ROOT / "Data" / "raw" / "Video" / "raw_data"
DEFAULT_AUDIO_DIR = REPO_ROOT / "Data" / "raw" / "Audio" / "raw_data"
DEFAULT_STATE_DB = REPO_ROOT / "state" / "audio_downloader.sqlite"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_wav_from_video(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio for daily-delta items")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    state = ScriptStateDB(Path(args.state_db))

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_delta(items, state, max_items=(args.max_items or None))

        success = 0
        failed = 0
        missing_video = 0
        skipped_existing = 0

        for item in delta:
            wav_path = audio_dir / f"{item.video_id}.wav"
            existing = state.get(item.video_id)
            if wav_path.exists() and existing and existing[3] == "success":
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "skipped_existing")
                skipped_existing += 1
                continue

            video_path = video_dir / f"{item.video_id}.mp4"
            if not video_path.exists():
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "missing_video", "video_not_found")
                missing_video += 1
                continue

            try:
                extract_wav_from_video(video_path, wav_path)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "")
                success += 1
            except Exception as exc:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "fail", str(exc))
                failed += 1

        print("Audio extractor summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"missing_video: {missing_video}")
        print(f"failed: {failed}")
        print(f"output_dir: {audio_dir}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
