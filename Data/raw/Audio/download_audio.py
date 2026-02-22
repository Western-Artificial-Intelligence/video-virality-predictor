"""Extract audio for daily-delta videos from Data/raw/Video into Data/raw/Audio."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.cloud_sync import CloudUploader, maybe_delete_local  # noqa: E402
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
    parser.add_argument(
        "--cloud_root_uri",
        default="",
        help="Cloud base URI for uploads (s3://bucket/prefix or gs://bucket/prefix)",
    )
    parser.add_argument("--cloud_video_prefix", default="video", help="Relative prefix under cloud_root_uri for video files")
    parser.add_argument("--cloud_audio_prefix", default="audio", help="Relative prefix under cloud_root_uri for audio files")
    parser.add_argument(
        "--download_video_from_cloud_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If local mp4 is missing, try to fetch it from cloud_root_uri/cloud_video_prefix",
    )
    parser.add_argument(
        "--cleanup_downloaded_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete temporary mp4 fetched from cloud after audio extraction",
    )
    parser.add_argument(
        "--cloud_delete_local_after_upload",
        action="store_true",
        help="Delete local wav after successful cloud upload",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    state = ScriptStateDB(Path(args.state_db))
    uploader = CloudUploader(args.cloud_root_uri)

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
            downloaded_video = False
            if not video_path.exists():
                if uploader.enabled and args.download_video_from_cloud_if_missing:
                    try:
                        uploader.download_file(
                            f"{args.cloud_video_prefix.strip('/')}/{item.video_id}.mp4",
                            video_path,
                        )
                        downloaded_video = True
                    except Exception:
                        downloaded_video = False

                if not video_path.exists():
                    state.upsert(item.video_id, item.source_hash, utc_now_iso(), "missing_video", "video_not_found")
                    missing_video += 1
                    continue

            try:
                extract_wav_from_video(video_path, wav_path)
                if uploader.enabled:
                    uploader.upload_file(wav_path, f"{args.cloud_audio_prefix.strip('/')}/{item.video_id}.wav")
                    maybe_delete_local(wav_path, args.cloud_delete_local_after_upload)
                if downloaded_video and args.cleanup_downloaded_video:
                    maybe_delete_local(video_path, True)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "")
                success += 1
            except Exception as exc:
                err = str(exc)
                status = "fail_cloud_upload" if "cloud" in err.lower() else "fail"
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), status, err)
                failed += 1

        print("Audio extractor summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"missing_video: {missing_video}")
        print(f"failed: {failed}")
        print(f"output_dir: {audio_dir}")
        if uploader.enabled:
            print(f"cloud_root_uri: {args.cloud_root_uri}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
