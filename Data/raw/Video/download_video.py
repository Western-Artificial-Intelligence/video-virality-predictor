"""Download only daily-delta videos from horizon metadata into Data/raw/Video."""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.cloud_sync import CloudUploader, maybe_delete_local  # noqa: E402
from Data.common.horizon_delta import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    ScriptStateDB,
    compute_delta,
    load_latest_horizon_rows,
)

DEFAULT_OUT_DIR = REPO_ROOT / "Data" / "raw" / "Video" / "raw_data"
DEFAULT_AUDIO_DIR = REPO_ROOT / "Data" / "raw" / "Audio" / "raw_data"
DEFAULT_STATE_DB = REPO_ROOT / "state" / "video_downloader.sqlite"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def classify_video_error(error_text: str) -> str:
    msg = (error_text or "").lower()
    if "video unavailable. this video has been removed by the uploader" in msg:
        return "fail_removed"
    if "video unavailable. this video is private" in msg:
        return "fail_private"
    if "video unavailable" in msg or "this video is not available" in msg:
        return "fail_unavailable"
    if "only images are available" in msg or "signature solving failed" in msg or "n challenge solving failed" in msg:
        return "fail_challenge_gated"
    if "http error 429" in msg or "too many requests" in msg:
        return "fail_rate_limited"
    if "http error 403" in msg or "forbidden" in msg:
        return "fail_forbidden"
    if "sign in to confirm" in msg or "cookies-from-browser" in msg:
        return "fail_auth"
    if "requested format is not available" in msg:
        return "fail_format_unavailable"
    if "failed to resolve" in msg or "nodename nor servname" in msg or "name or service not known" in msg:
        return "fail_network"
    if "ffmpeg" in msg or "pcm_s16le" in msg:
        return "fail_audio_extract"
    if "cloud_upload_failed" in msg:
        return "fail_cloud_upload"
    return "fail"


def is_terminal_video_failure(existing: tuple, item_hash: str) -> bool:
    # existing tuple schema: (video_id, source_hash, processed_at, status, error)
    existing_hash = existing[1]
    status = existing[3]
    if existing_hash != item_hash:
        return False
    return status in {"missing", "fail_removed", "fail_private", "fail_unavailable"}


def is_cooldown_active(existing: tuple, item_hash: str, cooldown_hours: float) -> bool:
    # existing tuple schema: (video_id, source_hash, processed_at, status, error)
    existing_hash = existing[1]
    processed_at = existing[2]
    status = existing[3]
    if existing_hash != item_hash:
        return False
    if status not in {"fail_challenge_gated", "fail_rate_limited", "fail_auth", "fail_forbidden"}:
        return False
    ts = parse_iso(processed_at)
    if ts is None:
        return False
    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    return age_hours < cooldown_hours


def build_ydl_opts(
    out_mp4: Path,
    cookies_file: str,
    cookies_from_browser: str,
    player_clients: str,
) -> dict:
    out_tmpl = str(out_mp4.with_suffix(".%(ext)s"))
    clients = [c.strip() for c in (player_clients or "").split(",") if c.strip()]
    using_cookies = bool(cookies_file or cookies_from_browser)
    if not clients:
        # Web clients can use authenticated cookies. Android/iOS clients cannot.
        clients = ["web", "web_safari"] if using_cookies else ["android", "ios"]
    elif using_cookies:
        # If caller explicitly passed unsupported cookie clients, fix automatically.
        mobile_only = {"android", "ios"}
        filtered = [c for c in clients if c not in mobile_only]
        clients = filtered if filtered else ["web", "web_safari"]

    ydl_opts: dict = {
        "outtmpl": out_tmpl,
        # Relax format constraints to reduce "Requested format is not available"
        # on Shorts where only non-mp4 streams are exposed.
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        # Avoid SABR-prone web clients where possible.
        "extractor_args": {"youtube": {"player_client": clients}},
        # Be conservative on fragment parallelism to reduce mid-stream 403s.
        "concurrent_fragment_downloads": 1,
        # Prevent long hangs on problematic IDs.
        "socket_timeout": 20,
        "retries": 1,
        "fragment_retries": 1,
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return ydl_opts


def download_video(
    url: str,
    out_mp4: Path,
    cookies_file: str,
    cookies_from_browser: str,
    player_clients: str,
) -> None:
    ydl_opts = build_ydl_opts(out_mp4, cookies_file, cookies_from_browser, player_clients)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if out_mp4.exists():
        return

    for ext in (".mkv", ".webm", ".m4a"):
        alt = out_mp4.with_suffix(ext)
        if alt.exists():
            alt.rename(out_mp4)
            return

    raise RuntimeError(f"Downloader did not produce expected output: {out_mp4}")


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


def download_video_with_fallback(
    url: str,
    out_mp4: Path,
    cookies_file: str,
    cookies_from_browser: str,
    player_clients: str,
) -> None:
    # Attempt 1: caller-configured strategy (typically cookies + web clients).
    first_error = ""
    try:
        download_video(
            url=url,
            out_mp4=out_mp4,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
            player_clients=player_clients,
        )
        return
    except Exception as exc:
        first_error = str(exc)

    # Attempt 2: challenge bypass fallback.
    # If web+cookies path fails with signature/n-challenge or format-unavailable,
    # try no-cookies with mobile/tv clients.
    first_status = classify_video_error(first_error)
    should_try_mobile_fallback = bool(cookies_file or cookies_from_browser) and first_status in {
        "fail_challenge_gated",
        "fail_format_unavailable",
    }
    if not should_try_mobile_fallback:
        raise RuntimeError(first_error)

    try:
        download_video(
            url=url,
            out_mp4=out_mp4,
            cookies_file="",
            cookies_from_browser="",
            player_clients="android,ios,tv",
        )
        return
    except Exception as exc2:
        second_error = str(exc2)
        raise RuntimeError(
            f"{first_error} || fallback_no_cookies_mobile_failed: {second_error}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily-delta videos from horizon metadata")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument(
        "--extract_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also extract wav to audio_dir",
    )
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument(
        "--challenge_cooldown_hours",
        type=float,
        default=24.0,
        help="Skip retrying challenge/rate/auth-gated IDs for this many hours",
    )
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    parser.add_argument("--cookies_file", default="", help="Path to Netscape cookies.txt for yt-dlp")
    parser.add_argument(
        "--cookies_from_browser",
        default="",
        help="Browser name for yt-dlp cookie import (e.g. chrome, firefox, edge, safari)",
    )
    parser.add_argument(
        "--player_clients",
        default="",
        help="Comma-separated yt-dlp YouTube player clients. Default auto: web/web_safari when cookies are used, else android/ios.",
    )
    parser.add_argument(
        "--cloud_root_uri",
        default="",
        help="Cloud base URI for uploads (s3://bucket/prefix or gs://bucket/prefix)",
    )
    parser.add_argument("--cloud_video_prefix", default="video", help="Relative prefix under cloud_root_uri for video files")
    parser.add_argument("--cloud_audio_prefix", default="audio", help="Relative prefix under cloud_root_uri for audio files")
    parser.add_argument(
        "--cloud_delete_local_after_upload",
        action="store_true",
        help="Delete local files after successful cloud upload",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)
    if args.extract_audio:
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
        skipped_existing = 0
        skipped_terminal = 0
        skipped_cooldown = 0
        extracted_audio = 0

        for item in delta:
            out_mp4 = out_dir / f"{item.video_id}.mp4"
            wav_path = audio_dir / f"{item.video_id}.wav"
            existing = state.get(item.video_id)
            outputs_ready = out_mp4.exists() and ((not args.extract_audio) or wav_path.exists())
            if outputs_ready and existing and existing[3] == "success":
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "skipped_existing")
                skipped_existing += 1
                continue
            if existing and is_cooldown_active(existing, item.source_hash, args.challenge_cooldown_hours):
                skipped_cooldown += 1
                continue
            if existing and is_terminal_video_failure(existing, item.source_hash):
                skipped_terminal += 1
                continue

            if not item.video_url:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "missing", "missing_url")
                failed += 1
                continue

            done = False
            error_text = ""
            if not out_mp4.exists():
                for attempt in range(1, args.retry + 2):
                    try:
                        download_video_with_fallback(
                            item.video_url,
                            out_mp4,
                            cookies_file=args.cookies_file,
                            cookies_from_browser=args.cookies_from_browser,
                            player_clients=args.player_clients,
                        )
                        done = True
                        break
                    except Exception as exc:
                        error_text = str(exc)
                        status = classify_video_error(error_text)
                        # Deterministic/gated failures should not be retried immediately.
                        if status in {
                            "fail_challenge_gated",
                            "fail_format_unavailable",
                            "fail_auth",
                            "fail_removed",
                            "fail_private",
                            "fail_unavailable",
                        }:
                            break
                        if attempt <= args.retry:
                            time.sleep(args.retry_delay * attempt)
            else:
                done = True

            if done and args.extract_audio and not wav_path.exists():
                try:
                    extract_wav_from_video(out_mp4, wav_path)
                    extracted_audio += 1
                except Exception as exc:
                    done = False
                    error_text = str(exc)

            if done and uploader.enabled:
                try:
                    uploader.upload_file(out_mp4, f"{args.cloud_video_prefix.strip('/')}/{item.video_id}.mp4")
                    maybe_delete_local(out_mp4, args.cloud_delete_local_after_upload)
                    if args.extract_audio and wav_path.exists():
                        uploader.upload_file(wav_path, f"{args.cloud_audio_prefix.strip('/')}/{item.video_id}.wav")
                        maybe_delete_local(wav_path, args.cloud_delete_local_after_upload)
                except Exception as exc:
                    done = False
                    error_text = f"cloud_upload_failed: {exc}"

            if not done:
                fail_status = classify_video_error(error_text)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), fail_status, error_text)
                failed += 1
            else:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "")
                success += 1

        print("Video downloader summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"extracted_audio: {extracted_audio}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"skipped_terminal: {skipped_terminal}")
        print(f"skipped_cooldown: {skipped_cooldown}")
        print(f"failed: {failed}")
        print(f"output_dir: {out_dir}")
        print(f"audio_output_dir: {audio_dir}")
        if uploader.enabled:
            print(f"cloud_root_uri: {args.cloud_root_uri}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
