"""Download only daily-delta videos from horizon metadata into Data/raw/Video."""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    ScriptStateDB,
    compute_delta,
    load_latest_horizon_rows,
)

DEFAULT_OUT_DIR = REPO_ROOT / "Data" / "raw" / "Video" / "raw_data"
DEFAULT_STATE_DB = REPO_ROOT / "state" / "video_downloader.sqlite"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_ydl_opts(
    out_mp4: Path,
    cookies_file: str,
    cookies_from_browser: str,
    player_clients: str,
) -> dict:
    out_tmpl = str(out_mp4.with_suffix(".%(ext)s"))
    clients = [c.strip() for c in (player_clients or "").split(",") if c.strip()]
    if not clients:
        clients = ["android", "ios"]

    ydl_opts: dict = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        # Avoid SABR-prone web clients where possible.
        "extractor_args": {"youtube": {"player_client": clients}},
        # Be conservative on fragment parallelism to reduce mid-stream 403s.
        "concurrent_fragment_downloads": 1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily-delta videos from horizon metadata")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    parser.add_argument("--cookies_file", default="", help="Path to Netscape cookies.txt for yt-dlp")
    parser.add_argument(
        "--cookies_from_browser",
        default="",
        help="Browser name for yt-dlp cookie import (e.g. chrome, firefox, edge, safari)",
    )
    parser.add_argument(
        "--player_clients",
        default="android,ios",
        help="Comma-separated yt-dlp YouTube player clients (default: android,ios)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = ScriptStateDB(Path(args.state_db))

    try:
        items = load_latest_horizon_rows(
            csv_path=Path(args.metadata_csv),
            include_captured_at_in_hash=args.include_captured_at_in_hash,
        )
        delta = compute_delta(items, state, max_items=(args.max_items or None))

        success = 0
        failed = 0
        skipped_existing = 0

        for item in delta:
            out_mp4 = out_dir / f"{item.video_id}.mp4"
            existing = state.get(item.video_id)
            if out_mp4.exists() and existing and existing[3] == "success":
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "skipped_existing")
                skipped_existing += 1
                continue

            if not item.video_url:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "missing", "missing_url")
                failed += 1
                continue

            done = False
            error_text = ""
            for attempt in range(1, args.retry + 2):
                try:
                    download_video(
                        item.video_url,
                        out_mp4,
                        cookies_file=args.cookies_file,
                        cookies_from_browser=args.cookies_from_browser,
                        player_clients=args.player_clients,
                    )
                    state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "")
                    success += 1
                    done = True
                    break
                except Exception as exc:
                    error_text = str(exc)
                    if attempt <= args.retry:
                        time.sleep(args.retry_delay * attempt)

            if not done:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "fail", error_text)
                failed += 1

        print("Video downloader summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"failed: {failed}")
        print(f"output_dir: {out_dir}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
