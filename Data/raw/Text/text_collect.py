"""Fetch captions/transcript metadata for daily-delta items from horizon metadata."""

import argparse
import json
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    ScriptStateDB,
    compute_delta,
    load_latest_horizon_rows,
)

DEFAULT_TEXT_DIR = REPO_ROOT / "Data" / "raw" / "Text" / "raw_data"
DEFAULT_STATE_DB = REPO_ROOT / "state" / "text_downloader.sqlite"
PREFERRED_LANGS = ["en", "en-US", "en-GB"]
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[\.,]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[\.,]\d{3}$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_ydl_common_opts(args: argparse.Namespace) -> Dict:
    opts: Dict = {
        "quiet": True,
        "skip_download": True,
    }
    if args.cookies_file:
        opts["cookiefile"] = args.cookies_file
    if args.cookies_from_browser:
        # yt-dlp expects tuple form, e.g. ("chrome",)
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    return opts


def classify_error(error_text: str) -> str:
    message = ANSI_RE.sub("", error_text or "").lower()
    if "http error 429" in message or "too many requests" in message or "rate limit" in message:
        return "fail_rate_limited"
    if "failed to resolve" in message or "name or service not known" in message or "nodename nor servname" in message:
        return "fail_network"
    if "timed out" in message or "connection reset" in message:
        return "fail_network"
    if "sign in to confirm your age" in message or "use --cookies-from-browser" in message:
        return "fail_auth"
    if "private video" in message or "video unavailable" in message:
        return "fail_unavailable"
    if "subtitles are not available" in message or "no subtitles" in message:
        return "no_captions"
    return "fail"


def choose_caption_language(info: Dict) -> Tuple[Optional[str], bool]:
    subtitles = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    for lang in PREFERRED_LANGS:
        if subtitles.get(lang):
            return lang, False
    if subtitles:
        return sorted(subtitles.keys())[0], False

    for lang in PREFERRED_LANGS:
        if auto.get(lang):
            return lang, True
    if auto:
        return sorted(auto.keys())[0], True

    return None, False


def parse_vtt_to_text(vtt_path: Path) -> str:
    lines = []
    with vtt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("WEBVTT"):
                continue
            if TIMESTAMP_RE.match(line):
                continue
            if line.startswith("NOTE"):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            lines.append(line)

    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
        prev = line

    return "\n".join(deduped).strip()


def download_caption_text(url: str, video_id: str, ydl_common_opts: Dict) -> Tuple[str, Dict]:
    with yt_dlp.YoutubeDL(dict(ydl_common_opts)) as probe:
        info = probe.extract_info(url, download=False)

    lang, is_auto = choose_caption_language(info)
    if not lang:
        return "", {"status": "no_captions", "lang": None, "is_auto": False}

    with tempfile.TemporaryDirectory(prefix=f"captions_{video_id}_") as tmp_dir:
        tmpl = str(Path(tmp_dir) / f"{video_id}.%(ext)s")
        opts = dict(ydl_common_opts)
        opts.update({
            "outtmpl": tmpl,
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        candidates = list(Path(tmp_dir).glob(f"{video_id}*.vtt"))
        if not candidates:
            return "", {"status": "no_captions", "lang": lang, "is_auto": is_auto}

        text = parse_vtt_to_text(candidates[0])
        if not text:
            return "", {"status": "no_captions", "lang": lang, "is_auto": is_auto}

        return text, {"status": "success", "lang": lang, "is_auto": is_auto}


def write_text_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect daily-delta transcript/caption outputs")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--out_dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    parser.add_argument("--cookies_file", default="", help="Path to Netscape cookies.txt for yt-dlp")
    parser.add_argument(
        "--cookies_from_browser",
        default="",
        help="Browser name for yt-dlp cookie import (e.g. chrome, firefox, edge, safari)",
    )
    parser.add_argument("--retry", type=int, default=4, help="Retries per item for transient errors")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Base delay in seconds for backoff")
    parser.add_argument("--sleep_between_items", type=float, default=0.25, help="Base pause between items")
    parser.add_argument("--jitter", type=float, default=0.35, help="Random extra pause (0..jitter seconds)")
    args = parser.parse_args()
    ydl_common_opts = build_ydl_common_opts(args)

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
        no_captions = 0
        failed = 0
        skipped_existing = 0

        for item in delta:
            out_path = out_dir / f"{item.video_id}.json"
            existing = state.get(item.video_id)
            if out_path.exists() and existing and existing[3] in {"success", "no_captions"}:
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), existing[3], "skipped_existing")
                skipped_existing += 1
                continue

            if not item.video_url:
                payload = {
                    "video_id": item.video_id,
                    "video_url": item.video_url,
                    "captured_at": item.captured_at,
                    "status": "missing",
                    "transcript": "",
                    "error": "missing_url",
                }
                write_text_json(out_path, payload)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "missing", "missing_url")
                failed += 1
                continue

            try:
                text = ""
                meta: Dict = {}
                status = "fail"
                last_error = ""

                for attempt in range(1, args.retry + 2):
                    try:
                        text, meta = download_caption_text(item.video_url, item.video_id, ydl_common_opts)
                        status = meta["status"]
                        last_error = ""
                        break
                    except Exception as exc:
                        last_error = ANSI_RE.sub("", str(exc)).strip()
                        status = classify_error(last_error)
                        if status == "fail_rate_limited" and attempt <= args.retry:
                            delay = (args.retry_delay * (2 ** (attempt - 1))) + random.uniform(0.0, args.jitter)
                            time.sleep(delay)
                            continue
                        if status == "fail_network" and attempt <= args.retry:
                            delay = (args.retry_delay * attempt) + random.uniform(0.0, args.jitter)
                            time.sleep(delay)
                            continue
                        break

                payload = {
                    "video_id": item.video_id,
                    "video_url": item.video_url,
                    "captured_at": item.captured_at,
                    "status": status,
                    "caption_lang": meta.get("lang") if meta else None,
                    "caption_is_auto": bool(meta.get("is_auto", False)) if meta else False,
                    "transcript": text,
                    "error": last_error,
                }
                write_text_json(out_path, payload)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), status, last_error)
                if status == "success":
                    success += 1
                elif status == "no_captions":
                    no_captions += 1
                else:
                    failed += 1
            except Exception as exc:
                status = classify_error(str(exc))
                payload = {
                    "video_id": item.video_id,
                    "video_url": item.video_url,
                    "captured_at": item.captured_at,
                    "status": status,
                    "transcript": "",
                    "error": str(exc),
                }
                write_text_json(out_path, payload)
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), status, str(exc))
                if status == "no_captions":
                    no_captions += 1
                else:
                    failed += 1
            finally:
                time.sleep(args.sleep_between_items + random.uniform(0.0, args.jitter))

        print("Text downloader summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"no_captions: {no_captions}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"failed: {failed}")
        print(f"output_dir: {out_dir}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
