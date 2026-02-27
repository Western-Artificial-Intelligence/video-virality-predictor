"""Generate text for daily-delta items using caption-first, ASR fallback.

Flow per video:
1) Try YouTube caption/subtitle extraction first.
2) If captions are missing or caption extraction fails, fall back to ASR from local audio.
"""

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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

DEFAULT_TEXT_DIR = REPO_ROOT / "Data" / "raw" / "Text" / "raw_data"
DEFAULT_AUDIO_DIR = REPO_ROOT / "Data" / "raw" / "Audio" / "raw_data"
DEFAULT_STATE_DB = REPO_ROOT / "state" / "text_downloader.sqlite"
PREFERRED_LANGS = ["en", "en-US", "en-GB"]
DEFAULT_MAX_WORKERS = max(1, min(os.cpu_count() or 1, 4))
_THREAD_LOCAL = __import__("threading").local()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(error_text: str) -> str:
    msg = (error_text or "").lower()
    if "missing_audio" in msg or "audio_not_found" in msg:
        return "missing_audio"
    if "subtitles are not available" in msg or "no subtitles" in msg:
        return "no_captions"
    if "requested format is not available" in msg:
        return "fail_format_unavailable"
    if "signature solving failed" in msg or "n challenge solving failed" in msg:
        return "fail_challenge_gated"
    if "openai_api_key" in msg or "api key" in msg:
        return "fail_auth"
    if "rate limit" in msg or "429" in msg:
        return "fail_rate_limited"
    if "module not found" in msg or "no module named" in msg:
        return "fail_asr_backend_missing"
    if "cloud_upload_failed" in msg:
        return "fail_cloud_upload"
    return "fail"


def build_caption_ydl_opts(args: argparse.Namespace, player_clients: str) -> Dict:
    clients = [c.strip() for c in (player_clients or "").split(",") if c.strip()]
    using_cookies = bool(args.cookies_file or args.cookies_from_browser)
    if not clients:
        clients = ["web", "web_safari"] if using_cookies else ["android", "ios", "tv"]
    elif using_cookies:
        mobile_only = {"android", "ios"}
        filtered = [c for c in clients if c not in mobile_only]
        clients = filtered if filtered else ["web", "web_safari"]

    opts: Dict = {
        "quiet": True,
        "skip_download": True,
        "format": "b/best",
        "extractor_args": {"youtube": {"player_client": clients}},
    }
    if args.cookies_file:
        opts["cookiefile"] = args.cookies_file
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    return opts


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
    for raw in vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line:
            continue
        if line.startswith("NOTE"):
            continue
        lines.append(line)
    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
        prev = line
    return "\n".join(deduped).strip()


def try_caption_first(url: str, video_id: str, opts: Dict) -> Tuple[Optional[str], Dict]:
    with yt_dlp.YoutubeDL(dict(opts)) as probe:
        info = probe.extract_info(url, download=False)
    lang, is_auto = choose_caption_language(info)
    if not lang:
        return None, {"caption_status": "no_captions", "caption_lang": None, "caption_is_auto": False}

    with tempfile.TemporaryDirectory(prefix=f"captions_{video_id}_") as tmp_dir:
        outtmpl = str(Path(tmp_dir) / f"{video_id}.%(ext)s")
        c_opts = dict(opts)
        c_opts.update(
            {
                "outtmpl": outtmpl,
                "writesubtitles": not is_auto,
                "writeautomaticsub": is_auto,
                "subtitleslangs": [lang],
                "subtitlesformat": "vtt",
            }
        )
        with yt_dlp.YoutubeDL(c_opts) as ydl:
            ydl.download([url])

        candidates = list(Path(tmp_dir).glob(f"{video_id}*.vtt"))
        if not candidates:
            return None, {"caption_status": "no_captions", "caption_lang": lang, "caption_is_auto": is_auto}
        text = parse_vtt_to_text(candidates[0])
        if not text:
            return None, {"caption_status": "no_captions", "caption_lang": lang, "caption_is_auto": is_auto}
        return text, {"caption_status": "success", "caption_lang": lang, "caption_is_auto": is_auto}


def transcribe_faster_whisper(audio_path: Path, model_name: str) -> Tuple[str, Dict]:
    from faster_whisper import WhisperModel  # type: ignore

    fw_models = getattr(_THREAD_LOCAL, "fw_models", None)
    if fw_models is None:
        fw_models = {}
        _THREAD_LOCAL.fw_models = fw_models

    model = fw_models.get(model_name)
    if model is None:
        model = WhisperModel(model_name, device="auto", compute_type="int8")
        fw_models[model_name] = model
    segments, info = model.transcribe(str(audio_path), vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    return text, {
        "source": "faster_whisper",
        "model": model_name,
        "language": getattr(info, "language", None),
    }


def transcribe_openai_whisper(audio_path: Path, model_name: str) -> Tuple[str, Dict]:
    import whisper  # type: ignore

    whisper_models = getattr(_THREAD_LOCAL, "whisper_models", None)
    if whisper_models is None:
        whisper_models = {}
        _THREAD_LOCAL.whisper_models = whisper_models

    model = whisper_models.get(model_name)
    if model is None:
        model = whisper.load_model(model_name)
        whisper_models[model_name] = model
    result = model.transcribe(str(audio_path))
    text = (result.get("text") or "").strip()
    return text, {
        "source": "openai_whisper_local",
        "model": model_name,
        "language": result.get("language"),
    }


def transcribe_openai_api(audio_path: Path, model_name: str) -> Tuple[str, Dict]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        try:
            from credentials import OPENAI_API_KEY  # type: ignore

            api_key = (OPENAI_API_KEY or "").strip()
        except Exception:
            api_key = ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for asr_backend=openai_api")

    from openai import OpenAI  # type: ignore

    api_model = (model_name or "").strip()
    # Local Whisper model aliases are not valid for OpenAI API.
    # Force to whisper-1 unless caller passed a known OpenAI model name.
    local_aliases = {
        "",
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
    }
    if api_model.lower() in local_aliases:
        api_model = "whisper-1"

    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        client = OpenAI(api_key=api_key)
        _THREAD_LOCAL.openai_client = client
    with audio_path.open("rb") as f:
        tx = client.audio.transcriptions.create(model=api_model, file=f)
    text = (getattr(tx, "text", None) or "").strip()
    return text, {
        "source": "openai_api",
        "model": api_model,
        "language": None,
    }


def transcribe_audio(audio_path: Path, backend: str, model_name: str) -> Tuple[str, Dict]:
    backend = (backend or "auto").lower()

    if backend == "faster_whisper":
        return transcribe_faster_whisper(audio_path, model_name)
    if backend == "whisper":
        return transcribe_openai_whisper(audio_path, model_name)
    if backend == "openai_api":
        return transcribe_openai_api(audio_path, model_name)

    # auto mode: faster-whisper -> whisper -> openai_api
    errors = []
    for candidate in ("faster_whisper", "whisper", "openai_api"):
        try:
            if candidate == "faster_whisper":
                return transcribe_faster_whisper(audio_path, model_name)
            if candidate == "whisper":
                return transcribe_openai_whisper(audio_path, model_name)
            return transcribe_openai_api(audio_path, "whisper-1")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("all_asr_backends_failed: " + " | ".join(errors))


def write_text_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def process_text_item(
    item,
    args: argparse.Namespace,
    caption_opts: Dict,
    out_dir: Path,
    audio_dir: Path,
) -> Dict[str, str]:
    out_path = out_dir / f"{item.video_id}.json"
    uploader = CloudUploader(args.cloud_root_uri)

    try:
        text = ""
        meta: Dict = {
            "source": None,
            "model": None,
            "language": None,
            "caption_status": None,
            "caption_lang": None,
            "caption_is_auto": False,
        }
        caption_error = ""
        caption_available = as_bool(item.row.get("caption_available"))

        if args.caption_first and caption_available:
            try:
                cap_text, cap_meta = try_caption_first(item.video_url, item.video_id, caption_opts)
                meta.update(cap_meta)
                if cap_text:
                    text = cap_text
                    meta["source"] = "youtube_caption"
            except Exception as cap_exc:
                caption_error = str(cap_exc)
                meta["caption_status"] = "error"
        elif args.caption_first and not caption_available:
            meta["caption_status"] = "metadata_no_captions"

        if not text:
            wav_path = audio_dir / f"{item.video_id}.wav"
            downloaded_audio = False
            if not wav_path.exists() and uploader.enabled and args.download_audio_from_cloud_if_missing:
                try:
                    uploader.download_file(
                        f"{args.cloud_audio_prefix.strip('/')}/{item.video_id}.wav",
                        wav_path,
                    )
                    downloaded_audio = True
                except Exception:
                    downloaded_audio = False

            if not wav_path.exists():
                payload = {
                    "video_id": item.video_id,
                    "video_url": item.video_url,
                    "captured_at": item.captured_at,
                    "status": "missing_audio",
                    "caption_status": meta.get("caption_status"),
                    "caption_lang": meta.get("caption_lang"),
                    "caption_is_auto": bool(meta.get("caption_is_auto", False)),
                    "transcript": "",
                    "error": "audio_not_found",
                }
                write_text_json(out_path, payload)
                return {
                    "video_id": item.video_id,
                    "source_hash": item.source_hash,
                    "status": "missing_audio",
                    "error": "audio_not_found",
                }

            asr_text, asr_meta = transcribe_audio(
                wav_path,
                backend=args.asr_backend,
                model_name=args.asr_model,
            )
            text = asr_text
            meta["source"] = asr_meta.get("source")
            meta["model"] = asr_meta.get("model")
            meta["language"] = asr_meta.get("language")
            if downloaded_audio and args.cleanup_downloaded_audio:
                maybe_delete_local(wav_path, True)

        status = "success" if text else "fail_empty_transcript"
        payload = {
            "video_id": item.video_id,
            "video_url": item.video_url,
            "captured_at": item.captured_at,
            "status": status,
            "transcript_source": meta.get("source"),
            "transcript_model": meta.get("model"),
            "transcript_language": meta.get("language"),
            "caption_status": meta.get("caption_status"),
            "caption_lang": meta.get("caption_lang"),
            "caption_is_auto": bool(meta.get("caption_is_auto", False)),
            "transcript": text,
            "error": caption_error if status == "success" and caption_error else "",
        }
        write_text_json(out_path, payload)

        if status == "success" and uploader.enabled:
            try:
                uploader.upload_file(out_path, f"{args.cloud_text_prefix.strip('/')}/{item.video_id}.json")
                maybe_delete_local(out_path, args.cloud_delete_local_after_upload)
            except Exception as exc:
                raise RuntimeError(f"cloud_upload_failed: {exc}")

        return {
            "video_id": item.video_id,
            "source_hash": item.source_hash,
            "status": status,
            "error": payload["error"],
        }
    except Exception as exc:
        err = str(exc)
        status = classify_error(err)
        payload = {
            "video_id": item.video_id,
            "video_url": item.video_url,
            "captured_at": item.captured_at,
            "status": status,
            "transcript": "",
            "error": err,
        }
        write_text_json(out_path, payload)
        return {
            "video_id": item.video_id,
            "source_hash": item.source_hash,
            "status": status,
            "error": err,
        }


def apply_empty_transcript_retry_policy(state: ScriptStateDB, result: Dict[str, str]) -> Dict[str, str]:
    """Retry fail_empty_transcript once, then mark terminal on second consecutive miss."""
    if result.get("status") != "fail_empty_transcript":
        return result

    existing = state.get(result["video_id"])
    if not existing:
        return result

    _, existing_hash, _, existing_status, _ = existing
    if existing_hash == result["source_hash"] and existing_status == "fail_empty_transcript":
        patched = dict(result)
        patched["status"] = "fail_empty_transcript_terminal"
        if patched.get("error"):
            patched["error"] = f"{patched['error']} | terminal_after_retry"
        else:
            patched["error"] = "terminal_after_retry"
        return patched

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ASR transcripts for daily-delta outputs")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--state_db", default=str(DEFAULT_STATE_DB))
    parser.add_argument("--out_dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--include_captured_at_in_hash", action="store_true")
    parser.add_argument(
        "--caption_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try YouTube captions before ASR",
    )
    parser.add_argument("--cookies_file", default="", help="Path to Netscape cookies.txt for caption extraction")
    parser.add_argument(
        "--cookies_from_browser",
        default="",
        help="Browser name for yt-dlp cookie import (e.g. chrome, firefox, edge, safari)",
    )
    parser.add_argument(
        "--player_clients",
        default="",
        help="Comma-separated yt-dlp player clients for caption extraction",
    )
    parser.add_argument(
        "--asr_backend",
        default="auto",
        choices=["auto", "faster_whisper", "whisper", "openai_api"],
        help="ASR backend selection",
    )
    parser.add_argument(
        "--asr_model",
        default="small",
        help="Model name for local backends (e.g., tiny/base/small/medium/large-v3). openai_api always uses whisper-1.",
    )
    parser.add_argument(
        "--cloud_root_uri",
        default="",
        help="Cloud base URI for uploads (s3://bucket/prefix or gs://bucket/prefix)",
    )
    parser.add_argument("--cloud_audio_prefix", default="audio", help="Relative prefix under cloud_root_uri for audio files")
    parser.add_argument("--cloud_text_prefix", default="text", help="Relative prefix under cloud_root_uri for text files")
    parser.add_argument(
        "--download_audio_from_cloud_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If local wav is missing, try to fetch it from cloud_root_uri/cloud_audio_prefix",
    )
    parser.add_argument(
        "--cleanup_downloaded_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete temporary wav fetched from cloud after ASR",
    )
    parser.add_argument(
        "--cloud_delete_local_after_upload",
        action="store_true",
        help="Delete local json after successful cloud upload",
    )
    args = parser.parse_args()
    caption_opts = build_caption_ydl_opts(args, args.player_clients)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)

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
        missing_audio = 0

        total = len(delta)
        pending = []
        for idx, item in enumerate(delta, start=1):
            out_path = out_dir / f"{item.video_id}.json"
            existing = state.get(item.video_id)
            if out_path.exists() and existing and existing[3] == "success":
                state.upsert(item.video_id, item.source_hash, utc_now_iso(), "success", "skipped_existing")
                skipped_existing += 1
                print(f"[text] {idx}/{total} {item.video_id}: skipped_existing", flush=True)
                continue
            pending.append((idx, item))

        workers = max(1, int(args.max_workers))
        if workers == 1:
            for idx, item in pending:
                print(f"[text] {idx}/{total} {item.video_id}: start", flush=True)
                result = process_text_item(item, args, caption_opts, out_dir, audio_dir)
                result = apply_empty_transcript_retry_policy(state, result)
                state.upsert(
                    result["video_id"],
                    result["source_hash"],
                    utc_now_iso(),
                    result["status"],
                    result["error"],
                )
                if result["status"] == "success":
                    success += 1
                elif result["status"] == "missing_audio":
                    missing_audio += 1
                else:
                    failed += 1
                print(f"[text] {idx}/{total} {item.video_id}: {result['status']}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for idx, item in pending:
                    print(f"[text] {idx}/{total} {item.video_id}: start", flush=True)
                    fut = executor.submit(process_text_item, item, args, caption_opts, out_dir, audio_dir)
                    futures[fut] = (idx, item)

                for fut in as_completed(futures):
                    idx, item = futures[fut]
                    result = fut.result()
                    result = apply_empty_transcript_retry_policy(state, result)
                    state.upsert(
                        result["video_id"],
                        result["source_hash"],
                        utc_now_iso(),
                        result["status"],
                        result["error"],
                    )
                    if result["status"] == "success":
                        success += 1
                    elif result["status"] == "missing_audio":
                        missing_audio += 1
                    else:
                        failed += 1
                    print(f"[text] {idx}/{total} {item.video_id}: {result['status']}", flush=True)

        print("Text downloader summary")
        print(f"metadata_rows_deduped: {len(items)}")
        print(f"delta_items: {len(delta)}")
        print(f"success: {success}")
        print(f"skipped_existing: {skipped_existing}")
        print(f"missing_audio: {missing_audio}")
        print(f"failed: {failed}")
        print(f"output_dir: {out_dir}")
        if uploader.enabled:
            print(f"cloud_root_uri: {args.cloud_root_uri}")
        print(f"state_db: {args.state_db}")
    finally:
        state.close()


if __name__ == "__main__":
    main()
