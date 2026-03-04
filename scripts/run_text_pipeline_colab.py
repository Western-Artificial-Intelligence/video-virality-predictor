"""Colab-oriented text raw pipeline runner with S3 state restore/persist and GPU defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text raw downloader on Colab with S3 state management")
    parser.add_argument("--metadata_csv", default="Data/raw/Metadata/shorts_metadata_horizon.csv")
    parser.add_argument("--state_db", default="state/text_downloader.sqlite")
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--raw_prefix", default="clipfarm/raw")
    parser.add_argument("--state_prefix", default="clipfarm/state")
    parser.add_argument("--audio_prefix", default="audio")
    parser.add_argument("--text_prefix", default="text")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--asr_backend", default="faster_whisper")
    parser.add_argument("--asr_model", default="small")
    parser.add_argument("--caption_first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--download_audio_from_cloud_if_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup_downloaded_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cloud_delete_local_after_upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_seconds", type=int, default=120)
    parser.add_argument("--install_deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_colab_secrets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faster_whisper_device", default=os.getenv("FASTER_WHISPER_DEVICE", "cuda"))
    parser.add_argument("--faster_whisper_compute_type", default=os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16"))
    parser.add_argument("--faster_whisper_batch_size", type=int, default=0, help="0 = auto-tune by detected GPU memory")
    parser.add_argument("--faster_whisper_batched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faster_whisper_beam_size", type=int, default=1)
    parser.add_argument("--faster_whisper_best_of", type=int, default=1)
    parser.add_argument("--faster_whisper_vad_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faster_whisper_cpu_threads", type=int, default=0)
    parser.add_argument("--faster_whisper_language", default="")
    parser.add_argument("--aws_access_key_id", default=os.getenv("AWS_ACCESS_KEY_ID", ""))
    parser.add_argument("--aws_secret_access_key", default=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    parser.add_argument("--aws_session_token", default=os.getenv("AWS_SESSION_TOKEN", ""))
    parser.add_argument("--aws_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--secret_s3_bucket", default="S3_BUCKET")
    parser.add_argument("--secret_aws_access_key_id", default="AWS_ACCESS_KEY_ID")
    parser.add_argument("--secret_aws_secret_access_key", default="AWS_SECRET_ACCESS_KEY")
    parser.add_argument("--secret_aws_session_token", default="AWS_SESSION_TOKEN")
    parser.add_argument("--secret_aws_region", default="AWS_REGION")
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args()


def _try_get_colab_secret(name: str) -> str:
    if not name:
        return ""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return ""
    try:
        return (userdata.get(name) or "").strip()
    except Exception:
        return ""


def _resolve_gpu_info() -> Tuple[bool, int, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False, 0, ""
        first = (proc.stdout or "").strip().splitlines()[0]
        if not first:
            return False, 0, ""
        parts = [p.strip() for p in first.split(",", maxsplit=1)]
        mem_mb = int(parts[0]) if parts and parts[0].isdigit() else 0
        name = parts[1] if len(parts) > 1 else ""
        return True, mem_mb, name
    except Exception:
        return False, 0, ""


def _recommend_batch_size(mem_mb: int) -> int:
    if mem_mb >= 45000:
        return 96
    if mem_mb >= 20000:
        return 64
    if mem_mb >= 14000:
        return 48
    if mem_mb >= 10000:
        return 32
    return 16


def _configure_aws_env(args: argparse.Namespace) -> None:
    env_map: Dict[str, str] = {
        "AWS_ACCESS_KEY_ID": (args.aws_access_key_id or "").strip(),
        "AWS_SECRET_ACCESS_KEY": (args.aws_secret_access_key or "").strip(),
        "AWS_SESSION_TOKEN": (args.aws_session_token or "").strip(),
        "AWS_REGION": (args.aws_region or "").strip(),
    }
    secret_names = {
        "AWS_ACCESS_KEY_ID": args.secret_aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": args.secret_aws_secret_access_key,
        "AWS_SESSION_TOKEN": args.secret_aws_session_token,
        "AWS_REGION": args.secret_aws_region,
    }

    if args.use_colab_secrets:
        for key, secret_name in secret_names.items():
            if env_map[key]:
                continue
            env_map[key] = _try_get_colab_secret(secret_name)

    for key, val in env_map.items():
        if val:
            os.environ[key] = val


def _set_faster_whisper_gpu_defaults(args: argparse.Namespace) -> None:
    has_gpu, mem_mb, gpu_name = _resolve_gpu_info()
    if args.require_gpu and not has_gpu:
        raise RuntimeError("No NVIDIA GPU detected in this runtime. Choose a GPU runtime in Colab.")

    if has_gpu:
        print(f"[gpu] detected: {gpu_name or 'unknown'} ({mem_mb} MB)")
    else:
        print("[gpu] not detected; falling back to CPU settings")

    device = (args.faster_whisper_device or "cuda").strip().lower()
    if not has_gpu and device in {"cuda", "gpu"}:
        device = "cpu"

    batch_size = int(args.faster_whisper_batch_size or 0)
    if batch_size <= 0:
        batch_size = _recommend_batch_size(mem_mb) if has_gpu else 8

    defaults = {
        "FASTER_WHISPER_DEVICE": device,
        "FASTER_WHISPER_COMPUTE_TYPE": (args.faster_whisper_compute_type or ("float16" if has_gpu else "int8")).strip(),
        "FASTER_WHISPER_BATCHED": "1" if args.faster_whisper_batched else "0",
        "FASTER_WHISPER_BATCH_SIZE": str(max(1, batch_size)),
        "FASTER_WHISPER_BEAM_SIZE": str(max(1, int(args.faster_whisper_beam_size))),
        "FASTER_WHISPER_BEST_OF": str(max(1, int(args.faster_whisper_best_of))),
        "FASTER_WHISPER_VAD_FILTER": "1" if args.faster_whisper_vad_filter else "0",
        "FASTER_WHISPER_CPU_THREADS": str(max(0, int(args.faster_whisper_cpu_threads))),
        "FASTER_WHISPER_LANGUAGE": (args.faster_whisper_language or "").strip(),
    }
    for key, value in defaults.items():
        os.environ[key] = value

    print(
        "[faster-whisper] "
        f"device={defaults['FASTER_WHISPER_DEVICE']} "
        f"compute_type={defaults['FASTER_WHISPER_COMPUTE_TYPE']} "
        f"batched={defaults['FASTER_WHISPER_BATCHED']} "
        f"batch_size={defaults['FASTER_WHISPER_BATCH_SIZE']}",
        flush=True,
    )


def _install_dependencies(python_bin: str) -> None:
    deps = ["yt-dlp", "requests", "boto3", "faster-whisper"]
    cmd = [python_bin, "-m", "pip", "install", "-q", "-U", *deps]
    print("[deps] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    _configure_aws_env(args)
    if not args.s3_bucket and args.use_colab_secrets:
        args.s3_bucket = _try_get_colab_secret(args.secret_s3_bucket)
    args.s3_bucket = (args.s3_bucket or "").strip()
    if not args.s3_bucket:
        raise ValueError("--s3_bucket is required (or provide secret S3_BUCKET in Colab)")

    if args.install_deps:
        _install_dependencies(args.python_bin)
    _set_faster_whisper_gpu_defaults(args)

    state_db = (REPO_ROOT / args.state_db).resolve() if not Path(args.state_db).is_absolute() else Path(args.state_db)
    state_db.parent.mkdir(parents=True, exist_ok=True)
    state_s3_key = f"{args.state_prefix.strip('/')}/text_downloader.sqlite"

    s3_region = (args.s3_region or os.getenv("AWS_REGION", "")).strip()
    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=s3_region)
    restored = s3.restore_state_if_exists(state_s3_key, state_db)
    print(f"[state] restore {'ok' if restored else 'skip'}: s3://{args.s3_bucket}/{state_s3_key}")

    cloud_root_uri = f"s3://{args.s3_bucket}/{args.raw_prefix.strip('/')}"
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "Data/raw/Text/text_collect.py"),
        "--metadata_csv",
        str(args.metadata_csv),
        "--state_db",
        str(state_db),
        "--cloud_root_uri",
        cloud_root_uri,
        "--cloud_audio_prefix",
        args.audio_prefix,
        "--cloud_text_prefix",
        args.text_prefix,
        "--asr_backend",
        args.asr_backend,
        "--asr_model",
        args.asr_model,
        "--max_workers",
        str(max(1, int(args.max_workers))),
        "--faster_whisper_device",
        os.environ.get("FASTER_WHISPER_DEVICE", "auto"),
        "--faster_whisper_compute_type",
        os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", ""),
        "--faster_whisper_batch_size",
        os.environ.get("FASTER_WHISPER_BATCH_SIZE", "16"),
        "--faster_whisper_cpu_threads",
        os.environ.get("FASTER_WHISPER_CPU_THREADS", "0"),
        "--faster_whisper_beam_size",
        os.environ.get("FASTER_WHISPER_BEAM_SIZE", "1"),
        "--faster_whisper_best_of",
        os.environ.get("FASTER_WHISPER_BEST_OF", "1"),
    ]
    if args.max_items > 0:
        cmd.extend(["--max_items", str(args.max_items)])
    cmd.append("--caption_first" if args.caption_first else "--no-caption_first")
    cmd.append("--faster_whisper_batched" if os.environ.get("FASTER_WHISPER_BATCHED", "1") == "1" else "--no-faster_whisper_batched")
    cmd.append("--faster_whisper_vad_filter" if os.environ.get("FASTER_WHISPER_VAD_FILTER", "1") == "1" else "--no-faster_whisper_vad_filter")
    fw_lang = os.environ.get("FASTER_WHISPER_LANGUAGE", "").strip()
    if fw_lang:
        cmd.extend(["--faster_whisper_language", fw_lang])
    cmd.append(
        "--download_audio_from_cloud_if_missing"
        if args.download_audio_from_cloud_if_missing
        else "--no-download_audio_from_cloud_if_missing"
    )
    cmd.append("--cleanup_downloaded_audio" if args.cleanup_downloaded_audio else "--no-cleanup_downloaded_audio")
    if args.cloud_delete_local_after_upload:
        cmd.append("--cloud_delete_local_after_upload")

    stop_event = threading.Event()

    def checkpoint_loop() -> None:
        interval = max(0, int(args.checkpoint_seconds))
        if interval <= 0:
            return
        while not stop_event.wait(interval):
            try:
                if state_db.exists():
                    s3.persist_state(state_db, state_s3_key)
                    print(f"[state] checkpoint: s3://{args.s3_bucket}/{state_s3_key}", flush=True)
            except Exception as exc:
                print(f"[state] checkpoint_error: {exc}", flush=True)

    ckpt_thread = threading.Thread(target=checkpoint_loop, daemon=True)
    ckpt_thread.start()

    exit_code = 0
    try:
        print("[run] " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
        exit_code = proc.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, cmd)
    finally:
        stop_event.set()
        ckpt_thread.join(timeout=2)
        try:
            if state_db.exists():
                s3.persist_state(state_db, state_s3_key)
                print(f"[state] persisted: s3://{args.s3_bucket}/{state_s3_key}", flush=True)
        except Exception as exc:
            print(f"[state] persist_error: {exc}", flush=True)

    print(f"[done] exit_code={exit_code}")


if __name__ == "__main__":
    main()
