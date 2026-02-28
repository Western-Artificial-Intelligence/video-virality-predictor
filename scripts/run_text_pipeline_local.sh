#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  _PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  _PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
else
  _PYTHON_BIN="python3"
fi

command -v "$_PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: python binary not found: $_PYTHON_BIN"
  exit 1
}
command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is required"
  exit 1
}

S3_BUCKET="${S3_BUCKET:-}"
if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: S3_BUCKET is required (example: export S3_BUCKET=clipfarm-prod-us-west-2)"
  exit 1
fi

METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
RAW_PREFIX="${RAW_PREFIX:-clipfarm/raw}"
STATE_PREFIX="${STATE_PREFIX:-clipfarm/state}"
AUDIO_PREFIX="${AUDIO_PREFIX:-audio}"
TEXT_PREFIX="${TEXT_PREFIX:-text}"

MAX_ITEMS="${MAX_ITEMS:-0}"
TEXT_MAX_WORKERS="${TEXT_MAX_WORKERS:-4}"
SYNC_METADATA_FROM_ORIGIN="${SYNC_METADATA_FROM_ORIGIN:-1}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
ORIGIN_BRANCH="${ORIGIN_BRANCH:-main}"

TEXT_ASR_BACKEND="${TEXT_ASR_BACKEND:-whisper_cpp}"
TEXT_ASR_MODEL="${TEXT_ASR_MODEL:-}"
if [[ -z "$TEXT_ASR_MODEL" ]]; then
  if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]]; then
    TEXT_ASR_MODEL="whisper-1"
  else
    TEXT_ASR_MODEL="small"
  fi
fi

COOKIES_FILE="${COOKIES_FILE:-}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
AUTO_UPGRADE_YTDLP="${AUTO_UPGRADE_YTDLP:-1}"

TEXT_STATE_DB="${TEXT_STATE_DB:-state/text_downloader.sqlite}"
TEXT_RAW_DIR="${TEXT_RAW_DIR:-Data/raw/Text/raw_data}"
AUDIO_RAW_DIR="${AUDIO_RAW_DIR:-Data/raw/Audio/raw_data}"
CLEAN_LOCAL_TEXT_AFTER_RUN="${CLEAN_LOCAL_TEXT_AFTER_RUN:-1}"
CLEAN_LOCAL_AUDIO_AFTER_RUN="${CLEAN_LOCAL_AUDIO_AFTER_RUN:-0}"

TEXT_STATE_KEY="${STATE_PREFIX}/text_downloader.sqlite"
CLOUD_ROOT_URI="s3://${S3_BUCKET}/${RAW_PREFIX}"
mkdir -p "$(dirname "$TEXT_STATE_DB")"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in yt_dlp requests boto3; do
  if ! check_python_module "$mod"; then
    missing_modules+=("$mod")
  fi
done
if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]] && ! check_python_module "openai"; then
  missing_modules+=("openai")
fi
if [[ "$TEXT_ASR_BACKEND" == "faster_whisper" ]] && ! check_python_module "faster_whisper"; then
  missing_modules+=("faster_whisper")
fi
if [[ "$TEXT_ASR_BACKEND" == "whisper" || "$TEXT_ASR_BACKEND" == "whisper_cpp" ]]; then
  WHISPER_CPP_BIN="${WHISPER_CPP_BIN:-}"
  if [[ -n "$WHISPER_CPP_BIN" ]]; then
    if ! command -v "$WHISPER_CPP_BIN" >/dev/null 2>&1 && [[ ! -x "$WHISPER_CPP_BIN" ]]; then
      missing_modules+=("whisper.cpp-binary")
    fi
  else
    if ! command -v whisper-cli >/dev/null 2>&1 && ! command -v whisper-cpp >/dev/null 2>&1 && ! command -v main >/dev/null 2>&1; then
      missing_modules+=("whisper.cpp-binary")
    fi
  fi
fi
if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "ERROR: missing dependencies: ${missing_modules[*]}"
  if ! printf '%s\n' "${missing_modules[@]}" | grep -q "whisper.cpp-binary"; then
    echo "Install with:"
    echo "  $_PYTHON_BIN -m pip install ${missing_modules[*]}"
  else
    pip_mods=()
    for mod in "${missing_modules[@]}"; do
      if [[ "$mod" != "whisper.cpp-binary" ]]; then
        pip_mods+=("$mod")
      fi
    done
    if [[ ${#pip_mods[@]} -gt 0 ]]; then
      echo "Install python deps with:"
      echo "  $_PYTHON_BIN -m pip install ${pip_mods[*]}"
    fi
    echo "  Also install whisper.cpp binary (whisper-cli) and set WHISPER_CPP_BIN / WHISPER_CPP_MODEL_DIR."
  fi
  exit 1
fi

if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is required when TEXT_ASR_BACKEND=openai_api"
  exit 1
fi

if ! [[ "$TEXT_MAX_WORKERS" =~ ^[0-9]+$ ]] || [[ "$TEXT_MAX_WORKERS" -lt 1 ]]; then
  echo "ERROR: TEXT_MAX_WORKERS must be a positive integer (current: $TEXT_MAX_WORKERS)"
  exit 1
fi

if [[ "$AUTO_UPGRADE_YTDLP" == "1" ]]; then
  echo "[deps] upgrading yt-dlp in $_PYTHON_BIN environment"
  "$_PYTHON_BIN" -m pip install -q -U yt-dlp || echo "[deps] warning: yt-dlp auto-upgrade failed; continuing"
fi

restore_state() {
  echo "[state] restoring from s3://${S3_BUCKET}/${STATE_PREFIX}"
  aws s3 cp "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" "$TEXT_STATE_DB" >/dev/null 2>&1 || true
}

persist_state() {
  echo "[state] persisting to s3://${S3_BUCKET}/${STATE_PREFIX}"
  [[ -f "$TEXT_STATE_DB" ]] && aws s3 cp "$TEXT_STATE_DB" "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" >/dev/null
}

cleanup_local() {
  if [[ "$CLEAN_LOCAL_TEXT_AFTER_RUN" == "1" && -d "$TEXT_RAW_DIR" ]]; then
    local n_text
    n_text=$(find "$TEXT_RAW_DIR" -type f | wc -l | tr -d ' ')
    if [[ "$n_text" -gt 0 ]]; then
      find "$TEXT_RAW_DIR" -type f -delete
      echo "[cleanup] deleted ${n_text} files under ${TEXT_RAW_DIR}"
    fi
  fi

  if [[ "$CLEAN_LOCAL_AUDIO_AFTER_RUN" == "1" && -d "$AUDIO_RAW_DIR" ]]; then
    local n_audio
    n_audio=$(find "$AUDIO_RAW_DIR" -type f | wc -l | tr -d ' ')
    if [[ "$n_audio" -gt 0 ]]; then
      find "$AUDIO_RAW_DIR" -type f -delete
      echo "[cleanup] deleted ${n_audio} files under ${AUDIO_RAW_DIR}"
    fi
  fi
}

trap persist_state EXIT

if [[ "$SYNC_METADATA_FROM_ORIGIN" == "1" ]]; then
  echo "[metadata] syncing ${METADATA_CSV} from ${ORIGIN_REMOTE}/${ORIGIN_BRANCH}"
  git fetch "$ORIGIN_REMOTE" "$ORIGIN_BRANCH" --quiet
  git checkout "${ORIGIN_REMOTE}/${ORIGIN_BRANCH}" -- "$METADATA_CSV"
fi

if [[ ! -f "$METADATA_CSV" ]]; then
  echo "ERROR: metadata csv not found: $METADATA_CSV"
  exit 1
fi

restore_state

text_args=(
  Data/raw/Text/text_collect.py
  --metadata_csv "$METADATA_CSV"
  --state_db "$TEXT_STATE_DB"
  --cloud_root_uri "$CLOUD_ROOT_URI"
  --cloud_audio_prefix "$AUDIO_PREFIX"
  --cloud_text_prefix "$TEXT_PREFIX"
  --cloud_delete_local_after_upload
  --asr_backend "$TEXT_ASR_BACKEND"
  --asr_model "$TEXT_ASR_MODEL"
  --max_workers "$TEXT_MAX_WORKERS"
)

if [[ "$MAX_ITEMS" != "0" ]]; then
  text_args+=(--max_items "$MAX_ITEMS")
fi

if [[ -n "$COOKIES_FILE" ]]; then
  text_args+=(--cookies_file "$COOKIES_FILE")
elif [[ -n "$COOKIES_FROM_BROWSER" ]]; then
  text_args+=(--cookies_from_browser "$COOKIES_FROM_BROWSER")
fi

echo "[text] starting collector"
echo "[env] python_bin=$_PYTHON_BIN asr_backend=${TEXT_ASR_BACKEND} asr_model=${TEXT_ASR_MODEL} max_workers=${TEXT_MAX_WORKERS}"
text_cmd=("$_PYTHON_BIN" "${text_args[@]}")
PYTHONUNBUFFERED=1 "${text_cmd[@]}"

cleanup_local

echo "[done] text pipeline completed"
