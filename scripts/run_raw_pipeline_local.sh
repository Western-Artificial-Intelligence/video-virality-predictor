#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python binary selection: explicit env override > project venv > system python.
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
"$_PYTHON_BIN" -V >/dev/null 2>&1 || {
  echo "ERROR: python binary is not runnable: $_PYTHON_BIN"
  exit 1
}

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is required for S3 state restore/persist and uploads"
  exit 1
}

S3_BUCKET="${S3_BUCKET:-}"
RAW_PREFIX="${RAW_PREFIX:-clipfarm/raw}"
STATE_PREFIX="${STATE_PREFIX:-clipfarm/state}"
VIDEO_PREFIX="${VIDEO_PREFIX:-video}"
AUDIO_PREFIX="${AUDIO_PREFIX:-audio}"
TEXT_PREFIX="${TEXT_PREFIX:-text}"

if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: S3_BUCKET is required (example: export S3_BUCKET=clipfarm-prod-us-west-2)"
  exit 1
fi

METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
MAX_ITEMS="${MAX_ITEMS:-100}"
SYNC_METADATA_FROM_ORIGIN="${SYNC_METADATA_FROM_ORIGIN:-1}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
ORIGIN_BRANCH="${ORIGIN_BRANCH:-main}"

COOKIES_FILE="${COOKIES_FILE:-}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
AUTO_UPGRADE_YTDLP="${AUTO_UPGRADE_YTDLP:-1}"

VIDEO_RETRY="${VIDEO_RETRY:-1}"
VIDEO_CHALLENGE_COOLDOWN_HOURS="${VIDEO_CHALLENGE_COOLDOWN_HOURS:-24}"
VIDEO_UPCOMING_COOLDOWN_HOURS="${VIDEO_UPCOMING_COOLDOWN_HOURS:-48}"
VIDEO_SLEEP_INTERVAL="${VIDEO_SLEEP_INTERVAL:-1.5}"
VIDEO_MAX_SLEEP_INTERVAL="${VIDEO_MAX_SLEEP_INTERVAL:-4.0}"

TEXT_ASR_BACKEND="${TEXT_ASR_BACKEND:-openai_api}"
TEXT_ASR_MODEL="${TEXT_ASR_MODEL:-}"
if [[ -z "$TEXT_ASR_MODEL" ]]; then
  if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]]; then
    TEXT_ASR_MODEL="whisper-1"
  else
    TEXT_ASR_MODEL="small"
  fi
fi

VIDEO_STATE_DB="${VIDEO_STATE_DB:-state/video_downloader.sqlite}"
TEXT_STATE_DB="${TEXT_STATE_DB:-state/text_downloader.sqlite}"
VIDEO_RAW_DIR="${VIDEO_RAW_DIR:-Data/raw/Video/raw_data}"
AUDIO_RAW_DIR="${AUDIO_RAW_DIR:-Data/raw/Audio/raw_data}"
TEXT_RAW_DIR="${TEXT_RAW_DIR:-Data/raw/Text/raw_data}"
CLEAN_LOCAL_RAW_AFTER_RUN="${CLEAN_LOCAL_RAW_AFTER_RUN:-1}"

VIDEO_STATE_KEY="${STATE_PREFIX}/video_downloader.sqlite"
TEXT_STATE_KEY="${STATE_PREFIX}/text_downloader.sqlite"
CLOUD_ROOT_URI="s3://${S3_BUCKET}/${RAW_PREFIX}"

mkdir -p "$(dirname "$VIDEO_STATE_DB")"
mkdir -p "$(dirname "$TEXT_STATE_DB")"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

# Preflight dependencies used by local raw pipeline.
missing_modules=()
for mod in yt_dlp requests boto3; do
  if ! check_python_module "$mod"; then
    missing_modules+=("$mod")
  fi
done
if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]] && ! check_python_module "openai"; then
  missing_modules+=("openai")
fi
if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "ERROR: missing python modules: ${missing_modules[*]}"
  echo "Install with:"
  echo "  $_PYTHON_BIN -m pip install ${missing_modules[*]}"
  exit 1
fi

if [[ "$TEXT_ASR_BACKEND" == "openai_api" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is required when TEXT_ASR_BACKEND=openai_api"
  exit 1
fi

if [[ "$AUTO_UPGRADE_YTDLP" == "1" ]]; then
  echo "[deps] upgrading yt-dlp in $_PYTHON_BIN environment"
  "$_PYTHON_BIN" -m pip install -q -U yt-dlp || echo "[deps] warning: yt-dlp auto-upgrade failed; continuing with installed version"
fi

restore_state() {
  echo "[state] restoring from s3://${S3_BUCKET}/${STATE_PREFIX}"
  aws s3 cp "s3://${S3_BUCKET}/${VIDEO_STATE_KEY}" "$VIDEO_STATE_DB" >/dev/null 2>&1 || true
  aws s3 cp "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" "$TEXT_STATE_DB" >/dev/null 2>&1 || true
}

persist_state() {
  echo "[state] persisting to s3://${S3_BUCKET}/${STATE_PREFIX}"
  [[ -f "$VIDEO_STATE_DB" ]] && aws s3 cp "$VIDEO_STATE_DB" "s3://${S3_BUCKET}/${VIDEO_STATE_KEY}" >/dev/null
  [[ -f "$TEXT_STATE_DB" ]] && aws s3 cp "$TEXT_STATE_DB" "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" >/dev/null
}

cleanup_local_raw_data() {
  if [[ "$CLEAN_LOCAL_RAW_AFTER_RUN" != "1" ]]; then
    echo "[cleanup] skipped (CLEAN_LOCAL_RAW_AFTER_RUN=${CLEAN_LOCAL_RAW_AFTER_RUN})"
    return
  fi

  local total_deleted=0
  for dir in "$VIDEO_RAW_DIR" "$AUDIO_RAW_DIR" "$TEXT_RAW_DIR"; do
    if [[ ! -d "$dir" ]]; then
      continue
    fi
    local before
    before=$(find "$dir" -type f | wc -l | tr -d ' ')
    if [[ "$before" -eq 0 ]]; then
      continue
    fi
    find "$dir" -type f -delete
    echo "[cleanup] deleted ${before} files under ${dir}"
    total_deleted=$((total_deleted + before))
  done
  echo "[cleanup] total files deleted: ${total_deleted}"
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

video_args=(
  Data/raw/Video/download_video.py
  --metadata_csv "$METADATA_CSV"
  --state_db "$VIDEO_STATE_DB"
  --cloud_root_uri "$CLOUD_ROOT_URI"
  --cloud_video_prefix "$VIDEO_PREFIX"
  --cloud_audio_prefix "$AUDIO_PREFIX"
  --cloud_delete_local_after_upload
  --retry "$VIDEO_RETRY"
  --challenge_cooldown_hours "$VIDEO_CHALLENGE_COOLDOWN_HOURS"
  --upcoming_cooldown_hours "$VIDEO_UPCOMING_COOLDOWN_HOURS"
  --sleep_interval "$VIDEO_SLEEP_INTERVAL"
  --max_sleep_interval "$VIDEO_MAX_SLEEP_INTERVAL"
)

if [[ "$MAX_ITEMS" != "0" ]]; then
  video_args+=(--max_items "$MAX_ITEMS")
fi

if [[ -n "$COOKIES_FILE" ]]; then
  video_args+=(--cookies_file "$COOKIES_FILE")
elif [[ -n "$COOKIES_FROM_BROWSER" ]]; then
  video_args+=(--cookies_from_browser "$COOKIES_FROM_BROWSER")
fi

echo "[video] starting downloader"
echo "[env] python_bin=$_PYTHON_BIN"
echo "[env] cookies_file=${COOKIES_FILE:-<none>} cookies_from_browser=${COOKIES_FROM_BROWSER:-<none>}"
video_cmd=("$_PYTHON_BIN" "${video_args[@]}")
PYTHONUNBUFFERED=1 "${video_cmd[@]}"

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
text_cmd=("$_PYTHON_BIN" "${text_args[@]}")
PYTHONUNBUFFERED=1 "${text_cmd[@]}"

cleanup_local_raw_data

echo "[done] raw pipeline completed"
