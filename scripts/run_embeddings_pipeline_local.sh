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
"$_PYTHON_BIN" -V >/dev/null 2>&1 || {
  echo "ERROR: python binary is not runnable: $_PYTHON_BIN"
  exit 1
}

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is required for state restore/persist"
  exit 1
}

S3_BUCKET="${S3_BUCKET:-}"
if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: S3_BUCKET is required (example: export S3_BUCKET=clipfarm-prod-us-west-2)"
  exit 1
fi

AWS_REGION="${AWS_REGION:-}"
METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
RAW_PREFIX="${RAW_PREFIX:-clipfarm/raw}"
EMB_PREFIX="${EMB_PREFIX:-clipfarm/embeddings}"
STATE_PREFIX="${STATE_PREFIX:-clipfarm/state}"
STAGES="${STAGES:-video,audio,text}"
MAX_ITEMS="${MAX_ITEMS:-0}"
INCLUDE_CAPTURED_AT_IN_HASH="${INCLUDE_CAPTURED_AT_IN_HASH:-0}"
SYNC_METADATA_FROM_ORIGIN="${SYNC_METADATA_FROM_ORIGIN:-1}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
ORIGIN_BRANCH="${ORIGIN_BRANCH:-main}"

VIDEO_STATE_DB="${VIDEO_STATE_DB:-state/video_embedding.sqlite}"
AUDIO_STATE_DB="${AUDIO_STATE_DB:-state/audio_embedding.sqlite}"
TEXT_STATE_DB="${TEXT_STATE_DB:-state/text_embedding.sqlite}"

VIDEO_STATE_KEY="${STATE_PREFIX}/video_embedding.sqlite"
AUDIO_STATE_KEY="${STATE_PREFIX}/audio_embedding.sqlite"
TEXT_STATE_KEY="${STATE_PREFIX}/text_embedding.sqlite"

VIDEO_MODEL_NAME="${VIDEO_MODEL_NAME:-}"
AUDIO_MODEL_NAME="${AUDIO_MODEL_NAME:-}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-16}"
VIDEO_DEVICE="${VIDEO_DEVICE:-cpu}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"
AUDIO_MAX_SECONDS="${AUDIO_MAX_SECONDS:-90}"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in boto3 numpy pandas torch transformers; do
  if ! check_python_module "$mod"; then
    missing_modules+=("$mod")
  fi
done
if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "ERROR: missing python modules: ${missing_modules[*]}"
  echo "Install with:"
  echo "  $_PYTHON_BIN -m pip install ${missing_modules[*]}"
  exit 1
fi

if [[ "$SYNC_METADATA_FROM_ORIGIN" == "1" ]]; then
  echo "[metadata] syncing ${METADATA_CSV} from ${ORIGIN_REMOTE}/${ORIGIN_BRANCH}"
  git fetch "$ORIGIN_REMOTE" "$ORIGIN_BRANCH" --quiet
  git checkout "${ORIGIN_REMOTE}/${ORIGIN_BRANCH}" -- "$METADATA_CSV"
fi

if [[ ! -f "$METADATA_CSV" ]]; then
  echo "ERROR: metadata csv not found: $METADATA_CSV"
  exit 1
fi

mkdir -p "$(dirname "$VIDEO_STATE_DB")" "$(dirname "$AUDIO_STATE_DB")" "$(dirname "$TEXT_STATE_DB")"

restore_state() {
  echo "[state] restoring from s3://${S3_BUCKET}/${STATE_PREFIX}"
  aws s3 cp "s3://${S3_BUCKET}/${VIDEO_STATE_KEY}" "$VIDEO_STATE_DB" >/dev/null 2>&1 || true
  aws s3 cp "s3://${S3_BUCKET}/${AUDIO_STATE_KEY}" "$AUDIO_STATE_DB" >/dev/null 2>&1 || true
  aws s3 cp "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" "$TEXT_STATE_DB" >/dev/null 2>&1 || true
}

persist_state() {
  echo "[state] persisting to s3://${S3_BUCKET}/${STATE_PREFIX}"
  [[ -f "$VIDEO_STATE_DB" ]] && aws s3 cp "$VIDEO_STATE_DB" "s3://${S3_BUCKET}/${VIDEO_STATE_KEY}" >/dev/null || true
  [[ -f "$AUDIO_STATE_DB" ]] && aws s3 cp "$AUDIO_STATE_DB" "s3://${S3_BUCKET}/${AUDIO_STATE_KEY}" >/dev/null || true
  [[ -f "$TEXT_STATE_DB" ]] && aws s3 cp "$TEXT_STATE_DB" "s3://${S3_BUCKET}/${TEXT_STATE_KEY}" >/dev/null || true
}

trap persist_state EXIT

restore_state

run_video_stage() {
  echo "[embed-video] starting"
  local args=(Data/embeddings/video/embed_video_delta.py)
  args+=(--metadata_csv "$METADATA_CSV")
  args+=(--s3_bucket "$S3_BUCKET")
  args+=(--s3_region "$AWS_REGION")
  args+=(--raw_prefix "$RAW_PREFIX")
  args+=(--emb_prefix "$EMB_PREFIX")
  args+=(--max_items "$MAX_ITEMS")
  if [[ "$INCLUDE_CAPTURED_AT_IN_HASH" == "1" ]]; then
    args+=(--include_captured_at_in_hash)
  fi
  args+=(--state_db "$VIDEO_STATE_DB")
  args+=(--state_s3_key "$VIDEO_STATE_KEY")
  args+=(--num_frames "$VIDEO_NUM_FRAMES")
  args+=(--device "$VIDEO_DEVICE")
  if [[ -n "$VIDEO_MODEL_NAME" ]]; then
    args+=(--model_name "$VIDEO_MODEL_NAME")
  fi
  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" "${args[@]}"
}

run_audio_stage() {
  echo "[embed-audio] starting"
  local args=(Data/embeddings/audio/embed_audio_delta.py)
  args+=(--metadata_csv "$METADATA_CSV")
  args+=(--s3_bucket "$S3_BUCKET")
  args+=(--s3_region "$AWS_REGION")
  args+=(--raw_prefix "$RAW_PREFIX")
  args+=(--emb_prefix "$EMB_PREFIX")
  args+=(--max_items "$MAX_ITEMS")
  if [[ "$INCLUDE_CAPTURED_AT_IN_HASH" == "1" ]]; then
    args+=(--include_captured_at_in_hash)
  fi
  args+=(--state_db "$AUDIO_STATE_DB")
  args+=(--state_s3_key "$AUDIO_STATE_KEY")
  args+=(--sample_rate "$AUDIO_SAMPLE_RATE")
  args+=(--max_audio_seconds "$AUDIO_MAX_SECONDS")
  if [[ -n "$AUDIO_MODEL_NAME" ]]; then
    args+=(--model_name "$AUDIO_MODEL_NAME")
  fi
  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" "${args[@]}"
}

run_text_stage() {
  echo "[embed-text] starting"
  local args=(Data/embeddings/text/embed_text_delta.py)
  args+=(--metadata_csv "$METADATA_CSV")
  args+=(--s3_bucket "$S3_BUCKET")
  args+=(--s3_region "$AWS_REGION")
  args+=(--raw_prefix "$RAW_PREFIX")
  args+=(--emb_prefix "$EMB_PREFIX")
  args+=(--max_items "$MAX_ITEMS")
  if [[ "$INCLUDE_CAPTURED_AT_IN_HASH" == "1" ]]; then
    args+=(--include_captured_at_in_hash)
  fi
  args+=(--state_db "$TEXT_STATE_DB")
  args+=(--state_s3_key "$TEXT_STATE_KEY")
  if [[ -n "$TEXT_MODEL_NAME" ]]; then
    args+=(--model_name "$TEXT_MODEL_NAME")
  fi
  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" "${args[@]}"
}

IFS=',' read -r -a stage_arr <<< "$STAGES"
if [[ ${#stage_arr[@]} -eq 0 ]]; then
  echo "ERROR: STAGES is empty. Use comma-separated stages: video,audio,text"
  exit 1
fi

for raw_stage in "${stage_arr[@]}"; do
  stage="$(echo "$raw_stage" | xargs | tr '[:upper:]' '[:lower:]')"
  case "$stage" in
    video)
      for mod in av PIL; do
        if ! check_python_module "$mod"; then
          missing_modules+=("$mod")
        fi
      done
      ;;
    audio)
      for mod in librosa; do
        if ! check_python_module "$mod"; then
          missing_modules+=("$mod")
        fi
      done
      ;;
    text)
      for mod in sentence_transformers; do
        if ! check_python_module "$mod"; then
          missing_modules+=("$mod")
        fi
      done
      ;;
    "") ;;
    *)
      echo "ERROR: unknown stage '$raw_stage' (allowed: video,audio,text)"
      exit 1
      ;;
  esac
done

if [[ ${#missing_modules[@]} -gt 0 ]]; then
  uniq_missing=($(printf "%s\n" "${missing_modules[@]}" | awk '!seen[$0]++'))
  echo "ERROR: missing stage dependencies: ${uniq_missing[*]}"
  echo "Install with:"
  echo "  $_PYTHON_BIN -m pip install ${uniq_missing[*]}"
  exit 1
fi

for raw_stage in "${stage_arr[@]}"; do
  stage="$(echo "$raw_stage" | xargs | tr '[:upper:]' '[:lower:]')"
  case "$stage" in
    video) run_video_stage ;;
    audio) run_audio_stage ;;
    text) run_text_stage ;;
    "")
      ;;
    *)
      echo "ERROR: unknown stage '$raw_stage' (allowed: video,audio,text)"
      exit 1
      ;;
  esac
done

echo "[done] embedding pipeline completed"
