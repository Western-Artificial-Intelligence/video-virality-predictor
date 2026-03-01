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

S3_BUCKET="${S3_BUCKET:-}"
if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: S3_BUCKET is required (example: export S3_BUCKET=clipfarm-prod-us-west-2)"
  exit 1
fi

AWS_REGION="${AWS_REGION:-}"
METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
EMB_PREFIX="${EMB_PREFIX:-clipfarm/embeddings}"
FUSED_PREFIX_BASE="${FUSED_PREFIX_BASE:-clipfarm/fused}"
TEXT_STATE_S3_KEY="${TEXT_STATE_S3_KEY:-clipfarm/state/text_downloader.sqlite}"
TERMINAL_TEXT_STATUSES="${TERMINAL_TEXT_STATUSES:-no_captions,fail_empty_transcript_terminal}"
TEXT_DIM="${TEXT_DIM:-768}"
MAX_FAIL_RETRIES="${MAX_FAIL_RETRIES:-3}"
SAMPLE_IDS_PER_STATUS="${SAMPLE_IDS_PER_STATUS:-10}"
MAX_ITEMS="${MAX_ITEMS:-0}"
RUN_DATE="${RUN_DATE:-}"
STRATEGIES="${STRATEGIES:-concat,sum_pool,max_pool}"
SYNC_METADATA_FROM_ORIGIN="${SYNC_METADATA_FROM_ORIGIN:-1}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
ORIGIN_BRANCH="${ORIGIN_BRANCH:-main}"
INCLUDE_CAPTURED_AT_IN_HASH="${INCLUDE_CAPTURED_AT_IN_HASH:-0}"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in boto3 numpy pandas pyarrow; do
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

IFS=',' read -r -a strategy_arr <<< "$STRATEGIES"
if [[ ${#strategy_arr[@]} -eq 0 ]]; then
  echo "ERROR: STRATEGIES is empty. Use comma-separated values (concat,sum_pool,max_pool)."
  exit 1
fi

echo "[fusion] starting local fusion pipeline"
echo "[fusion] strategies=${STRATEGIES}"
echo "[fusion] max_items=${MAX_ITEMS} run_date=${RUN_DATE:-<utc_today>}"

for raw_strategy in "${strategy_arr[@]}"; do
  strategy="$(echo "$raw_strategy" | xargs | tr '[:upper:]' '[:lower:]')"
  [[ -z "$strategy" ]] && continue
  case "$strategy" in
    concat|sum_pool|max_pool) ;;
    *)
      echo "ERROR: unknown strategy '$raw_strategy' (allowed: concat,sum_pool,max_pool)"
      exit 1
      ;;
  esac

  echo "[fusion] strategy=${strategy}: start"

  args=(
    Data/common/fuse_embeddings_delta.py
    --metadata_csv "$METADATA_CSV"
    --s3_bucket "$S3_BUCKET"
    --s3_region "$AWS_REGION"
    --emb_prefix "$EMB_PREFIX"
    --fusion_strategy "$strategy"
    --fused_prefix_base "$FUSED_PREFIX_BASE"
    --state_db "state/fusion_${strategy}.sqlite"
    --state_s3_key "clipfarm/state/fusion_${strategy}.sqlite"
    --text_state_s3_key "$TEXT_STATE_S3_KEY"
    --terminal_text_statuses "$TERMINAL_TEXT_STATUSES"
    --text_dim "$TEXT_DIM"
    --max_fail_retries "$MAX_FAIL_RETRIES"
    --sample_ids_per_status "$SAMPLE_IDS_PER_STATUS"
    --max_items "$MAX_ITEMS"
  )

  if [[ -n "$RUN_DATE" ]]; then
    args+=(--run_date "$RUN_DATE")
  fi
  if [[ "$INCLUDE_CAPTURED_AT_IN_HASH" == "1" ]]; then
    args+=(--include_captured_at_in_hash)
  fi

  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" "${args[@]}"

  echo "[fusion] strategy=${strategy}: done"
done

echo "[done] fusion pipeline completed"

