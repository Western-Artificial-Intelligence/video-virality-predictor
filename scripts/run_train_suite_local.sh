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
SNAPSHOT_PREFIX="${SNAPSHOT_PREFIX:-clipfarm/models/snapshots}"
RUN_ID="${RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEED="${SEED:-42}"
SYNC_METADATA_FROM_ORIGIN="${SYNC_METADATA_FROM_ORIGIN:-1}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
ORIGIN_BRANCH="${ORIGIN_BRANCH:-main}"

STRATEGIES="${STRATEGIES:-concat,sum_pool,max_pool}"
HORIZONS="${HORIZONS:-7,30}"
RANK_METRIC="${RANK_METRIC:-rmse_log}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
PATIENCE="${PATIENCE:-6}"
PROJECTOR_DIM="${PROJECTOR_DIM:-128}"

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "ERROR: MAX_PARALLEL must be a positive integer (current: $MAX_PARALLEL)"
  exit 1
fi

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in boto3 pandas numpy pyarrow sklearn torch joblib; do
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

IFS=',' read -r -a STRATEGY_ARR <<< "$STRATEGIES"
IFS=',' read -r -a HORIZON_ARR <<< "$HORIZONS"

job_specs=()
for strategy in "${STRATEGY_ARR[@]}"; do
  s="$(echo "$strategy" | xargs)"
  [[ -z "$s" ]] && continue
  for horizon in "${HORIZON_ARR[@]}"; do
    h="$(echo "$horizon" | xargs)"
    [[ -z "$h" ]] && continue
    job_specs+=("${s}:${h}")
  done
done

if [[ ${#job_specs[@]} -eq 0 ]]; then
  echo "ERROR: no jobs to run. Check STRATEGIES/HORIZONS env values."
  exit 1
fi

LOG_DIR="state/train_suite_logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "[train] run_id=$RUN_ID"
echo "[train] snapshot_prefix=$SNAPSHOT_PREFIX"
echo "[train] max_parallel=$MAX_PARALLEL"
echo "[train] jobs=${#job_specs[@]} (${job_specs[*]})"

declare -a active_pids=()
declare -a active_specs=()
next_idx=0
fail_count=0
declare -a failed_specs=()

total_jobs=${#job_specs[@]}

run_train_job() {
  local strategy="$1"
  local horizon="$2"
  local fused_manifest_key="clipfarm/fused/${strategy}/fused_manifest.parquet"
  local log_file="$LOG_DIR/${strategy}_h${horizon}.log"

  echo "[job] start strategy=${strategy} horizon=${horizon} log=${log_file}"
  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" Super_Predict/train_suite_from_horizon.py \
    --metadata_csv "$METADATA_CSV" \
    --s3_bucket "$S3_BUCKET" \
    --s3_region "$AWS_REGION" \
    --fused_manifest_s3_key "$fused_manifest_key" \
    --fusion_strategy "$strategy" \
    --target_horizon_days "$horizon" \
    --snapshot_prefix "$SNAPSHOT_PREFIX" \
    --run_id "$RUN_ID" \
    --seed "$SEED" \
    --rank_metric "$RANK_METRIC" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --projector_dim "$PROJECTOR_DIM" \
    >"$log_file" 2>&1
}

while [[ $next_idx -lt $total_jobs || ${#active_pids[@]} -gt 0 ]]; do
  while [[ $next_idx -lt $total_jobs && ${#active_pids[@]} -lt $MAX_PARALLEL ]]; do
    spec="${job_specs[$next_idx]}"
    strategy="${spec%%:*}"
    horizon="${spec##*:}"

    run_train_job "$strategy" "$horizon" &
    pid=$!

    active_pids+=("$pid")
    active_specs+=("$spec")

    echo "[job] launched ${spec} pid=${pid}"
    next_idx=$((next_idx + 1))
  done

  sleep 1

  new_pids=()
  new_specs=()
  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    spec="${active_specs[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_specs+=("$spec")
      continue
    fi

    set +e
    wait "$pid"
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
      fail_count=$((fail_count + 1))
      failed_specs+=("$spec")
      echo "[job] failed ${spec} (rc=$rc)"
      echo "[job] log: $LOG_DIR/${spec%%:*}_h${spec##*:}.log"
    else
      echo "[job] completed ${spec}"
    fi
  done

  active_pids=("${new_pids[@]}")
  active_specs=("${new_specs[@]}")
done

if [[ $fail_count -ne 0 ]]; then
  echo "ERROR: ${fail_count} training jobs failed."
  for spec in "${failed_specs[@]}"; do
    echo "  - ${spec} (log: $LOG_DIR/${spec%%:*}_h${spec##*:}.log)"
  done
  exit 1
fi

COMPARE_OUT_DIR="state/train_suite_logs/${RUN_ID}/comparison"
mkdir -p "$COMPARE_OUT_DIR"

echo "[compare] aggregating run metrics"
"$_PYTHON_BIN" Super_Predict/aggregate_train_suite_results.py \
  --s3_bucket "$S3_BUCKET" \
  --s3_region "$AWS_REGION" \
  --snapshot_prefix "$SNAPSHOT_PREFIX" \
  --run_id "$RUN_ID" \
  --strategies "$STRATEGIES" \
  --horizons "$HORIZONS" \
  --rank_metric "$RANK_METRIC" \
  --output_dir "$COMPARE_OUT_DIR"

echo "[compare] local outputs: $COMPARE_OUT_DIR"
if [[ -f "$COMPARE_OUT_DIR/metrics_comparison.csv" ]]; then
  echo "[compare] top rows"
  "$_PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

csv_path = Path("$COMPARE_OUT_DIR/metrics_comparison.csv")
if csv_path.exists() and csv_path.stat().st_size > 0:
    df = pd.read_csv(csv_path)
    rank_col = "$RANK_METRIC"
    if not rank_col.startswith("val_"):
        rank_col = f"val_{rank_col}"
    if rank_col in df.columns:
        df = df.sort_values(rank_col, ascending=True)
    cols = [
        c
        for c in [
            "fusion_strategy",
            "target_horizon_days",
            "model",
            rank_col,
            "val_mae_log",
            "val_rmse_log",
        ]
        if c in df.columns
    ]
    if cols:
        print(df[cols].head(10).to_string(index=False))
PY
fi

echo "[done] training suite run completed"
