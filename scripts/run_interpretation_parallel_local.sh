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
AWS_REGION="${AWS_REGION:-}"
METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
CLUSTER_CSV="${CLUSTER_CSV:-Unsup_Cluster/cluster_results.csv}"
RAW_PREFIX="${RAW_PREFIX:-clipfarm/raw}"
OUTPUT_INTERPRETATION_CSV="${OUTPUT_INTERPRETATION_CSV:-Interpretation/interpretation.csv}"

WORKERS="${WORKERS:-4}"
FETCH_MISSING_MEDIA="${FETCH_MISSING_MEDIA:-1}"
RESUME_EXISTING_OUTPUT="${RESUME_EXISTING_OUTPUT:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
FPS_SAMPLE="${FPS_SAMPLE:-2}"
DIFF_THRESH="${DIFF_THRESH:-25}"
EDGE_THRESH="${EDGE_THRESH:-20}"
KEEP_TMP="${KEEP_TMP:-0}"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
  echo "ERROR: WORKERS must be a positive integer (current: $WORKERS)"
  exit 1
fi
if ! [[ "$CHECKPOINT_EVERY" =~ ^[0-9]+$ ]] || [[ "$CHECKPOINT_EVERY" -lt 1 ]]; then
  echo "ERROR: CHECKPOINT_EVERY must be a positive integer (current: $CHECKPOINT_EVERY)"
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
for mod in boto3 numpy pandas av; do
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

if [[ ! -f "$METADATA_CSV" ]]; then
  echo "ERROR: metadata csv not found: $METADATA_CSV"
  exit 1
fi
if [[ ! -f "$CLUSTER_CSV" ]]; then
  echo "ERROR: cluster csv not found: $CLUSTER_CSV"
  exit 1
fi

to_bool_flag() {
  local raw="$1"
  local positive_flag="$2"
  local negative_flag="$3"
  local normalized
  normalized="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$normalized" in
    1|true|yes|y|on)
      echo "$positive_flag"
      ;;
    0|false|no|n|off)
      echo "$negative_flag"
      ;;
    *)
      echo ""
      ;;
  esac
}

FETCH_MISSING_FLAG="$(to_bool_flag "$FETCH_MISSING_MEDIA" "--fetch_missing" "--no-fetch_missing")"
if [[ -z "$FETCH_MISSING_FLAG" ]]; then
  echo "ERROR: FETCH_MISSING_MEDIA must be boolean-like (1/0/true/false/yes/no), got: $FETCH_MISSING_MEDIA"
  exit 1
fi

RESUME_BOOL="$(to_bool_flag "$RESUME_EXISTING_OUTPUT" "1" "0")"
if [[ -z "$RESUME_BOOL" ]]; then
  echo "ERROR: RESUME_EXISTING_OUTPUT must be boolean-like (1/0/true/false/yes/no), got: $RESUME_EXISTING_OUTPUT"
  exit 1
fi

if [[ "$FETCH_MISSING_FLAG" == "--fetch_missing" ]]; then
  if [[ -z "$S3_BUCKET" ]]; then
    echo "ERROR: S3_BUCKET is required when FETCH_MISSING_MEDIA is enabled"
    exit 1
  fi
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_ROOT="state/interpretation_parallel/${RUN_STAMP}"
BATCH_DIR="${TMP_ROOT}/batches"
OUT_DIR="${TMP_ROOT}/outputs"
LOG_DIR="${TMP_ROOT}/logs"
mkdir -p "$BATCH_DIR" "$OUT_DIR" "$LOG_DIR"

echo "[interpret-parallel] preparing batches"
set +e
"$_PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path
import math

cluster_csv = Path("$CLUSTER_CSV")
output_csv = Path("$OUTPUT_INTERPRETATION_CSV")
batch_dir = Path("$BATCH_DIR")
workers = int("$WORKERS")
resume = bool(int("$RESUME_BOOL"))

df = pd.read_csv(cluster_csv, low_memory=False)
if "video_id" not in df.columns:
    raise ValueError("cluster csv must include video_id")
if "cluster" not in df.columns:
    raise ValueError("cluster csv must include cluster")

df["video_id"] = df["video_id"].astype(str)
df = df.drop_duplicates(subset=["video_id"], keep="last")

if resume and output_csv.exists():
    done = pd.read_csv(output_csv, low_memory=False)
    if "video_id" in done.columns:
        done_ids = set(done["video_id"].astype(str).tolist())
        before = len(df)
        df = df[~df["video_id"].isin(done_ids)].copy()
        print(f"[interpret-parallel] resume: skipped {before - len(df)} rows already in output")

if df.empty:
    print("[interpret-parallel] no pending rows after resume filter")
    raise SystemExit(20)

total = len(df)
batch_size = int(math.ceil(total / workers))
parts = []
for i in range(workers):
    start = i * batch_size
    end = min(total, (i + 1) * batch_size)
    if start >= end:
        continue
    parts.append(df.iloc[start:end].copy())

for i, part in enumerate(parts, start=1):
    part.to_csv(batch_dir / f"batch_{i:03d}.csv", index=False)
print(f"[interpret-parallel] pending_rows={len(df)} batches={len(parts)}")
PY
split_rc=$?
set -e
if [[ $split_rc -eq 20 ]]; then
  echo "[interpret-parallel] nothing to do"
  exit 0
fi
if [[ $split_rc -ne 0 ]]; then
  echo "ERROR: failed while preparing interpretation batches"
  exit "$split_rc"
fi

run_batch() {
  local batch_csv="$1"
  local out_csv="$2"
  local log_file="$3"
  PYTHONUNBUFFERED=1 "$_PYTHON_BIN" Interpretation/build_interpretation.py \
    --cluster_csv "$batch_csv" \
    --metadata_csv "$METADATA_CSV" \
    --output_csv "$out_csv" \
    --s3_bucket "$S3_BUCKET" \
    --s3_region "$AWS_REGION" \
    --raw_prefix "$RAW_PREFIX" \
    "$FETCH_MISSING_FLAG" \
    --no-resume \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --fps_sample "$FPS_SAMPLE" \
    --diff_thresh "$DIFF_THRESH" \
    --edge_thresh "$EDGE_THRESH" \
    >"$log_file" 2>&1
}

echo "[interpret-parallel] launching workers"
declare -a pids=()
declare -a batch_names=()
for batch_csv in "$BATCH_DIR"/batch_*.csv; do
  batch_name="$(basename "$batch_csv" .csv)"
  out_csv="$OUT_DIR/${batch_name}_interpretation.csv"
  log_file="$LOG_DIR/${batch_name}.log"
  echo "[worker] start ${batch_name}"
  run_batch "$batch_csv" "$out_csv" "$log_file" &
  pids+=("$!")
  batch_names+=("$batch_name")
done

fail_count=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${batch_names[$i]}"
  set +e
  wait "$pid"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    echo "[worker] failed ${name} rc=${rc} log=${LOG_DIR}/${name}.log"
  else
    echo "[worker] done ${name}"
  fi
done

if [[ $fail_count -ne 0 ]]; then
  echo "ERROR: ${fail_count} worker batch(es) failed"
  exit 1
fi

echo "[interpret-parallel] merging worker outputs"
"$_PYTHON_BIN" - <<PY
import csv
import pandas as pd
from pathlib import Path

out_dir = Path("$OUT_DIR")
final_csv = Path("$OUTPUT_INTERPRETATION_CSV")
resume = bool(int("$RESUME_BOOL"))

frames = []
if resume and final_csv.exists():
    frames.append(pd.read_csv(final_csv, low_memory=False))
for p in sorted(out_dir.glob("*_interpretation.csv")):
    if p.exists() and p.stat().st_size > 0:
        frames.append(pd.read_csv(p, low_memory=False))

if not frames:
    raise SystemExit("no output frames to merge")

df = pd.concat(frames, ignore_index=True)
if "video_id" in df.columns:
    df["video_id"] = df["video_id"].astype(str)
    df = df.drop_duplicates(subset=["video_id"], keep="last")
if "cluster" in df.columns and "video_id" in df.columns:
    df = df.sort_values(["cluster", "video_id"]).reset_index(drop=True)

final_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(final_csv, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"[interpret-parallel] final_rows={len(df)} output={final_csv}")
PY

if [[ "$KEEP_TMP" == "1" || "$KEEP_TMP" == "true" || "$KEEP_TMP" == "yes" ]]; then
  echo "[interpret-parallel] kept temp dir: $TMP_ROOT"
else
  rm -rf "$TMP_ROOT"
fi

echo "[done] interpretation parallel run completed"
