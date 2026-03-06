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
FUSED_PREFIX_BASE="${FUSED_PREFIX_BASE:-clipfarm/fused}"
RAW_PREFIX="${RAW_PREFIX:-clipfarm/raw}"

STRATEGIES="${STRATEGIES:-concat,sum_pool,max_pool}"
FUSION_STATUSES="${FUSION_STATUSES:-success_full,success_text_placeholder}"
STRATEGY_SELECTION="${STRATEGY_SELECTION:-auto}"
K_MIN="${K_MIN:-6}"
K_MAX="${K_MAX:-16}"
RANDOM_SEEDS="${RANDOM_SEEDS:-13,23,37,53,71}"
MIN_CLUSTER_FRACTION="${MIN_CLUSTER_FRACTION:-0.01}"
MAX_SILHOUETTE_SAMPLES="${MAX_SILHOUETTE_SAMPLES:-2000}"
ENABLE_UMAP_CLUSTER="${ENABLE_UMAP_CLUSTER:-0}"
UMAP_CLUSTER_DIM="${UMAP_CLUSTER_DIM:-15}"
UMAP_VIZ_NEIGHBORS="${UMAP_VIZ_NEIGHBORS:-15}"

FETCH_MISSING_MEDIA="${FETCH_MISSING_MEDIA:-1}"
FPS_SAMPLE="${FPS_SAMPLE:-2}"
DIFF_THRESH="${DIFF_THRESH:-25}"
EDGE_THRESH="${EDGE_THRESH:-20}"

OUTPUT_CLUSTER_CSV="${OUTPUT_CLUSTER_CSV:-Unsup_Cluster/cluster_results.csv}"
OUTPUT_DIAGNOSTICS_JSON="${OUTPUT_DIAGNOSTICS_JSON:-Unsup_Cluster/cluster_diagnostics.json}"
OUTPUT_CLUSTER_LINKS_CSV="${OUTPUT_CLUSTER_LINKS_CSV:-Interpretation/cluster_links.csv}"
OUTPUT_INTERPRETATION_CSV="${OUTPUT_INTERPRETATION_CSV:-Interpretation/interpretation.csv}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-Latence/latent_space_outputs/embeddings}"
PLOTS_DIR="${PLOTS_DIR:-Latence/latent_space_outputs/plots}"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in boto3 numpy pandas sklearn matplotlib av; do
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

UMAP_CLUSTER_FLAG="$(to_bool_flag "$ENABLE_UMAP_CLUSTER" "--enable_umap_cluster" "--no-enable_umap_cluster")"
if [[ -z "$UMAP_CLUSTER_FLAG" ]]; then
  echo "ERROR: ENABLE_UMAP_CLUSTER must be boolean-like (1/0/true/false/yes/no), got: $ENABLE_UMAP_CLUSTER"
  exit 1
fi

FETCH_MISSING_FLAG="$(to_bool_flag "$FETCH_MISSING_MEDIA" "--fetch_missing" "--no-fetch_missing")"
if [[ -z "$FETCH_MISSING_FLAG" ]]; then
  echo "ERROR: FETCH_MISSING_MEDIA must be boolean-like (1/0/true/false/yes/no), got: $FETCH_MISSING_MEDIA"
  exit 1
fi

echo "[cluster] starting"
PYTHONUNBUFFERED=1 "$_PYTHON_BIN" Unsup_Cluster/cluster.py \
  --metadata_csv "$METADATA_CSV" \
  --s3_bucket "$S3_BUCKET" \
  --s3_region "$AWS_REGION" \
  --fused_prefix_base "$FUSED_PREFIX_BASE" \
  --strategies "$STRATEGIES" \
  --fusion_statuses "$FUSION_STATUSES" \
  --strategy_selection "$STRATEGY_SELECTION" \
  --k_min "$K_MIN" \
  --k_max "$K_MAX" \
  --random_seeds "$RANDOM_SEEDS" \
  --min_cluster_fraction "$MIN_CLUSTER_FRACTION" \
  --max_silhouette_samples "$MAX_SILHOUETTE_SAMPLES" \
  "$UMAP_CLUSTER_FLAG" \
  --umap_cluster_dim "$UMAP_CLUSTER_DIM" \
  --umap_viz_neighbors "$UMAP_VIZ_NEIGHBORS" \
  --output_csv "$OUTPUT_CLUSTER_CSV" \
  --diagnostics_json "$OUTPUT_DIAGNOSTICS_JSON" \
  --embeddings_dir "$EMBEDDINGS_DIR" \
  --plots_dir "$PLOTS_DIR"

echo "[cluster-links] joining metadata URLs"
PYTHONUNBUFFERED=1 "$_PYTHON_BIN" Interpretation/combine_cluster_and_links.py \
  --clusters_csv "$OUTPUT_CLUSTER_CSV" \
  --metadata_csv "$METADATA_CSV" \
  --out_csv "$OUTPUT_CLUSTER_LINKS_CSV"

echo "[interpretation] building interpretation CSV"
PYTHONUNBUFFERED=1 "$_PYTHON_BIN" Interpretation/build_interpretation.py \
  --cluster_csv "$OUTPUT_CLUSTER_CSV" \
  --metadata_csv "$METADATA_CSV" \
  --output_csv "$OUTPUT_INTERPRETATION_CSV" \
  --s3_bucket "$S3_BUCKET" \
  --s3_region "$AWS_REGION" \
  --raw_prefix "$RAW_PREFIX" \
  "$FETCH_MISSING_FLAG" \
  --fps_sample "$FPS_SAMPLE" \
  --diff_thresh "$DIFF_THRESH" \
  --edge_thresh "$EDGE_THRESH"

echo "[done] cluster pipeline completed"
