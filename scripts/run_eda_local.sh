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

METADATA_CSV="${METADATA_CSV:-Data/raw/Metadata/shorts_metadata_horizon.csv}"
CLUSTER_CSV="${CLUSTER_CSV:-Unsup_Cluster/cluster_results.csv}"
INTERPRETATION_CSV="${INTERPRETATION_CSV:-Interpretation/interpretation.csv}"
EDA_OUT_DIR="${EDA_OUT_DIR:-EDA/output}"
TOP_N_CATEGORIES="${TOP_N_CATEGORIES:-15}"

check_python_module() {
  local module="$1"
  "$_PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
}

missing_modules=()
for mod in pandas numpy; do
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

echo "[eda] running full EDA"
PYTHONUNBUFFERED=1 "$_PYTHON_BIN" scripts/run_full_eda.py \
  --metadata_csv "$METADATA_CSV" \
  --cluster_csv "$CLUSTER_CSV" \
  --interpretation_csv "$INTERPRETATION_CSV" \
  --out_dir "$EDA_OUT_DIR" \
  --top_n_categories "$TOP_N_CATEGORIES"

echo "[done] EDA complete"
