#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

RUN_BACKEND=1
RUN_FRONTEND=1
SETUP_ONLY=0
SKIP_INSTALL=0

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_local.sh [options]

Options:
  --setup-only      Install backend/frontend dependencies only, do not start servers
  --backend-only    Start backend only
  --frontend-only   Start frontend only
  --skip-install    Skip dependency installation checks/installs
  -h, --help        Show this help

Required environment (for backend model loading):
  MODEL_S3_BUCKET (or S3_BUCKET)
  AWS credentials (for S3 read access), e.g.:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION

Optional environment:
  MODEL_S3_REGION
  MODEL_SNAPSHOT_PREFIX
  VIRALITY_ASR_BACKEND
  VIRALITY_ASR_MODEL
  VIRALITY_MAX_UPLOAD_MB
  BACKEND_HOST, BACKEND_PORT
  FRONTEND_HOST, FRONTEND_PORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only)
      SETUP_ONLY=1
      shift
      ;;
    --backend-only)
      RUN_BACKEND=1
      RUN_FRONTEND=0
      shift
      ;;
    --frontend-only)
      RUN_BACKEND=0
      RUN_FRONTEND=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$RUN_BACKEND" == "1" ]]; then
  if [[ -z "${MODEL_S3_BUCKET:-}" && -z "${S3_BUCKET:-}" ]]; then
    echo "ERROR: set MODEL_S3_BUCKET (or S3_BUCKET) before running backend."
    exit 1
  fi

  # Fail fast on local machines without AWS auth instead of waiting on metadata endpoints.
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -z "${AWS_PROFILE:-}" && -z "${AWS_WEB_IDENTITY_TOKEN_FILE:-}" ]]; then
    echo "ERROR: AWS authentication is missing."
    echo "Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (and AWS_REGION) or AWS_PROFILE."
    exit 1
  fi

  export AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}"
fi

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "ERROR: npm not found"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found"; exit 1; }

pick_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
      echo "ERROR: PYTHON_BIN not found: $PYTHON_BIN"
      exit 1
    }
    echo "$PYTHON_BIN"
    return
  fi

  # Prefer Python 3.11/3.12 for sklearn model compatibility.
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done

  echo "ERROR: no compatible python found"
  exit 1
}

PYTHON_BIN="$(pick_python_bin)"
PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "[env] python_bin=${PYTHON_BIN} (python ${PYTHON_VERSION})"

if [[ "$PYTHON_VERSION" == "3.14" ]]; then
  echo "ERROR: Python 3.14 is not supported for these saved sklearn artifacts."
  echo "Install/use Python 3.11 or 3.12 (or set PYTHON_BIN explicitly)."
  exit 1
fi

BACKEND_VENV="${BACKEND_VENV_DIR:-$BACKEND_DIR/.venv-py311}"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  if [[ ! -d "$BACKEND_VENV" ]]; then
    echo "[setup] creating backend venv: $BACKEND_VENV"
    "$PYTHON_BIN" -m venv "$BACKEND_VENV"
  fi

  echo "[setup] installing backend requirements"
  "$BACKEND_VENV/bin/python" -m pip install -r "$BACKEND_DIR/requirements.txt"

  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "[setup] installing frontend dependencies"
    (cd "$FRONTEND_DIR" && npm install)
  else
    echo "[setup] frontend dependencies already present (node_modules)"
  fi
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
  echo "[done] setup complete"
  exit 0
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  set +e
  if [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$FRONTEND_PID" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$RUN_BACKEND" == "1" ]]; then
  echo "[run] starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
  (
    cd "$BACKEND_DIR"
    exec "$BACKEND_VENV/bin/uvicorn" app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
  ) &
  BACKEND_PID="$!"
fi

if [[ "$RUN_FRONTEND" == "1" ]]; then
  echo "[run] starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
  (
    cd "$FRONTEND_DIR"
    exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
  ) &
  FRONTEND_PID="$!"
fi

echo
echo "[ready]"
if [[ "$RUN_FRONTEND" == "1" ]]; then
  echo "  Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
fi
if [[ "$RUN_BACKEND" == "1" ]]; then
  echo "  Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}"
  echo "  Health:   http://${BACKEND_HOST}:${BACKEND_PORT}/healthz"
fi
echo
echo "Press Ctrl+C to stop."

if [[ "$RUN_BACKEND" == "1" && "$RUN_FRONTEND" == "1" ]]; then
  # Bash 3.2 on macOS does not support `wait -n`, so use a portable poll loop.
  while true; do
    if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
      wait "$BACKEND_PID" || true
      break
    fi
    if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
      wait "$FRONTEND_PID" || true
      break
    fi
    sleep 1
  done
elif [[ "$RUN_BACKEND" == "1" ]]; then
  wait "$BACKEND_PID"
else
  wait "$FRONTEND_PID"
fi
