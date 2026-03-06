# MP4-to-Virality Web Predictor

This app provides:

- `fast` mode: pinned `gbdt + concat` snapshot
- `full` mode: all four model families with per-model best fusion strategy and min-max prediction ranges

## Structure

- `backend/`: FastAPI inference API
- `frontend/`: React UI

## Local Dev

Backend:

```bash
cd virality-webapp/backend
python3.11 -m venv .venv-py311
. .venv-py311/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd virality-webapp/frontend
npm install
npm run dev
```

## Run With Script

From `virality-webapp/`:

```bash
export MODEL_S3_BUCKET=your-bucket
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
./scripts/run_local.sh
```

One-command launcher (setup + run):

```bash
cd /Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/virality-webapp
export MODEL_S3_BUCKET=your-bucket
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
./scripts/run_local.sh
```

Note:
- Use Python `3.11` (recommended) or `3.12` for backend venv.
- `run_local.sh` auto-prefers `python3.11` and creates `backend/.venv-py311`.

Useful flags:

```bash
./scripts/run_local.sh --setup-only
./scripts/run_local.sh --backend-only
./scripts/run_local.sh --frontend-only
./scripts/run_local.sh --skip-install
```
