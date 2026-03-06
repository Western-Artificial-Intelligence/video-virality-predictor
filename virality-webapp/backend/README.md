# Backend (FastAPI)

## Run

```bash
cd virality-webapp/backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Required Environment

- `MODEL_S3_BUCKET` (or `S3_BUCKET`): S3 bucket containing snapshot artifacts
- `MODEL_S3_REGION` (optional): region override
- `MODEL_SNAPSHOT_PREFIX` (optional, default `clipfarm/models/snapshots`)

## Optional Environment

- `VIRALITY_MODEL_CACHE_DIR` (default `state/virality_webapp/model_cache` under repo root)
- `VIRALITY_ASR_BACKEND` (default `auto`)
- `VIRALITY_ASR_MODEL` (default `small`)
- `VIRALITY_MAX_UPLOAD_MB` (default `512`)
- `VIRALITY_CORS_ALLOW_ORIGINS` (comma-separated, default `*`)
- `VIRALITY_WEBAPP_SKIP_STARTUP_LOAD=1` (skip model loading on startup)

## API

- `GET /api/schema`
- `POST /api/predict` (multipart form fields: `video_file`, `mode`, `metadata_json`)
