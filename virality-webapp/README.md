# MP4-to-Virality Web Predictor

This app provides:

- `fast` mode: pinned `gbdt + concat` snapshot
- `full` mode: all four model families with per-model best fusion strategy and robust padded prediction ranges

## Prediction Modes

Both modes run the same single-request online pipeline (`MP4 -> WAV -> ASR -> video/audio/text embeddings -> fusion -> metadata coercion -> inference`).  
The difference is which model artifacts are executed and how outputs are returned.

### `fast` mode

- Uses one pinned (best) snapshot for each horizon (`7d`, `30d`):
  - model family: `gbdt`
  - fusion strategy: `concat`
  - run_id: `colab-gbdt-20260306T070623Z`
- Response returns one prediction for `7d` and one for `30d`.
- Best for lower latency.

### `full` mode

- Runs all four model families for each horizon (`7d`, `30d`) with per-model best fusion:
  - `gbdt` + `concat` (`colab-gbdt-20260306T070623Z`)
  - `concat_mlp` + `max_pool` (`colab-concat_mlp-20260306T054620Z`)
  - `gated_fusion_mlp` + `concat` (`colab-gated_fusion_mlp-20260306T054937Z`)
  - `ridge` + `sum_pool` (`colab-ridge-20260306T070324Z`)
- Response returns per-model outputs for `7d` and `30d`, plus `range_7d` / `range_30d` computed with robust trimming and padding:
  - work in `prediction_log` space
  - sort model outputs and drop lowest/highest when 4+ models are present
  - keep core middle values and compute `spread = max(core_high - core_low, 0.08)`
  - apply padding `pad = 0.5 * spread + 0.10`
  - range is `[core_low - pad, core_high + pad]` (log), then converted to raw with `expm1` and floored at `0`
- Best for comparison and uncertainty range checks (at higher latency).

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
cd video-virality-predictor/virality-webapp
export MODEL_S3_BUCKET=your-bucket
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
./scripts/run_local.sh
```

Note:
- Use Python `3.11` (recommended) or `3.12` for backend venv.
- `run_local.sh` auto-prefers `python3.11` and creates `backend/.venv-py311`.
- First prediction will take longer for the model for embedding model to load

Useful flags:

```bash
./scripts/run_local.sh --setup-only
./scripts/run_local.sh --backend-only
./scripts/run_local.sh --frontend-only
./scripts/run_local.sh --skip-install
```
