# Colab Training Entry Points

This folder contains one notebook per model family:

- `train_concat_mlp.ipynb`
- `train_gated_fusion_mlp.ipynb`
- `train_ridge.ipynb`
- `train_gbdt.ipynb`
- `text_raw_pipeline_gpu.ipynb` (raw text collection on Colab GPU)
- `text_sqlite_monitor.ipynb` (live sqlite progress monitor, 60s refresh)
- `text_sqlite_monitor_s3.ipynb` (pull sqlite from S3 every 60s and monitor progress)

Each notebook trains **one model family** across the full matrix:

- Fusion strategies: `concat`, `sum_pool`, `max_pool`
- Horizons: `7`, `30`

So each notebook produces **6 training outputs** under one `run_id`.

## Required Inputs

- AWS credentials in Colab Secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- Optional Colab secret for private repos:
  - `GITHUB_TOKEN`
- S3 target:
  - `S3_BUCKET`
  - `AWS_REGION`
- Repo checkout available in Colab filesystem (default path in notebooks):
  - `/content/video-virality-predictor`

Each notebook bootstrap cell pulls secrets with:

`from google.colab import userdata`

`userdata.get("SECRET_NAME")`

and handles clone/update automatically.

## Output Layout

For model-specific Colab runs, snapshots are written to:

`clipfarm/models/snapshots/run_id=<run_id>/model=<model_family>/strategy=<fusion>/horizon=<days>/`

Run-level comparison output:

`clipfarm/models/snapshots/run_id=<run_id>/comparison/`

## Core Runner

Notebooks call:

`Super_Predict/run_model_colab_matrix.py`

That script invokes:

- `Super_Predict/train_suite_from_horizon.py` (6 times)
- `Super_Predict/aggregate_train_suite_results.py` (once)

## Text Raw Pipeline On Colab GPU

For transcript generation on Colab GPU, use:

- `scripts/run_text_pipeline_colab.py`

This runner:
- restores `text_downloader.sqlite` from S3,
- runs `Data/raw/Text/text_collect.py` with `faster_whisper` GPU defaults,
- checkpoints state to S3 during the run,
- persists final state to S3 even on failure.

### Minimal Colab Cell

```python
from google.colab import userdata
import os, subprocess, sys

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_REGION"] = userdata.get("AWS_REGION")
os.environ["S3_BUCKET"] = userdata.get("S3_BUCKET")

repo_dir = "/content/video-virality-predictor"
if not os.path.isdir(repo_dir):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/<org>/<repo>.git", repo_dir], check=True)

cmd = [
    sys.executable,
    f"{repo_dir}/scripts/run_text_pipeline_colab.py",
    "--s3_bucket", os.environ["S3_BUCKET"],
    "--s3_region", os.environ["AWS_REGION"],
    "--asr_backend", "faster_whisper",
    "--asr_model", "small",
    "--max_workers", "1",
    "--checkpoint_seconds", "60",
]
subprocess.run(cmd, check=True, cwd=repo_dir)
```

For T4/L4 runtimes, keep `max_workers=1`; GPU batching is handled internally.
