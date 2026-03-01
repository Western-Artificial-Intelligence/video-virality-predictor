# Colab Training Entry Points

This folder contains one notebook per model family:

- `train_concat_mlp.ipynb`
- `train_gated_fusion_mlp.ipynb`
- `train_ridge.ipynb`
- `train_gbdt.ipynb`

Each notebook trains **one model family** across the full matrix:

- Fusion strategies: `concat`, `sum_pool`, `max_pool`
- Horizons: `7`, `30`

So each notebook produces **6 training outputs** under one `run_id`.

## Required Inputs

- AWS credentials in Colab environment:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- S3 target:
  - `S3_BUCKET`
  - `AWS_REGION`
- Repo checkout available in Colab filesystem (default path in notebooks):
  - `/content/video-virality-predictor`

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
