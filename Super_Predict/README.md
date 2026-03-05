# Prediction Stage (Training Suite)

This module trains and compares four model families for virality prediction using fused embeddings + engineered metadata.

## Entrypoints
- Training script (single strategy+horizon job): `Super_Predict/train_suite_from_horizon.py`
- Run comparison aggregator: `Super_Predict/aggregate_train_suite_results.py`
- Colab matrix runner (single model family): `Super_Predict/run_model_colab_matrix.py`
- Colab notebooks (one per model): `colab/*.ipynb`

## Scope
Each training job runs one pair:
- `fusion_strategy`: `concat`, `sum_pool`, or `max_pool`
- `target_horizon_days`: `7` or `30`

Per pair, the suite can train:
1. `concat_mlp`
2. `gated_fusion_mlp`
3. `ridge`
4. `gbdt` (on projected fused vectors)

Use `--model_family` to run one model only:
- `all` (default)
- `concat_mlp`
- `gated_fusion_mlp`
- `ridge`
- `gbdt`

## Target and Splits
- Target transform: `y = log1p(horizon_view_count)`
- Split: random `70/15/15` (`train/val/test`) with fixed seed
- Ranking metric default: `val_rmse_log`

## Leakage Policy
Training enforces leakage-safe inputs:
- Forbidden columns are excluded (post-horizon and post-publish signals), e.g.
  - `horizon_view_count`, `horizon_days`
  - `view_count`, `like_count`, `comment_count`
  - rate-derived leakage columns (`views_per_day`, `likes_per_view`, etc.)
  - `virality_score`
- A hard guard checks selected feature columns against leakage set.

## Metadata Handling
- Low-card categorical: one-hot for classical models, embedding tables for NN models.
- High-card IDs are excluded in v1:
  - `channel_id`, `channel_title`

## Required Inputs
- Metadata CSV:
  - `Data/raw/Metadata/shorts_metadata_horizon.csv`
- Strategy-specific fused manifest in S3:
  - `clipfarm/fused/<strategy>/fused_manifest.parquet`
- S3 shard pointers from manifest are used to reconstruct fused vectors for training.

## Snapshot Outputs (S3)
Per run and pair (`--model_family all`):
- Prefix:
  - `clipfarm/models/snapshots/run_id=<run_id>/strategy=<strategy>/horizon=<h>/`
- Artifacts:
  - `config_used.json`
  - `data_summary.json`
  - `split_manifest.parquet`
  - `metrics_summary.json`
  - `leaderboard.csv`
  - `models/concat_mlp.pt`
  - `models/gated_fusion_mlp.pt`
  - `models/ridge.joblib`
  - `models/gbdt.joblib`
  - `models/gbdt_projector.pt`
  - `curves/*.json`
  - `predictions/val.parquet`
  - `predictions/test.parquet`
  - `slice_metrics_text_present.json`

Per run and pair (model-specific Colab run):
- Prefix:
  - `clipfarm/models/snapshots/run_id=<run_id>/model=<model_family>/strategy=<strategy>/horizon=<h>/`
- Artifacts:
  - Same structure, but only the selected model's artifacts/metrics are emitted.

Run-level aggregate:
- Prefix:
  - `clipfarm/models/snapshots/run_id=<run_id>/comparison/`
- Files:
  - `metrics_comparison.json`
  - `metrics_comparison.csv`
  - `best_by_mae_log.json`

## Metrics
Per model, both `val` and `test` include:
- `rmse_log`, `mae_log`, `r2_log`
- `rmse_raw`, `mae_raw` (inverse transformed with `expm1`)

Slice metrics are emitted for:
- `text_present=1`
- `text_present=0`

## Colab Training (Recommended)
Use one notebook per model family:
- `colab/train_concat_mlp.ipynb`
- `colab/train_gated_fusion_mlp.ipynb`
- `colab/train_ridge.ipynb`
- `colab/train_gbdt.ipynb`

Each notebook runs:
- 3 fusion strategies × 2 horizons = 6 outputs
- one model family only (no overwrite across model families)

See `colab/README.md` for required env vars and S3 output paths.

## Reproducibility
`config_used.json` includes:
- seed
- run_id
- git SHA
- package versions
- feature set version
- model hyperparameters

The suite also sets deterministic seeds for Python, NumPy, and Torch.
