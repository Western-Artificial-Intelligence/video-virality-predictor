# Training Pipeline Summary

## 1. Scope and Objective
The training stage builds supervised virality predictors from:
1. horizon-labeled metadata (`shorts_metadata_horizon.csv`)
2. fused embedding pointers (`fused_manifest.parquet`)

Primary script:
- `Super_Predict/train_suite_from_horizon.py`

It trains a model suite for each `(fusion_strategy, target_horizon_days)` pair and writes snapshot artifacts to S3.

---

## 2. Dataset Construction

## 2.1 Input Sources
- Metadata CSV: `Data/raw/Metadata/shorts_metadata_horizon.csv`
- Fused manifest key (example): `clipfarm/fused/<strategy>/fused_manifest.parquet`
- Fused shard files referenced by manifest (`.npz`)
- Original modality embedding files referenced by manifest (for gated models)

## 2.2 Row Assembly Flow
1. Filter metadata to one horizon (`horizon_days == 7` or `30`).
2. Sort by `captured_at`, dedupe by `video_id` keep latest.
3. Build target:
   - `target_raw = horizon_view_count`
   - `target_log = log1p(target_raw)`
4. Load fused manifest, filter by fusion strategy.
5. Resolve latest fused row per `video_id`.
6. Reconstruct fused vectors from `fused_key` + `shard_idx`.
7. Load modality vectors (`video_emb_key`, `audio_emb_key`, `text_emb_key`) for gated model path.
8. Inner join fused rows with metadata by `video_id`.

---

## 3. Label Sourcing

Label source:
- `horizon_view_count` from metadata horizon system.

Target transform:
- `y = log1p(horizon_view_count)` (used for training/optimization).
- Raw-scale metrics are computed via inverse transform (`expm1`) for interpretability.

Rationale:
- Log transform compresses heavy-tailed view count distribution.
- Improves optimization stability and metric comparability.

---

## 4. Feature Construction and Leakage Controls

## 4.1 Metadata Features
Derived metadata includes numeric and low-cardinality categorical fields.

Low-card categorical candidates:
- `channel_country`
- `default_language`
- `default_audio_language`
- `query`

High-cardinality exclusions:
- `channel_id`
- `channel_title`

## 4.2 Leakage Prevention (Implemented)
A hard leakage set is excluded from features:
- horizon labels (`horizon_view_count`, `horizon_days`, `horizon_label_type`)
- post-outcome counters (`view_count`, `like_count`, `comment_count`)
- rate-derived columns (`views_per_day`, etc.)
- `age_days`
- `virality_score`

`assert_no_feature_leakage(...)` fails training if any leakage columns survive selection.

## 4.3 Temporal Leakage Prevention (Current State)
Current implementation uses random split with fixed seed (`70/15/15`) and does **not** enforce chronological holdout by `captured_at`.

What is prevented:
- direct feature leakage from post-horizon columns
- duplicate `video_id` rows via dedupe

What is not strictly prevented:
- potential temporal leakage from random split across nearby capture times.

Conference-report assumption to state explicitly:
- This pipeline is leakage-safe on features, but not strict time-series split-safe unless split logic is changed to time-based partitioning.

---

## 5. Train/Validation/Test Splits

Method:
- shuffle indices with deterministic RNG seed (`default 42`)
- allocate counts by configured ratios:
  - train: `0.70`
  - val: `0.15`
  - test: `0.15`

Constraints:
- Minimum total rows required: `>= 12`.
- Each split must have at least one sample.

Split manifest artifact:
- `split_manifest.parquet`
- includes `video_id`, `split`, `text_present`, `target_log`, `target_raw`.

---

## 6. Model Architectures

## 6.1 `concat_mlp`
Inputs:
- strategy-specific fused vector (`concat` / `sum_pool` / `max_pool`)
- standardized numeric metadata
- embedded categorical metadata

Architecture:
- categorical embedding tables per feature (`embedding_dim ~= sqrt(cardinality)`, capped in code)
- input assembly: `[fused || numeric || categorical_embeddings]`
- MLP head: `[1024, 512, 256]` with `GELU + Dropout(0.20)`
- output: scalar regression prediction in log space (`log1p(horizon_view_count)` target)

Notes:
- This model is directly affected by fusion strategy because fused input changes by strategy.

## 6.2 `gated_fusion_mlp`
Inputs:
- separate video/audio/text vectors
- numeric metadata
- categorical metadata
- `text_present` mask

Architecture:
- three modality towers (`video`, `audio`, `text`): `Linear -> GELU -> LayerNorm` (tower dim `256`)
- gate network consumes metadata context (`numeric + categorical_embeddings + text_present`) and predicts 3-way weights
- gate weights are softmax-normalized and applied to tower outputs
- weighted modality sum is passed to prediction head with metadata + mask context
- prediction head hidden layers: `[256, 128]` with dropout (`0.15`)

This architecture explicitly models modality reliability and text missingness.

Notes:
- It does not consume the fused vector directly; it uses raw modality embeddings.
- In current pipeline, strategy mostly affects which rows are selected from manifest, not the core gated feature path.

## 6.3 `ridge`
Pipeline:
- numeric: median impute + standard scale
- categorical: most-frequent impute + one-hot
- append fused vector dimensions
- `RidgeCV` over alpha grid (`0.1, 0.3, 1, 3, 10, 30, 100`)

Notes:
- This model is directly affected by fusion strategy via fused features.

## 6.4 `gbdt` (Projected Fused)
Two-stage:
1. Train neural projector (`ProjectorRegressor`) from fused vector to low-dim latent (`projector_dim`, default `128`).
2. Train `HistGradientBoostingRegressor` on projected latent + metadata features.

This hybrid keeps tree model expressive while reducing fused input dimensionality.

Projector details:
- projector core: `Linear(fused_dim -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> projector_dim)`
- auxiliary regression head supervises projector training during stage 1

GBDT details:
- `learning_rate=0.05`
- `max_depth=6`
- `max_iter=500`
- `min_samples_leaf=20`
- `l2_regularization=1e-4`
- `early_stopping=True`

Notes:
- This model is directly affected by fusion strategy because projector input is strategy-specific fused vector.

## 6.5 Fusion-Strategy Applicability Summary

| Model | Directly strategy-sensitive? | Why |
|---|---|---|
| `concat_mlp` | Yes | Consumes strategy-specific fused vector directly |
| `ridge` | Yes | Consumes strategy-specific fused features directly |
| `gbdt` | Yes | Projector input is strategy-specific fused vector |
| `gated_fusion_mlp` | Not directly | Uses modality vectors (`video/audio/text`) rather than fused vector |

---

## 7. Loss, Optimization, and Training Strategy

Neural models (`concat_mlp`, `gated_fusion_mlp`, projector):
- Loss: MSE on `target_log`
- Optimizer: AdamW
- Early stopping on validation RMSE
- Max epochs: configurable (`default 40`; projector capped lower)
- Patience: configurable (`default 6`)

Classical models:
- RidgeCV chooses regularization via internal CV.
- HistGradientBoosting uses built-in early stopping.

---

## 8. Evaluation Metrics

Per model, both `val` and `test`:
- `rmse_log`
- `mae_log`
- `r2_log`
- `rmse_raw` (inverse-transformed scale)
- `mae_raw`

Slice metrics:
- `text_present_1`
- `text_present_0`

Ranking:
- configurable rank metric (`--rank_metric`, default `rmse_log`, internally normalized to `val_<metric>`).

Leaderboard artifact:
- `leaderboard.csv` sorted by selected ranking metric.

---

## 9. Checkpointing and Artifact Outputs

Snapshot root pattern:
- all-model run:
  - `clipfarm/models/snapshots/run_id=<run_id>/strategy=<strategy>/horizon=<h>/`
- single-model run:
  - `clipfarm/models/snapshots/run_id=<run_id>/model=<family>/strategy=<strategy>/horizon=<h>/`

Core artifacts:

| Artifact | Description |
|---|---|
| `config_used.json` | Full run config, versions, feature sets, hyperparams |
| `data_summary.json` | Dataset size, split counts, missingness, target stats |
| `metrics_summary.json` | Model metrics and best-model selection |
| `slice_metrics_text_present.json` | Metrics split by text availability |
| `split_manifest.parquet` | Train/val/test assignment |
| `leaderboard.csv` | Ranked model table |
| `models/*.pt` | Torch checkpoints (`concat_mlp`, `gated_fusion_mlp`, `gbdt_projector`) |
| `models/*.joblib` | Sklearn artifacts (`ridge`, `gbdt`) |
| `curves/*.json` | Training curves |
| `predictions/val.parquet` | Validation predictions |
| `predictions/test.parquet` | Test predictions |

---

## 10. Failure Handling and Edge Cases

Critical fail-fast checks in training script:
1. Missing fused manifest in S3 -> `FileNotFoundError`.
2. Empty fused manifest after strategy filter -> hard failure.
3. No rows for selected `horizon_days` -> hard failure.
4. Empty join between metadata and fused rows -> hard failure.
5. Invalid split ratios or insufficient rows (`<12`) -> hard failure.
6. Leakage column accidentally selected -> hard failure.
7. No model metrics produced for requested model family -> hard failure.

Edge-case handling:
- Missing modality vectors referenced by manifest rows are skipped during reconstruction.
- Missing text embedding for `text_present=1` row is coerced to `text_present=0` with zero text vector.
- Text vectors are padded/truncated to inferred text dimension for consistency.

Operational implication:
- Training prefers explicit termination over silent degradation for structural errors.

---

## 11. Versioning and Reproducibility

Controls implemented:
1. Global seeds for Python, NumPy, Torch.
2. CUDNN deterministic mode enabled; benchmark disabled.
3. Run identifier (`run_id`) baked into output path.
4. `git_sha` stored in `config_used.json`.
5. Package versions captured (`numpy`, `pandas`, `scikit-learn`, `torch`, Python).
6. Explicit list of numeric/categorical/dropped columns persisted.

Residual nondeterminism:
- GPU kernel-level variation may still occur depending on environment.
- Random split means sample composition changes if seed changes.

---

## 12. Rolling Daily Updates and Retraining Interaction

Upstream daily changes:
- New metadata rows
- Updated fused manifest pointers

Retraining behavior:
1. Training run reads latest metadata + latest manifest rows.
2. Reconstructs vectors at run time from current pointers.
3. Produces new snapshot version under a new `run_id`.

Implication:
- Model artifacts are immutable per run.
- New daily data does not mutate old snapshots; it creates new snapshot lineage.

Operational recommendation:
- Use regular cadence retraining with explicit run IDs.
- Compare runs via `aggregate_train_suite_results.py` outputs in:
  - `clipfarm/models/snapshots/run_id=<run_id>/comparison/*`

---

## 13. Deployment Assumptions (Abstracted)

This repository does not include an online serving stack in this stage. Deployment assumptions for conference-level architecture:
1. Inference service must load:
   - chosen model artifact
   - matching preprocessing bundle
   - matching fusion strategy/schema
2. Online feature generation must mirror training transformations exactly.
3. Model registry should bind:
   - `model_version` -> `run_id`, `fusion_strategy`, `horizon_days`, `feature_set_version`.

Without this binding, train/serve skew is likely.

---

## 14. Constraints and Tradeoffs

Constraints:
1. Training depends on S3 availability for manifest and shard resolution.
2. Dataset quality depends on upstream fusion completeness.
3. Random splitting does not provide strict temporal generalization guarantees.

Tradeoffs:
1. Multi-model suite increases compute cost but reduces model-selection risk.
2. Log-scale optimization stabilizes training but requires careful raw-scale interpretation.
3. Gated model is more expressive but operationally heavier than linear/tree baselines.
