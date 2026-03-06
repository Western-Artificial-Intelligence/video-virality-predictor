# Cluster + PCA Pipeline (Dev1/Dev2) Detailed Summary

## 1. Scope
This document describes the current `video_id`-native clustering pipeline implemented in:

- `Unsup_Cluster/cluster.py`
- `scripts/run_cluster_pipeline_local.sh` (cluster stage section)

It covers:

1. Data loading from S3 fused manifests/shards
2. Dimensionality reduction (PCA + optional UMAP)
3. Cluster candidate scoring and selection
4. Deterministic label canonicalization
5. Output artifacts and diagnostics
6. Runtime/memory behavior and safeguards

---

## 2. Input Contracts

## 2.1 Primary Inputs

Required:

- Metadata CSV (`--metadata_csv`, default `Data/raw/Metadata/shorts_metadata_horizon.csv`)
- S3 bucket (`--s3_bucket` or `S3_BUCKET`)
- Fused manifest + shard artifacts under:
  - `clipfarm/fused/concat/*`
  - `clipfarm/fused/sum_pool/*`
  - `clipfarm/fused/max_pool/*`

Optional runtime controls:

- `--strategies` (default: `concat,sum_pool,max_pool`)
- `--fusion_statuses` (default: `success_full,success_text_placeholder`)
- `--k_min`, `--k_max` (default: `6..16`)
- `--random_seeds` (default: `13,23,37,53,71`)
- `--enable_umap_cluster` / `--no-enable_umap_cluster`
- `--umap_cluster_dim` (default `15`)
- `--umap_viz_neighbors` (default `15`)
- `--min_cluster_fraction` (default `0.01`)
- `--max_silhouette_samples` (default `4000`; shell default currently `2000`)

## 2.2 Metadata Filter Semantics (`video_id` universe)

The metadata side is loaded through `load_latest_horizon_rows(...)` in `Data/common/horizon_delta.py`, which:

1. Resolves/normalizes `video_id`
2. Keeps latest row per `video_id` by parsed `captured_at`
3. Returns one canonical `video_id` row for downstream filtering

Clustering uses only the `video_id` set from these latest rows.

---

## 3. Strategy Data Loading (Per Fusion Strategy)

For each strategy (`concat`, `sum_pool`, `max_pool`):

1. Manifest key resolution:
   - `clipfarm/fused/<strategy>/fused_manifest.parquet`
2. Manifest validation: required columns must exist:
   - `video_id`, `source_hash`, `captured_at`, `fused_key`, `shard_idx`, `fusion_status`
3. Row filtering:
   - Keep only rows where `fusion_status` is in allowed statuses
   - Keep only rows with `video_id` in metadata latest set
4. Dedupe (latest manifest row per `video_id`):
   - Sort by `video_id,captured_at,source_hash,fused_key,shard_idx`
   - `drop_duplicates(video_id, keep="last")`
5. Vector reconstruction from NPZ shards:
   - Group manifest rows by `fused_key`
   - Download each shard once
   - Load `vectors` array from NPZ
   - Pick row `vectors[shard_idx]` for each manifest row
   - Preserve canonical manifest order via an internal `row_idx`

Notes:

- The loader now processes one shard at a time to reduce peak RAM.
- NPZ local temporary files are removed after read.
- Errors are raised for missing `vectors`, invalid `shard_idx`, or malformed array ranks.

---

## 4. Dimensionality Reduction (Dev1)

## 4.1 Cluster-space reduction

Given reconstructed matrix `X` (shape `n_samples x n_features`):

1. Standardize:
   - `X_scaled = StandardScaler().fit_transform(X)`
2. PCA for clustering:
   - `pca_dim = min(50, n_features, max(2, n_samples - 1))`
   - `pca_cluster = PCA(n_components=pca_dim).fit_transform(X_scaled)`
3. Optional UMAP cluster space:
   - Only when:
     - `enable_umap_cluster == True`
     - UMAP package is available
     - `n_samples >= 30`
     - PCA feature count `>= 4`
   - Parameters:
     - `n_neighbors = min(30, max(10, n_samples // 30))`
     - `n_components = min(max(2, umap_cluster_dim), pca_cluster_dim)`
     - `min_dist = 0.0`, metric `euclidean`
   - Result becomes cluster matrix and `cluster_space_name = "umap<n_components>"`

Fallback:

- If UMAP cluster-space criteria are not met, clustering uses PCA space (`cluster_space_name = "pca"`).

## 4.2 Visualization-space reduction

Always generated:

- `pca_viz2d = PCA(n_components=2).fit_transform(X_scaled)`

Optional visualization UMAP:

- If UMAP available and `n_samples >= 15`
- `n_neighbors = clamp(umap_viz_neighbors, 5, n_samples-1)`
- 2D UMAP on PCA cluster space (`min_dist=0.1`)

---

## 5. Candidate Clustering Evaluation (Dev2 Auto-selection)

For each strategy and each `k in [k_min..k_max]`:

1. Skip invalid `k`:
   - Must satisfy `2 <= k < n_samples`
2. For each seed in `random_seeds`:
   - Fit `KMeans(n_clusters=k, random_state=seed, n_init=20)`
   - Collect labels
   - Compute silhouette:
     - if one cluster only: `-1.0`
     - else:
       - sampled silhouette if `n_samples > max_silhouette_samples > 0`
       - full silhouette otherwise
3. Stability:
   - Mean pairwise ARI across seed labelings
4. Tiny-cluster penalty:
   - `min_cluster_size = max(2, ceil(min_cluster_fraction * n_samples))`
   - For each seed labeling, fraction of clusters smaller than `min_cluster_size`
   - penalty = mean of those fractions
5. Composite score:
   - `0.55 * silhouette + 0.35 * stability - 0.10 * tiny_penalty`

Tie-break order (descending):

1. `composite`
2. `silhouette`
3. `stability`
4. lower `tiny_cluster_penalty`
5. larger `k`

---

## 6. Memory/Execution Model

Current pipeline is strategy-by-strategy:

1. Load one strategy dataset
2. Reduce + evaluate candidates
3. Keep only current best reduced-space object
4. Release non-best strategy arrays (`gc.collect()`)

This avoids retaining all strategy matrices concurrently.

Additional memory control:

- `max_silhouette_samples` caps silhouette computation sample size.

---

## 7. Final Fit and Deterministic Labels

After selecting best `(strategy, k)`:

1. Final fit:
   - `KMeans(n_clusters=k, random_state=random_seeds[0], n_init=50)`
2. Canonical label remap:
   - Compute centroids per predicted label
   - Sort labels by rounded centroid tuple (`round(..., 12)`), then old label id
   - Remap to `0..k-1`

This provides deterministic `cluster` / `cluster_id` numbering for fixed inputs.

---

## 8. Output Artifacts

## 8.1 Main tabular outputs

`Unsup_Cluster/cluster_results.csv`:

- `video_id`
- `cluster`
- `cluster_id` (same numeric value for compatibility)
- `fusion_strategy`
- `k_selected`

Sorted by `cluster, video_id`.

`Unsup_Cluster/cluster_diagnostics.json` includes:

- selected strategy/k and candidate metrics
- per-strategy row counts and vector dimensions
- all evaluated candidates
- scoring config (`max_silhouette_samples`)
- artifact paths
- clustered row count

## 8.2 Latent-space artifacts (Dev1 outputs)

Under `Latence/latent_space_outputs/embeddings`:

- `cluster_pca20.npy` (first up-to-20 PCA cluster dims)
- `cluster_pca50_umap15.npy` (cluster matrix or PCA fallback)
- `cluster_reduction.npz` (`video_ids`, `embeddings`, `strategy`, `cluster_space`)
- `viz_pca2d.npz`
- optional `viz_umap2d.npz`

Under `Latence/latent_space_outputs/plots`:

- `pca_2d.png`
- `umap_2d.png` (UMAP plot or PCA fallback placeholder)

If matplotlib is unavailable, placeholder text files are written at expected plot paths.

---

## 9. CLI and Runner Defaults

Runner script: `scripts/run_cluster_pipeline_local.sh`

Current relevant defaults:

- `ENABLE_UMAP_CLUSTER=0`
- `MAX_SILHOUETTE_SAMPLES=2000`
- `K_MIN=6`, `K_MAX=16`
- `RANDOM_SEEDS=13,23,37,53,71`
- `MIN_CLUSTER_FRACTION=0.01`

Boolean env vars are translated to argparse flags:

- `1/true/yes/...` -> `--enable_umap_cluster` / `--fetch_missing`
- `0/false/no/...` -> `--no-enable_umap_cluster` / `--no-fetch_missing`

---

## 10. Failure and Guardrail Behavior

Hard errors (fail fast):

- Missing S3 bucket
- Missing manifest key per strategy
- Missing required manifest columns
- Empty candidate set after filtering
- Invalid k range
- Invalid seed list

Candidate-level skipping:

- Invalid `k` values (`k < 2` or `k >= n_samples`)
- Strategy with no valid candidates is skipped with log line

Runtime notes:

- `Killed: 9` from shell indicates OS termination (usually memory), not a Python exception.
- Stage logs now show progress:
  - loading strategy
  - loaded rows/dim
  - reduced space info
  - candidate count
  - current best update

