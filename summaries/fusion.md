# Fusion Pipeline Summary

## 1. Scope and Objective
Fusion converts per-modality embeddings into one fused representation per `video_id` and emits:
1. Sharded fused vector files (`.npz`)
2. A fused manifest (`.parquet`) with pointer-based retrieval metadata
3. A schema lock file (`schema.json`)

Primary script:
- `Data/common/fuse_embeddings_delta.py`

Supported strategies:
- `concat`
- `sum_pool`
- `max_pool`

---

## 2. Core Data Contract

Required upstream artifacts in S3:
- `clipfarm/embeddings/video/<video_id>.npy`
- `clipfarm/embeddings/audio/<video_id>.npy`
- `clipfarm/embeddings/text/<video_id>.npy` (optional under terminal text-missing policy)
- text stage state snapshot:
  - `clipfarm/state/text_downloader.sqlite`

Fusion identity:
- keyed by `(video_id, source_hash)` from metadata horizon delta row.

---

## 3. Modality Existence Assumption and Intersection Logic

## 3.1 Assumption in Practice
Production contract assumes all three modalities should exist eventually, but treats text as conditionally optional in terminal failure cases.

Hard requirements:
- Video embedding must exist.
- Audio embedding must exist.

Text handling:
- If text embedding exists: full fusion.
- If text missing: allowed only when text-state status is terminal and approved (default: `no_captions`, `fail_empty_transcript_terminal`).
- Otherwise: defer item (`missing_text_pending`).

## 3.2 Why Intersection Filtering Is Required
Fusion vectors are used directly for supervised training. Missing required modalities without explicit policy causes:
- variable semantics per row
- hidden target leakage risk (missingness correlated with labels)
- unstable model behavior

Therefore:
- enforce intersection across required modalities (`video ∩ audio`)
- gate optional text by explicit status-backed policy
- append a `text_present` mask so downstream models can condition on text availability

## 3.3 Missing Modality Edge Cases

| Case | Status | Behavior |
|---|---|---|
| Missing video/audio embedding | `missing_required_modality` | Skip fusion, keep in delta |
| Missing text + text DB unavailable | `missing_text_state_unavailable` | Skip fusion |
| Missing text + no text state row | `missing_text_pending` | Skip fusion |
| Missing text + non-terminal text status | `missing_text_pending` | Skip fusion |
| Missing text + terminal text status | `success_text_placeholder` | Fuse with zero text vector |

---

## 4. Fusion Algorithm Details

Let:
- `v` = video vector
- `a` = audio vector
- `t` = text vector (real or zero placeholder)
- `m` = text presence mask (`1.0` or `0.0`)

## 4.1 Strategy A: No Projection (Implemented)

### `concat`
```text
base = [v || a || t]
fused = [base || m]   # when append mask is enabled (default true)
```

### `sum_pool`
```text
dim = max(len(v), len(a), len(t))
v',a',t' = zero-pad each to dim
base = v' + a' + t'
fused = [base || m]
```

### `max_pool`
```text
dim = max(len(v), len(a), len(t))
v',a',t' = zero-pad each to dim
base = elementwise_max(v', a', t')
fused = [base || m]
```

Why no learned projection in current production:
1. Fully deterministic under fixed inputs.
2. No extra training dependency in fusion stage.
3. Stable vector semantics across daily runs.

### Output fused vector shape
Let:
- `Dv = len(video_vec)`
- `Da = len(audio_vec)`
- `Dt = len(text_vec)`
- `M = 1` if text mask appended else `0`

Then:
- `concat`: `fused_dim = Dv + Da + Dt + M`
- `sum_pool`: `fused_dim = max(Dv, Da, Dt) + M`
- `max_pool`: `fused_dim = max(Dv, Da, Dt) + M`

With default base models (`Dv=Da=Dt=768`) and mask enabled:
- `concat`: `2305`
- `sum_pool`: `769`
- `max_pool`: `769`

---

## 5. Projection Strategy Analysis (Architectural Options)

The repository currently uses non-learned fusion strategies. For conference reporting, projection options should be understood explicitly.

## 5.1 Option 1: No Projection (Current)
Pros:
- Deterministic and simple.
- No train-time artifact dependency in fusion job.
- Easy reproducibility.

Cons:
- High dimensionality (especially concat).
- No learned cross-modal alignment at fusion time.

## 5.2 Option 2: Deterministic Projection (Static Matrix)
Pattern:
- Apply fixed, versioned projection matrix `P` (e.g., PCA components or random Gaussian matrix with fixed seed).
- Persist `P` artifact with version tag.

Requirements for correctness:
1. `P` must be immutable after release.
2. Matrix hash/version must be stored in schema and manifest.
3. All reruns for same release use identical `P`.

## 5.3 Option 3: Trained MLP Projection
Pattern:
- Train projector offline (supervised/self-supervised).
- Freeze weights.
- Use frozen checkpoint during daily fusion.

Requirements:
1. Projector checkpoint path must be explicit versioned input.
2. No online re-training inside daily fusion loop.
3. Manifest/schema include projector version ID.

---

## 6. Why Random Reinitialization Breaks Stability

If projection weights are randomly reinitialized each day:
1. Same `(video_id, source_hash)` maps to different fused coordinates.
2. Day-over-day vector drift is artificial, not content-driven.
3. Historical model checkpoints become incompatible with newly fused data.
4. Metrics degrade due to representation space shift.

Conclusion:
- Projection artifacts must be frozen and versioned.
- Fusion should be deterministic given same upstream embeddings + same strategy/config.

---

## 7. Embedding Stability Guarantees Across Days

Implemented stability controls:
1. Deterministic fusion operators (`concat/sum/max` + optional mask append).
2. Schema lock (`schema.json`) enforced:
   - `video_dim`, `audio_dim`, `text_dim`, `fused_dim`, `mask_appended`, `fusion_strategy`
3. If incoming dimensions mismatch existing schema, fusion run raises error.
4. Manifest upsert by `(video_id, source_hash)` keeps latest pointer deterministically.

This means day-over-day changes reflect either:
- upstream embedding change
- source hash change
- explicit strategy/config/schema change

---

## 8. Outputs and Storage Schemas

## 8.1 Sharded Fused Vectors (`.npz`)
Location:
- `clipfarm/fused/<strategy>/shards/date=YYYY-MM-DD/part-XXXX.npz`

Per shard payload:
- `video_ids`: array[str]
- `vectors`: array[float32] shape `(N, fused_dim)`
- `captured_at`: array[str]

## 8.2 Manifest (`fused_manifest.parquet`)
Location:
- `clipfarm/fused/<strategy>/fused_manifest.parquet`

Core columns:

| Column | Meaning |
|---|---|
| `video_id` | Canonical ID |
| `captured_at` | Metadata capture time |
| `source_hash` | Input hash for idempotent upsert |
| `fusion_strategy` | `concat`/`sum_pool`/`max_pool` |
| `fused_dim` | Vector dimension |
| `fused_key` | S3 key of NPZ shard |
| `shard_idx` | Row index in shard |
| `shard_n` | Rows in shard |
| `video_emb_key`, `audio_emb_key`, `text_emb_key` | Source embedding pointers |
| `text_present` | 1/0 |
| `text_source` | `full` or `placeholder` |
| `text_state_status` | Text stage status snapshot |
| `text_missing_reason` | Terminal reason if placeholder |
| `fusion_status` | `success_full` or `success_text_placeholder` |
| `retry_count_at_write` | Retry count snapshot |

Manifest merge behavior:
- concatenate old+new
- sort by `(video_id, source_hash, captured_at, fused_key)`
- drop duplicates on `(video_id, source_hash)`, keep latest

## 8.3 Schema Lock (`schema.json`)
Location:
- `clipfarm/fused/<strategy>/schema.json`

Fields:
- `fusion_strategy`
- `video_dim`
- `audio_dim`
- `text_dim`
- `fused_dim`
- `mask_appended`
- `updated_at`

---

## 9. State, Retries, and Failure Handling

State DB (per strategy):
- `state/fusion_<strategy>.sqlite`
- mirrored to `clipfarm/state/fusion_<strategy>.sqlite`

Terminal fusion statuses:
- `success_full`
- `success_text_placeholder`
- `fail_terminal`

Retry policy:
- Non-terminal failures call `upsert_with_retry(...)`.
- `retry_count` increments for same `source_hash`.
- status becomes `fail_terminal` after `max_fail_retries` (default `3`).

Sample status accounting is emitted per run for observability.

---

## 10. Determinism and Reproducibility

Deterministic elements:
1. Input set defined by metadata+state delta logic.
2. Fixed fusion operations.
3. Stable shard row pointer (`fused_key`,`shard_idx`) recorded in manifest.
4. Schema lock prevents silent shape drift.

Non-deterministic/external factors:
1. Upstream embedding recomputation differences.
2. S3 object availability timing.
3. Run date partition assignment if `--run_date` not fixed (`UTC today` default).

For strict reproducibility:
- pin `run_date`
- keep upstream embedding artifacts immutable
- retain manifest and schema snapshots

---

## 11. Assumptions, Constraints, Tradeoffs

Constraints:
1. Fusion cannot proceed without text state DB when text embedding is missing.
2. Schema lock rejects dimension changes unless migration is intentional.

Tradeoffs:
1. Placeholder text allows throughput but introduces modality incompleteness.
2. Concat preserves information but increases dimension and model cost.
3. Sum/max pooling reduce size but may lose modality-specific detail.
