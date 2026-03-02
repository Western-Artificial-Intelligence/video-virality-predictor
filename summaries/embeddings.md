# Embeddings Pipeline Summary

## 1. Scope and Objective
The embeddings stage converts raw artifacts into modality-specific dense vectors and stores them in S3 with deterministic keying by `video_id`.

Modalities:
1. Video
2. Audio
3. Text

Primary scripts:
- `Data/embeddings/video/embed_video_delta.py`
- `Data/embeddings/audio/embed_audio_delta.py`
- `Data/embeddings/text/embed_text_delta.py`

All three run as delta processors against `shorts_metadata_horizon.csv` and per-stage state DBs.

---

## 2. End-to-End Flow

```text
shorts_metadata_horizon.csv
  -> resolve latest row per video_id
  -> compute source_hash
  -> compare stage state (processed_items)
  -> for each delta item:
       download raw artifact from S3
       compute embedding
       save <video_id>.npy
       upload to S3 embeddings prefix
       upsert stage state
```

S3 naming contract:
- `clipfarm/embeddings/video/<video_id>.npy`
- `clipfarm/embeddings/audio/<video_id>.npy`
- `clipfarm/embeddings/text/<video_id>.npy`

---

## 3. Modality Details

## 3.1 Video Embeddings

| Item | Specification |
|---|---|
| Script | `Data/embeddings/video/embed_video_delta.py` |
| Model class | `transformers.VideoMAEModel` |
| Processor | `VideoMAEImageProcessor` |
| Default model | `MCG-NJU/videomae-base` |
| Raw input | `clipfarm/raw/video/<video_id>.mp4` |
| Output | `clipfarm/embeddings/video/<video_id>.npy` |

Preprocessing steps:
1. Decode frames using PyAV.
2. Uniformly sample `num_frames` (`default 16`) across video timeline.
3. If frame count is unknown, take first `N` decodable frames.
4. Pad by repeating final frame if fewer than `N`.
5. Normalize tensor layout and resize to model expected frame count and spatial size.

Embedding extraction:
- Use CLS token: `outputs.last_hidden_state[:, 0]`.
- L2 normalize vector.
- Save `float32` 1D `.npy`.

Vector dimension:
- Determined by model hidden size.
- With default VideoMAE base, expected `768`.

Determinism notes:
- Frame index selection is deterministic for fixed decode order.
- Inference has no explicit random seed but uses `eval()` mode.
- Hardware backend differences (CPU/CUDA/MPS) can induce minor numeric variation.

---

## 3.2 Audio Embeddings

| Item | Specification |
|---|---|
| Script | `Data/embeddings/audio/embed_audio_delta.py` |
| Model class | `transformers.Wav2Vec2Model` |
| Processor | `Wav2Vec2Processor` |
| Default model | `facebook/wav2vec2-base-960h` |
| Raw input | `clipfarm/raw/audio/<video_id>.wav` |
| Output | `clipfarm/embeddings/audio/<video_id>.npy` |

Preprocessing steps:
1. Load WAV with librosa at fixed sample rate (`default 16kHz`).
2. Trim decode to max duration (`default 90s`) for memory safety.
3. Run processor tokenization/padding.

Embedding extraction:
- Mean-pool `last_hidden_state` over time dimension.
- L2 normalize final vector.
- Save `float32` 1D `.npy`.

Vector dimension:
- Determined by model hidden size.
- With default wav2vec2-base, expected `768`.

Determinism notes:
- Same input waveform and same runtime generally yields stable outputs.
- Minor numeric drift may appear across hardware/library builds.

---

## 3.3 Text Embeddings

| Item | Specification |
|---|---|
| Script | `Data/embeddings/text/embed_text_delta.py` |
| Model class | `sentence_transformers.SentenceTransformer` |
| Default model | `all-MiniLM-L6-v2` |
| Raw input | `clipfarm/raw/text/<video_id>.json` + metadata row |
| Output | `clipfarm/embeddings/text/<video_id>.npy` |

Input text construction:
1. Build metadata text from `[title, description, query]` joined by `[SEP]`.
2. Load transcript from JSON field `transcript`.

Embedding extraction:
1. Encode metadata text (normalized embedding).
2. Encode transcript text (normalized embedding).
3. Concatenate `[meta_emb || transcript_emb]`.
4. L2 normalize concatenated vector.

Vector dimension:
- For MiniLM-L6-v2 each encode output is `384`.
- Concatenated vector is `768`.

Determinism notes:
- No stochastic training behavior; inference only.
- Drift can occur if model weights/version change or tokenizer behavior changes.

---

## 4. Delta, State, and Failure Handling

All embedding stages use `StageStateDB` (`Data/common/stage_state.py`).

State schema:

| Column | Meaning |
|---|---|
| `video_id` (PK) | Canonical item ID |
| `source_hash` | Metadata-derived source signature |
| `processed_at` | Last update time |
| `status` | `success`, `fail`, `missing_raw_object`, etc. |
| `error` | Error payload |
| `artifact_key` | S3 embedding key |
| `vector_id` | Reserved identifier |
| `retry_count` | Incremented by retry helper where used |

Delta inclusion rule:
- Include item if:
  - no prior state, or
  - source hash changed, or
  - previous status not terminal (`success` for embedding stages).

Failure behavior:
- Missing raw object -> `missing_raw_object`.
- Runtime exception -> `fail`.
- Item remains in future delta until `success` or upstream correction.

---

## 5. Linking Embeddings Back to `video_id`

Linking is achieved by three independent invariants:
1. File key includes `video_id`: `.../<video_id>.npy`.
2. State row keyed by same `video_id` stores `artifact_key`.
3. Fusion stage reconstructs vectors using manifest pointers and original `video_id`.

No surrogate IDs are generated in embedding stages; `video_id` is the sole join key.

---

## 6. Manifest Strategy and Parquet Structures

## 6.1 Implemented Manifest Strategy (Current Pipeline)
Current production delta embedding scripts do **not** emit a per-modality Parquet manifest. They rely on:
- deterministic S3 object key convention
- per-stage SQLite state as processing ledger

The first explicit Parquet manifest in the active pipeline is produced at fusion:
- `clipfarm/fused/<strategy>/fused_manifest.parquet`

This is intentional to avoid duplicating modality manifests when fusion already captures resolved pointers.

## 6.2 Legacy/Offline Parquet Artifacts (Repository)
The repository also contains non-delta batch artifacts/scripts:
- `Data/embeddings/video/videomae_embeddings.parquet`
- `Data/embeddings/text/text_embeddings.parquet`
- `Data/embeddings/text/final_text_inputs.parquet`

These are legacy/offline batch formats, not authoritative for current S3 delta operations.

## 6.3 Recommended Embedding Manifest Schema (If Needed)
If conference reporting requires per-modality manifests, use:

| Column | Type | Description |
|---|---|---|
| `video_id` | STRING | Canonical ID |
| `source_hash` | STRING | Input hash used for idempotency |
| `captured_at` | TIMESTAMP/STRING | Metadata capture time |
| `modality` | STRING | `video` / `audio` / `text` |
| `embedding_key` | STRING | S3 object pointer |
| `embedding_dim` | INT | Vector length |
| `model_name` | STRING | Exact model identifier |
| `model_version` | STRING | Frozen model revision/tag |
| `preprocess_version` | STRING | Preprocessing contract version |
| `status` | STRING | Success/failure state |
| `processed_at` | TIMESTAMP | Processing timestamp |

---

## 7. Determinism and Reproducibility Strategy

Implemented reproducibility controls:
1. Stable identity (`video_id`) and key naming.
2. Stable delta decisions (`source_hash` + terminal status logic).
3. Stage state persisted to S3 (`clipfarm/state/*_embedding.sqlite`).
4. Model name passed explicitly via CLI (`--model_name`).

Gaps (must be explicit in report):
1. No explicit pinning of model revision hash in code (model hub may update unless environment pins caches/versions).
2. No global inference seed for embedding scripts.
3. Backend/hardware differences may cause small float variation.

---

## 8. Embedding Drift Avoidance

Current anti-drift mechanisms:
- Keep model names fixed by configuration.
- Keep preprocessing parameters fixed (`num_frames`, `sample_rate`, `max_audio_seconds`, text construction template).
- Normalize vectors to unit norm before storage.

Required operational controls (assumption for reproducible deployment):
1. Pin exact package versions (`transformers`, `sentence-transformers`, `torch`, `librosa`).
2. Pin model artifact revisions (by SHA or local artifact cache).
3. Treat model/preprocess changes as versioned migration requiring re-embedding.

---

## 9. Batch vs Streaming Considerations

Current mode is micro-batch delta processing:
- Input is batch-loaded from current metadata CSV.
- Workset is reduced to unresolved delta.
- Each run processes finite items and exits.

Not true streaming:
- No event bus.
- No online feature store.
- No exactly-once stream processor.

Operationally this provides near-streaming behavior when scheduled frequently, while retaining simple replay semantics via stage state.

---

## 10. Assumptions, Constraints, Tradeoffs

Constraints:
1. S3 availability and object consistency determine stage throughput.
2. Model downloads and runtime memory constrain worker scalability.
3. Missing raw objects block embedding generation (`missing_raw_object`).

Tradeoffs:
1. `.npy` per item favors random access and idempotent overwrite, but increases object count.
2. Deferring manifest creation to fusion simplifies embedding stage but shifts pointer validation downstream.
3. L2 normalization improves cross-sample comparability but removes absolute magnitude information.
