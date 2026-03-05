# Fusion Stage (Delta, S3-Backed)

This stage builds fused vectors from modality embeddings stored in S3.

## Entrypoint
- Script: `Data/common/fuse_embeddings_delta.py`
- Workflow: `.github/workflows/fusion-delta.yml`

## What It Requires
- Metadata CSV: `Data/raw/Metadata/shorts_metadata_horizon.csv`
- S3 embeddings:
  - `clipfarm/embeddings/video/<video_id>.npy`
  - `clipfarm/embeddings/audio/<video_id>.npy`
  - `clipfarm/embeddings/text/<video_id>.npy` (optional in terminal text-missing cases)
- Text stage state DB in S3:
  - `clipfarm/state/text_downloader.sqlite`

## Fusion Rules
- Required modalities: `video` and `audio`
- Text handling:
  - If text embedding exists: use full fusion
  - If missing, allow placeholder only when text status is terminal:
    - `no_captions`
    - `fail_empty_transcript_terminal`
  - Otherwise skip as pending
- Placeholder behavior:
  - Uses zero text vector (`--text_dim`, default `768`)
  - Appends `text_present` mask (1.0 or 0.0) when `--append_text_presence_mask` is enabled (default true)

## Supported Strategies
- `concat`
  - `fused = [video || audio || text || mask]`
- `sum_pool`
  - pad to max modality dim, sum elementwise, append mask
- `max_pool`
  - pad to max modality dim, max elementwise, append mask

## S3 Outputs
Per strategy (`<strategy>` in `concat|sum_pool|max_pool`):
- Shards:
  - `clipfarm/fused/<strategy>/shards/date=YYYY-MM-DD/part-XXXX.npz`
- Manifest:
  - `clipfarm/fused/<strategy>/fused_manifest.parquet`
- Schema:
  - `clipfarm/fused/<strategy>/schema.json`
- State:
  - `clipfarm/state/fusion_<strategy>.sqlite`

## Manifest Columns (core)
- `video_id`
- `captured_at`
- `source_hash`
- `fusion_strategy`
- `fused_dim`
- `fused_key`
- `shard_idx`
- `shard_n`
- `video_emb_key`, `audio_emb_key`, `text_emb_key`
- `text_present`, `text_source`
- `text_state_status`, `text_missing_reason`
- `fusion_status` (`success_full` or `success_text_placeholder`)

## Idempotency and Retry
- Uses stage DB to process only delta rows.
- Re-runs do not duplicate manifest rows for same `(video_id, source_hash)`; latest row is upserted.
- Failure retries are tracked with `retry_count`.
- Rows can terminalize to `fail_terminal` after max retries.

## Quick Local Run
```bash
python Data/common/fuse_embeddings_delta.py \
  --metadata_csv Data/raw/Metadata/shorts_metadata_horizon.csv \
  --s3_bucket "$S3_BUCKET" \
  --s3_region "$AWS_REGION" \
  --fusion_strategy concat \
  --fused_prefix_base clipfarm/fused \
  --emb_prefix clipfarm/embeddings \
  --state_db state/fusion_concat.sqlite \
  --state_s3_key clipfarm/state/fusion_concat.sqlite
```

## Quick GitHub Run
- Open Actions -> `Fusion Delta` -> `Run workflow`
- It runs all three strategies in matrix.
