# Data Collection Pipeline Summary

## 1. Scope and Objective
This stage constructs the canonical daily dataset used by all downstream ML stages:

1. Discover and label YouTube Shorts metadata at fixed horizons (`7d`, `30d`).
2. Materialize raw media artifacts for only changed items.
3. Derive audio.
4. Derive transcript text.

The mandatory orchestration order is:

```text
metadata horizon CSV
   -> video download
      -> audio extraction
         -> text transcription
```

This ordering is required because each downstream stage consumes the `video_id` universe and change signal generated in metadata.

---

## 2. Metadata Horizon System

### 2.1 What It Is
The metadata horizon system writes rows to `Data/raw/Metadata/shorts_metadata_horizon.csv` where labels are measured at a fixed post-publish age (`horizon_days`), rather than at arbitrary capture times.

Why this exists:
- Raw `view_count` alone is temporally inconsistent.
- Fixed-horizon labels reduce target noise for supervised learning.

Primary collector:
- `Data/raw/Metadata/daily_horizon_collector.py`

Primary output artifacts:
- `Data/raw/Metadata/shorts_metadata_horizon.csv` (append-only)
- `Data/raw/Metadata/horizon_labels.sqlite` (dedupe on `(video_id, horizon_days)`)

### 2.2 Discovery and Labeling Flow
Per run:
1. Build publish windows centered at `now - horizon_days` with tolerance (`default +/-0.5 days`).
2. Discover candidates through `search.list(videoDuration=short)` over broad seed queries.
3. Dedupe discovered `video_id`s.
4. Fetch full video metadata (`videos.list`) and channel metadata (`channels.list`).
5. Build a normalized row using `collect_metadata.build_row(...)`.
6. For each horizon window match, set:
   - `horizon_days` (`7` or `30`)
   - `horizon_view_count` = current `view_count` at labeling time
   - `horizon_label_type` = `views_at_age`
7. Append rows to `shorts_metadata_horizon.csv`.
8. Insert `(video_id, horizon_days)` into `horizon_labels.sqlite` with `INSERT OR IGNORE`.

### 2.3 “Yesterday vs Today” Comparison Model
Downstream raw collectors do not do a file-level diff of two CSV snapshots. They use a deterministic row-reduction + hash comparison:

1. Load current `shorts_metadata_horizon.csv`.
2. Resolve canonical URL column (`video_url`, `url`, `youtube_url`, or URL-like fallback).
3. Resolve `video_id`:
   - Use `video_id` if present.
   - Else parse from URL.
4. Deduplicate to one latest row per `video_id` by max `captured_at`.
5. Compute `source_hash = sha256(normalized_url)` (default).
   - Optional mode: `sha256(url|captured_at)`.
6. Compare against script-specific SQLite state (`processed_items.video_id -> source_hash,status`).

Identity and change detection contract (critical):
- `video_id`: canonical ID for filenames, joins, and state row key.
- `source_hash`: change detector (URL-only by default).

Interpretation:
- If hash changed, item is treated as updated.
- If hash unchanged but prior status is non-terminal, item is retried.
- If hash unchanged and status terminal, item is skipped.

This is logically equivalent to comparing prior-day processed snapshot to current snapshot, but implemented against persistent stage state instead of full-file diff.

### 2.4 Why URL Is Canonical Source of Truth
Canonicality decision:
- `source_hash` is built from normalized URL by default.
- URL is stable across stages and cloud object keys are keyed by `video_id` extracted from URL.

Reasoning:
- `video_id` can be missing in malformed rows; URL still allows deterministic extraction.
- Hashing URL avoids unnecessary reprocessing when non-critical metadata columns change.
- URL-centric hashing ensures idempotent materialization when content pointer is unchanged.

Tradeoff:
- If meaningful metadata changes while URL remains same, stages that hash only URL will not reprocess unless configured with `--include_captured_at_in_hash`.

### 2.5 Unified `video_id` Generation Strategy
`Data/common/horizon_delta.py::parse_video_id` supports:
- Raw 11-char IDs (`^[A-Za-z0-9_-]{11}$`)
- `youtube.com/watch?v=...`
- `/shorts/<id>`, `/embed/<id>`, `/v/<id>`
- `youtu.be/<id>`
- Regex fallback for any 11-char ID token

This creates one canonical ID namespace across metadata/video/audio/text/embeddings/fusion/training.

---

## 3. Daily Asynchronous Update Workflow

## 3.1 Detection of New or Updated Videos
Each stage independently runs:
- `load_latest_horizon_rows(...)` to compute current canonical row set.
- `compute_delta(...)` to select items requiring work.

Terminal statuses for raw ScriptState delta logic:
- `success`
- `no_captions`
- `fail_empty_transcript_terminal`

Any non-terminal status remains in future delta until terminal or successful.

## 3.2 “Process Only Deltas” Contract
Every raw script uses its own state DB:
- `state/video_downloader.sqlite`
- `state/audio_downloader.sqlite`
- `state/text_downloader.sqlite`

State schema:

| Column | Type | Meaning |
|---|---|---|
| `video_id` | TEXT PK | Canonical identity |
| `source_hash` | TEXT | Last processed source signature |
| `processed_at` | TEXT ISO8601 | Last stage write time |
| `status` | TEXT | Terminal or retryable state |
| `error` | TEXT | Last error string |

Implication:
- Stages are asynchronous and independently restartable.
- A stage can run without waiting for all others; it only consumes the unresolved delta for its own state.

## 3.3 Idempotency Guarantees
Idempotency primitives:
1. Deterministic `video_id`.
2. Deterministic `source_hash`.
3. Per-stage upsert state (`ON CONFLICT(video_id) DO UPDATE`).
4. Terminal status exclusion from future delta.

Result:
- Re-running same inputs does not duplicate output files intentionally.
- Existing successful outputs are marked `skipped_existing` instead of recomputed.

Known caveat:
- If local/cloud artifact exists but state is missing, artifact may be recreated then overwritten in-place (still functionally idempotent for consumers).

---

## 4. Video Download Pipeline

## 4.1 Input
- Delta items from metadata horizon CSV.
- URL from resolved URL column.

Entrypoint:
- `Data/raw/Video/download_video.py`

## 4.2 Output
Local default:
- `Data/raw/Video/raw_data/<video_id>.mp4`

Optional cloud:
- `s3://<bucket>/<prefix>/video/<video_id>.mp4` or equivalent GCS path.

## 4.3 File Naming Convention
- Filename stem is canonical `video_id`.
- Extension normalized to `.mp4`.
- If yt-dlp returns `.mkv/.webm/.m4a`, file is renamed to `.mp4`.

## 4.4 Storage Layout

```text
Data/raw/Video/raw_data/
  <video_id>.mp4

Data/raw/Audio/raw_data/            # optional same-run extraction
  <video_id>.wav

Cloud root (optional):
  <cloud_root_uri>/video/<video_id>.mp4
  <cloud_root_uri>/audio/<video_id>.wav
```

## 4.5 Retry and Fallback Logic
Two layers:

1. Immediate retry loop (`--retry`, exponential delay multiplier).
2. Multi-strategy yt-dlp fallback sequence:
   - caller strategy
   - no-cookie mobile/tv clients
   - cookie web clients
   - final generic `best`

Deterministic terminal failures break early:
- `fail_removed`, `fail_private`, `fail_unavailable`, `fail_upcoming`, `fail_age_restricted`, `fail_auth`.

Example failure statuses:
- `fail_challenge_gated`
- `fail_rate_limited`
- `fail_forbidden`
- `fail_network`
- `fail_format_unavailable`
- `fail_cloud_upload`

---

## 5. Audio Extraction Pipeline

Entrypoint:
- `Data/raw/Audio/download_audio.py`

## 5.1 How Audio Is Derived
From `video/<video_id>.mp4` via ffmpeg:

```bash
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le output.wav
```

## 5.2 Format and Sampling Decisions
- Container: WAV
- Codec: PCM 16-bit little-endian (`pcm_s16le`)
- Channels: mono (`-ac 1`)
- Sample rate: `16 kHz` (`-ar 16000`)

Rationale:
- Matches ASR and wav2vec defaults.
- Avoids resampling mismatch downstream.
- PCM WAV is robust and deterministic for ML ingestion.

## 5.3 Inputs and Outputs
Input:
- `Data/raw/Video/raw_data/<video_id>.mp4` (or cloud fetch fallback)

Output:
- `Data/raw/Audio/raw_data/<video_id>.wav`
- optional cloud upload to `<cloud_root_uri>/audio/<video_id>.wav`

## 5.4 Retry Semantics
- No inner ffmpeg retry loop.
- Non-terminal statuses remain in delta and are retried on next run.
- Missing video is explicit status: `missing_video`.

---

## 6. Text Transcription Pipeline

Entrypoint:
- `Data/raw/Text/text_collect.py`

Flow:
1. Optional caption-first extraction from YouTube subtitles.
2. Fallback ASR on WAV if caption unavailable/fails.

## 6.1 Triggering
Triggered only for text-stage delta items (`compute_delta`).
For each item:
- If output JSON exists and state `success`: skip.
- Else process.

## 6.2 Model and Backend Assumptions
Operational backend used in this project:
- `whisper_cpp` only (alias `whisper`).
- ASR backend is set explicitly via `--asr_backend whisper_cpp` (or equivalent env through local runner).
- No backend switching is used in normal operation.

Execution assumptions:
- `caption_first=True` remains enabled unless explicitly disabled.
- If caption text is not used/available, ASR falls through to `whisper_cpp`.

## 6.3 Output Format
One JSON per video:
- `Data/raw/Text/raw_data/<video_id>.json`

Representative schema fields:

| Field | Meaning |
|---|---|
| `video_id`, `video_url`, `captured_at` | Identity and provenance |
| `status` | `success`, `missing_audio`, `no_captions`, `fail_*` |
| `transcript` | Final text payload |
| `transcript_source` | Operationally `whisper_cpp` (or `youtube_caption` when caption path succeeds) |
| `transcript_model` | Backend model identifier |
| `transcript_language` | Optional detected language |
| `caption_status`, `caption_lang`, `caption_is_auto` | Caption path metadata |
| `timestamps` | Optional ASR timing segments |
| `subtitle_srt`, `subtitle_vtt` | Optional subtitle payload |
| `error` | Error context |

## 6.4 Error Handling and Edge Cases
Classified statuses include:
- `missing_audio`
- `no_captions`
- `fail_format_unavailable`
- `fail_challenge_gated`
- `fail_asr_backend_missing`
- `fail_rate_limited`
- `fail_auth`
- `fail_cloud_upload`
- generic `fail`

Special empty-transcript retry policy (two attempts total):
- Attempt 1:
  - If transcription result is empty, status is written as `fail_empty_transcript`.
- Attempt 2 (next delta run, same `video_id` + same `source_hash`):
  - If transcription is empty again, status is promoted to terminal `fail_empty_transcript_terminal`.
- Operational interpretation:
  - Two consecutive empty transcriptions are treated as “no words/no usable speech.”

This terminalization is critical for fusion placeholder logic.

## 6.5 Parallel Batch Processing
Text transcription is executed as parallel batch processing for pending delta items:
- Implementation uses `ThreadPoolExecutor` + `as_completed` in `text_collect.py`.
- Concurrency is controlled by `--max_workers`.
- Local pipeline default sets `TEXT_MAX_WORKERS=4` (`scripts/run_text_pipeline_local.sh`).
- Processing behavior:
  - Build pending delta list.
  - Submit each item to worker pool.
  - Persist each result to state as soon as its future completes.
  - Keep idempotency guarantees via per-item state upsert (`video_id`, `source_hash`, `status`).

---

## 7. Orchestration: End-to-End Order and Control Plane

## 7.1 Required Execution Order

```text
[1] Horizon metadata collection
      outputs shorts_metadata_horizon.csv
      outputs horizon_labels.sqlite

[2] Video downloader (delta only)
      consumes metadata CSV
      outputs MP4 (+optional WAV)
      updates state/video_downloader.sqlite

[3] Audio extractor (delta only)
      consumes metadata CSV + MP4
      outputs WAV
      updates state/audio_downloader.sqlite

[4] Text collector (delta only)
      consumes metadata CSV + WAV (+optional captions)
      outputs JSON transcript
      updates state/text_downloader.sqlite
```

## 7.2 Scheduling Reality
- Horizon metadata is scheduled in GitHub Actions every 6 hours (`horizon-collector.yml`).
- Legacy `raw-data-async.yml` is intentionally disabled.
- Operational ingestion path is local orchestrators:
  - `scripts/run_raw_pipeline_local.sh` (video+audio)
  - `scripts/run_text_pipeline_local.sh` (text)

Asynchrony is preserved through stage-local delta state, not strict workflow DAG coupling.

---

## 8. Directory and Artifact Layout

```text
Data/raw/Metadata/
  shorts_metadata_horizon.csv
  horizon_labels.sqlite

Data/raw/Video/raw_data/
  <video_id>.mp4

Data/raw/Audio/raw_data/
  <video_id>.wav

Data/raw/Text/raw_data/
  <video_id>.json

state/
  video_downloader.sqlite
  audio_downloader.sqlite
  text_downloader.sqlite
```

Optional cloud mirrors:
- `clipfarm/raw/video/<video_id>.mp4`
- `clipfarm/raw/audio/<video_id>.wav`
- `clipfarm/raw/text/<video_id>.json`
- `clipfarm/state/*.sqlite`

---

## 9. Determinism, Reproducibility, Versioning

Deterministic elements:
- URL -> `video_id` parsing rules.
- Latest-row selection per `video_id` by `captured_at`.
- URL hash generation for delta decisions.
- Stable file naming by `video_id`.

Versioning elements:
- `collector_version` column (`COLLECTOR_VERSION`, default `v0.4.0-horizon`).
- State DB snapshots persisted in cloud.
- Artifact keys include modality prefix + `video_id`.

Non-deterministic elements:
- External API responses (YouTube ranking/availability).
- yt-dlp source format variability.
- ASR backend internals and hardware/runtime differences.

---

## 10. Assumptions, Constraints, Tradeoffs

Constraints:
1. API quota/rate limits affect coverage.
2. Private/removed content is terminally unavailable.
3. Downloader auth/challenge gating can block subsets of videos.

Tradeoffs:
1. URL-only hash improves idempotency but may ignore non-URL metadata changes.
2. Fixed 16 kHz mono WAV simplifies downstream models but discards stereo/high-frequency details.
3. Caption-first reduces ASR cost but may mix manual and automatic subtitle quality.
