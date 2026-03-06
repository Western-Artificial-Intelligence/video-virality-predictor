# Interpretation Pipeline (Dev4) Detailed Summary

## 1. Scope
This document explains the interpretation stage that converts cluster assignments into human-auditable per-video style attributes.

Primary files:

- `Interpretation/build_interpretation.py`
- `Interpretation/combine_cluster_and_links.py`
- cluster pipeline orchestration in `scripts/run_cluster_pipeline_local.sh`

The output target is:

- `Interpretation/interpretation.csv`

---

## 2. Upstream Inputs

Required:

- `Unsup_Cluster/cluster_results.csv` (must include at least `video_id`, `cluster`)
- Metadata CSV (`Data/raw/Metadata/shorts_metadata_horizon.csv` by default)

Optional:

- S3 access for missing local media retrieval:
  - `clipfarm/raw/video/<video_id>.mp4`
  - `clipfarm/raw/audio/<video_id>.wav`

Default local media cache directories:

- `Data/raw/Video/raw_data`
- `Data/raw/Audio/raw_data`

---

## 3. URL Join Compatibility Step

Script:

- `Interpretation/combine_cluster_and_links.py`

Purpose:

- Build compatibility table `cluster_links.csv` by attaching URL to each clustered `video_id`.

Mechanics:

1. Build `video_id -> url` map from latest metadata rows using `load_latest_horizon_rows(...)`.
2. Read cluster CSV as `DictReader`, enforce required columns:
   - `video_id`, `cluster`
3. Attach `url` (empty string if not found)
4. Sort by cluster then video_id
5. Write out:
   - default `Interpretation/cluster_links.csv`

Output includes original cluster fields plus `url`.

---

## 4. Interpretation CSV Build Flow

Script:

- `Interpretation/build_interpretation.py`

High-level sequence:

1. Parse args and validate input files
2. Load cluster rows
3. Create local media dirs if absent
4. Build `video_id -> url` from latest metadata
5. Initialize optional S3 client when `fetch_missing` is enabled and bucket is set
6. Iterate all cluster rows and compute metrics per `video_id`
7. Sort output by `cluster, video_id`
8. Write `interpretation.csv`

---

## 5. Latest Metadata / URL Semantics

URL mapping uses:

- `load_latest_horizon_rows(...)` from `Data/common/horizon_delta.py`

Important behavior:

1. Canonicalizes `video_id` if missing by parsing URL fields
2. Chooses latest row per `video_id` by parsed `captured_at`
3. Returns one URL per `video_id` for join

This keeps interpretation URL mapping consistent with other `video_id`-native stages.

---

## 6. Missing Media Policy

Function:

- `_download_if_missing(...)`

Behavior:

1. If local file exists -> available
2. Else, if no S3 client -> unavailable
3. Else attempt S3 download
4. On exception -> unavailable

Fetch keys:

- Video: `<raw_prefix>/video/<video_id>.mp4`
- Audio: `<raw_prefix>/audio/<video_id>.wav`

Default `raw_prefix`:

- `clipfarm/raw`

Traceability counters:

- `video_downloaded`
- `audio_downloaded`
- `video_missing`
- `audio_missing`

---

## 7. Metric Definitions

Each row computes:

1. `motion_mean`
2. `cut_rate_per_min`
3. `audio_rms_mean`
4. `audio_rms_std`
5. `visual_density`

All metrics are numeric and default to `0.0` when unavailable or decode fails.

## 7.1 `motion_mean` and `cut_rate_per_min` (`pacing_metrics`)

Input:

- video file path
- `fps_sample` (default `2`)
- `diff_thresh` (default `25.0`)

Process:

1. Open video stream with PyAV
2. Determine source FPS:
   - `stream.average_rate` if present else `30.0`
3. Frame step:
   - `step = max(1, int(src_fps / max(1, fps_sample)))`
4. Sample grayscale frames
5. Compute frame-to-frame mean absolute difference:
   - `diff = mean(abs(gray_t - gray_{t-1}))`
6. `motion_mean = mean(diffs)`
7. Count cut events where `diff > diff_thresh`
8. Estimate duration:
   - `duration_s = total_frames / src_fps`
9. `cut_rate_per_min = cuts / (duration_s / 60)`

Failure fallback:

- Missing file, decode issues, no video stream, or runtime errors -> both `0.0`

## 7.2 `audio_rms_mean` and `audio_rms_std` (`audio_energy_metrics`)

Input:

- audio file path

Process:

1. Open audio with PyAV
2. Decode audio frames
3. Flatten samples to 1D float32
4. Per frame RMS:
   - `sqrt(mean(samples^2))`
5. Aggregate:
   - `audio_rms_mean = mean(rms_values)`
   - `audio_rms_std = std(rms_values)`

Failure fallback:

- Missing file, decode issues, no audio stream, empty frames, runtime errors -> both `0.0`

## 7.3 `visual_density` (`visual_density`)

Input:

- video file path
- `fps_sample` (function default `1`)
- `edge_thresh` (default `20.0`)

Process:

1. Open video with PyAV
2. Sample grayscale frames using step from fps ratio
3. Compute gradients:
   - `gx, gy = gradient(gray)`
4. Magnitude:
   - `mag = sqrt(gx^2 + gy^2)`
5. Per frame density:
   - fraction of pixels where `mag > edge_thresh`
6. `visual_density = mean(frame_density_values)`

Failure fallback:

- Missing file, decode issues, no video stream, runtime errors -> `0.0`

---

## 8. Availability Flags and Row Retention

For every cluster input row, output keeps row (unless `video_id` is empty) and adds:

- `video_available` (`0`/`1`)
- `audio_available` (`0`/`1`)

Missing media does not remove rows.
Unavailable/failed metrics are represented with `0.0` plus availability flags for traceability.

---

## 9. Output Schema and Ordering

`interpretation.csv` row fields:

- `video_id`
- `cluster`
- `cluster_id`
- `fusion_strategy`
- `k_selected`
- `url`
- `video_available`
- `audio_available`
- `motion_mean`
- `cut_rate_per_min`
- `audio_rms_mean`
- `audio_rms_std`
- `visual_density`

Ordering:

- sorted by `cluster`, then `video_id`

---

## 10. Decode-Failure Counters

Additional counters are tracked and logged:

- `video_decode_failures`
- `audio_decode_failures`

Heuristic used:

- If media file exists but computed metric pair is exactly zeros:
  - video pair (`motion_mean`, `cut_rate_per_min`) -> video decode failure
  - audio pair (`audio_rms_mean`, `audio_rms_std`) -> audio decode failure

This is a practical telemetry signal, not a perfect decoder-health oracle.

---

## 11. CLI Arguments (Interpretation)

Key args in `build_interpretation.py`:

- `--cluster_csv` (default `Unsup_Cluster/cluster_results.csv`)
- `--metadata_csv` (default metadata path)
- `--output_csv` (default `Interpretation/interpretation.csv`)
- `--video_dir` / `--audio_dir`
- `--s3_bucket` / `--s3_region`
- `--raw_prefix` (default `clipfarm/raw`)
- `--fetch_missing` / `--no-fetch_missing`
- `--fps_sample` (default `2`)
- `--diff_thresh` (default `25.0`)
- `--edge_thresh` (default `20.0`)

---

## 12. End-to-End Orchestration Order

In `scripts/run_cluster_pipeline_local.sh`, interpretation is stage 3:

1. Run clustering (`cluster.py`) -> `cluster_results.csv`
2. Run URL join compatibility script -> `cluster_links.csv`
3. Run interpretation builder -> `interpretation.csv`

Relevant environment defaults:

- `FETCH_MISSING_MEDIA=1` (translated to flag form)
- `FPS_SAMPLE=2`
- `DIFF_THRESH=25`
- `EDGE_THRESH=20`
- `RAW_PREFIX=clipfarm/raw`

---

## 13. Failure Modes and Expected Behavior

Hard failures:

- Missing cluster CSV
- Missing metadata CSV
- Missing required cluster columns (`video_id`, `cluster`)

Soft failures handled in-row:

- Missing local media
- S3 download errors
- Decode/open failures
- Missing URL map entry

Soft failures keep row and fill zeros/flags.

---

## 14. Practical Interpretation Notes

1. Metrics are intentionally lightweight and scalable; they are not full cinematic feature extraction.
2. Thresholds (`diff_thresh`, `edge_thresh`) strongly affect relative cluster profiles and should remain fixed for run-to-run comparability.
3. `video_available`/`audio_available` should be included in any downstream aggregate analysis to avoid misreading zero-filled metrics as true low-signal content.
4. For reproducible reports, record:
   - exact threshold values
   - fetch policy (`fetch_missing`)
   - raw prefix and media directories
   - timestamp/run id

