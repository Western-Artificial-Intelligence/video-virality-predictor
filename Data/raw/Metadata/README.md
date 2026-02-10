**Overview**
This folder contains the YouTube Shorts metadata collectors and outputs used to build fixed‑horizon labels (for example, `views_7d` and `views_30d`). The goal is to predict absolute view counts pre‑publish. Raw `view_count` is noisy because videos are captured at inconsistent ages, and rate features like `views_per_day` assume linear growth that Shorts do not follow (spike + decay). Horizon labeling fixes this by always measuring views at a consistent age.

**Core Idea (Horizon Labeling)**
- Each day we discover videos that are approximately 7 days old and 30 days old.
- We fetch their current stats and write a row with `horizon_days` and `horizon_view_count`.
- This keeps the output schema compatible with `shorts_metadata.csv` while giving clean, fixed‑age labels.

**Shorts Definition**
We rely on the YouTube API's short‑form filter:
- `search.list(..., videoDuration="short")` (under ~4 minutes)
- We do not enforce the strict <=60s Shorts rule.

**Key Files**
- `collect_metadata.py`
  - Existing collector for known IDs. Defines the canonical schema and derived features.
- `daily_horizon_collector.py`
  - Daily discovery + horizon labeling collector.
- `schema_mapper.py`
  - Ensures output column order matches `shorts_metadata.csv` and appends horizon fields.
- `shorts_metadata.csv`
  - Schema reference for column names and ordering.
- `shorts_metadata_horizon.csv`
  - Append‑only output for horizon labeled rows.
- `horizon_labels.sqlite`
  - Dedupe store keyed by `(video_id, horizon_days)`.
  - Stored under `Data/raw/Metadata/`.

**How It Works (Daily Collector)**
1. Compute publish windows for each horizon (default 7 and 30 days, tolerance +/- 0.5 days).
2. Discover candidates using `search.list` with multiple broad seed queries.
3. Deduplicate video IDs aggressively.
4. Fetch full metadata using `videos.list` in batches of 50 IDs.
5. Fetch channel metadata using `channels.list`.
6. Assign horizon labels:
   - If age is within tolerance of 7 days, write `horizon_days=7` and `horizon_view_count=view_count`.
   - If age is within tolerance of 30 days, write `horizon_days=30` and `horizon_view_count=view_count`.
7. Append to `Data/raw/Metadata/shorts_metadata_horizon.csv` and record `(video_id, horizon_days)` in `Data/raw/Metadata/horizon_labels.sqlite`.

**Output Schema**
The output mirrors `shorts_metadata.csv` as closely as possible. New horizon fields are additive:
- `horizon_days` (7 or 30)
- `horizon_view_count` (current `view_count` at labeling time)
- `horizon_label_type` (default `views_at_age`)

**Channel Median Shorts Views**
A channel‑level median of recent Shorts can be computed but is disabled by default to save quota.
- Enabled via: `--channel-median-sample 25`
- Disabled via: `--channel-median-sample 0`
- When disabled, `channel_median_shorts_view_count` is set to `NULL` for every row so the column is consistent.

**Usage**
Basic run (horizon labels only):
```bash
python Data/raw/Metadata/daily_horizon_collector.py
```

Low‑quota test run:
```bash
python Data/raw/Metadata/daily_horizon_collector.py \
  --seed shorts --pages-per-query 1 --max-results 5 --sleep-seconds 0.1
```

Enable channel median (higher quota cost):
```bash
python Data/raw/Metadata/daily_horizon_collector.py \
  --channel-median-sample 25
```

**Scheduling (macOS)**
A LaunchAgent can run the collector daily at 11:30 PM. The current plist is:
- `~/Library/LaunchAgents/com.clipfarm.horizon-collector.plist`

Verify it is loaded:
```bash
launchctl list | grep clipfarm
```

Inspect schedule:
```bash
plutil -p ~/Library/LaunchAgents/com.clipfarm.horizon-collector.plist
```

**Always-On Scheduling (GitHub Actions)**
- Workflow file: `.github/workflows/horizon-collector.yml`
- Trigger:
  - Nightly cron (`07:30 UTC`, approximately `11:30 PM` Pacific; DST shifts by 1 hour)
  - Manual run (`workflow_dispatch`)
- Runtime behavior:
  - Installs minimal collector dependency (`requests`)
  - Writes `credentials.py` from GitHub secret `YOUTUBE_API_KEY`
  - Runs `python Data/raw/Metadata/daily_horizon_collector.py --channel-median-sample 0`

**GitHub Secrets Required**
- Always required:
  - `YOUTUBE_API_KEY`
- Required only for S3 persistence mode:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`
  - `S3_BUCKET`
- Required only for GCS persistence mode:
  - `GCP_SA_KEY_JSON`
  - `GCS_BUCKET`

**Persistence Modes**
- The workflow supports three output persistence modes via `PERSIST_MODE` in `.github/workflows/horizon-collector.yml`:
  - `repo` (default): commits `Data/raw/Metadata/shorts_metadata_horizon.csv` and `Data/raw/Metadata/horizon_labels.sqlite` back to git
  - `s3`: uploads outputs to `s3://<S3_BUCKET>/metadata/`
  - `gcs`: uploads outputs to `gs://<GCS_BUCKET>/metadata/`
- To switch mode, edit:
  - `PERSIST_MODE: repo` to `PERSIST_MODE: s3` or `PERSIST_MODE: gcs`

**Quota Notes**
- `search.list` is expensive (100 units per request). This affects discovery and channel median sampling.
- `videos.list` and `channels.list` are cheaper and batched.
- If you see many 403s, you are likely at or near quota limits.

**Troubleshooting**
- Missing `requests` error: install deps in the correct venv.
- Schema mismatch error: move/delete an old output file if the header differs.
- 403s on channel median: reduce or disable `--channel-median-sample`.

**Why This Solves the Problem**
Horizon labeling stabilizes the target by measuring views at consistent ages. This creates a clean supervised target for prediction while allowing the rest of the schema (text, channel, and metadata features) to remain compatible with existing pipelines.

**Daily Delta Downloaders**
- Source of truth for all downstream media/text jobs:
  - `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/Data/raw/Metadata/shorts_metadata_horizon.csv`
- Canonical ID:
  - `video_id` column is used when present.
  - Otherwise parsed from URL (`video_url`, `url`, or `youtube_url`).
- Shared preprocessing logic implemented in:
  - `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/Data/common/horizon_delta.py`
- What the helper does:
  - Loads horizon CSV.
  - Resolves URL column.
  - Ensures `video_id`.
  - Deduplicates by `video_id` (keeps latest `captured_at`).
  - Computes `source_hash` from normalized URL (or URL+captured_at if enabled).
  - Computes daily delta against script-specific SQLite state.

**Per-Script State**
- `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/state/video_downloader.sqlite`
- `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/state/audio_downloader.sqlite`
- `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/state/text_downloader.sqlite`
- State schema fields:
  - `video_id` (primary key)
  - `source_hash`
  - `processed_at`
  - `status`
  - `error`

**Output Layout**
- Video files:
  - `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/Data/raw/Video/<video_id>.mp4`
- Audio files:
  - `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/Data/raw/Audio/<video_id>.wav`
- Text files:
  - `/Users/wkdghdus/Desktop/coding/clipfarm/video-virality-predictor/Data/raw/Text/<video_id>.json`

**Run Commands (Daily Safe/Idempotent)**
- Video downloader:
```bash
python Data/raw/Video/download_video.py
```
- Audio extractor:
```bash
python Data/raw/Audio/download_audio.py
```
- Text downloader/transcript collector:
```bash
python Data/raw/Text/text_collect.py
```
- Optional cap for testing:
  - `--max_items 10`
