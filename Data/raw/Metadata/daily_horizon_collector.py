# Daily horizon labeling collector for YouTube Shorts.
#
# The goal of this script is to:
# - Discover videos that are ~7 days old and ~30 days old.
# - Fetch their current stats.
# - Write rows that match the existing shorts_metadata.csv schema.
# - Add only additive horizon fields for labeling.
# - Deduplicate so the same (video_id, horizon_days) is never written twice.

import os
import sys
import csv
import time
import random
import sqlite3
import requests
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure a clear collector version for horizon labeling runs.
# If the env var is not set, we set a default that identifies this collector.
if "COLLECTOR_VERSION" not in os.environ:
    os.environ["COLLECTOR_VERSION"] = "v0.4.0-horizon"

# Resolve repo root based on the location of this file.
# parents[3] walks up: Data/raw/Metadata/daily_horizon_collector.py -> Data/raw/Metadata -> Data/raw -> Data -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Add repo root to import path so we can import local modules reliably.
sys.path.insert(0, str(REPO_ROOT))

# Import existing metadata helpers and constants from the main collector.
# This keeps feature naming and derived fields consistent with shorts_metadata.csv.
from Data.raw.Metadata import collect_metadata as base

# Import schema helper to preserve the existing header layout.
from Data.raw.Metadata.schema_mapper import build_output_schema

# YouTube API endpoint for search.list calls (discovery step).
YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Output file for horizon-labeled rows.
OUTPUT_CSV = REPO_ROOT / "Data" / "raw" / "Metadata" / "shorts_metadata_horizon.csv"

# Existing schema reference file so we match column ordering and names.
SCHEMA_REFERENCE_CSV = REPO_ROOT / "Data" / "raw" / "Metadata" / "shorts_metadata.csv"

# SQLite DB used for dedupe: primary key is (video_id, horizon_days).
DEDUPE_DB = REPO_ROOT / "Data" / "raw" / "Metadata" / "horizon_labels.sqlite"

# Default target horizons in days.
DEFAULT_HORIZONS = [7, 30]

# Default tolerance window around each horizon in days.
# Example: horizon=7, tolerance=0.5 means 6.5 to 7.5 days old.
DEFAULT_TOLERANCE_DAYS = 0.5

# Default number of pages to request per query.
DEFAULT_PAGES_PER_QUERY = 1

# Default max results per search.list request (max allowed by API is 50).
DEFAULT_MAX_RESULTS = 25

# Default sleep between API calls to reduce quota spikes and rate limit risk.
DEFAULT_SLEEP_SECONDS = 0.2

# Default number of recent shorts to sample per channel when computing
# channel_median_shorts_view_count. Set to 0 to disable extra API calls.
DEFAULT_CHANNEL_MEDIAN_SAMPLE = 0

# Seed queries for discovery when no seed file or overrides are provided.
DEFAULT_SEEDS = [
    {"query": "shorts", "category_type": "seed"},
    {"query": "funny", "category_type": "seed"},
    {"query": "music", "category_type": "seed"},
    {"query": "gaming", "category_type": "seed"},
    {"query": "sports", "category_type": "seed"},
    {"query": "news", "category_type": "seed"},
    {"query": "recipe", "category_type": "seed"},
    {"query": "dance", "category_type": "seed"},
    {"query": "animals", "category_type": "seed"},
    {"query": "comedy", "category_type": "seed"},
    {"query": "tutorial", "category_type": "seed"},
    {"query": "challenge", "category_type": "seed"},
]

# Extra columns added for horizon labeling. These are appended to the schema.
HORIZON_EXTRA_FIELDS = [
    "horizon_days",
    "horizon_view_count",
    "horizon_label_type",
    "channel_median_shorts_view_count",
]


# Convert a datetime to RFC3339 UTC format required by YouTube API.
# Example: 2026-02-05T12:00:00Z
def to_rfc3339(dt: datetime) -> str:
    # Force UTC to avoid ambiguity and format exactly as expected by API.
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Compute the publishedAfter/publishedBefore window for a given horizon.
# The window is centered around now - horizon_days with +/- tolerance.
# Returns two datetimes: (after, before).
def compute_window(now: datetime, horizon_days: int, tolerance_days: float):
    # Earliest acceptable publish time (older bound).
    after = now - timedelta(days=horizon_days + tolerance_days)

    # Latest acceptable publish time (newer bound).
    before = now - timedelta(days=horizon_days - tolerance_days)

    # Defensive check to ensure a valid time range.
    if after >= before:
        raise ValueError("Invalid window: publishedAfter must be earlier than publishedBefore")

    # Return the computed window.
    return after, before


# Call a URL with exponential backoff for transient errors or quota limits.
# Returns the final response object or None if all attempts fail.
def request_with_backoff(url: str, params: dict, max_retries: int = 5, base_delay: float = 1.2):
    # Track the last response so we can return it if we exit the loop.
    last_resp = None

    # Attempt the request multiple times with exponential backoff.
    for attempt in range(max_retries):
        try:
            # Execute the request with a reasonable timeout.
            resp = requests.get(url, params=params, timeout=30)
            last_resp = resp
        except Exception:
            # Any exception results in a None response for this attempt.
            resp = None
            last_resp = None

        # Success case: HTTP 200 indicates a good response.
        if resp is not None and resp.status_code == 200:
            return resp

        # Decide whether to retry based on status codes or network failure.
        status = resp.status_code if resp is not None else None
        if status in (403, 429, 500, 503) or resp is None:
            # Exponential backoff plus a small random jitter.
            sleep_s = base_delay * (2 ** attempt) + random.random() * 0.25
            time.sleep(sleep_s)
            continue

        # For other errors (e.g., 400), return immediately.
        return resp

    # If all retries fail, return the last response (may be None).
    return last_resp


# Load seed queries that guide the search.list discovery.
# Priority: explicit --seed overrides > seed file > DEFAULT_SEEDS.
def load_seeds(seed_file: Path = None, seed_overrides=None):
    # If user provided --seed arguments, use those and skip the file/defaults.
    if seed_overrides:
        return [{"query": s.strip(), "category_type": "seed"} for s in seed_overrides if s.strip()]

    # If a seed CSV is provided and exists, load it.
    if seed_file and seed_file.exists():
        out = []
        with seed_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize the query field.
                q = (row.get("query") or "").strip()
                if not q:
                    continue

                # Keep category_type if present, otherwise default to "seed".
                out.append({
                    "query": q,
                    "category_type": (row.get("category_type") or "seed").strip() or "seed",
                })

        # If we loaded any seeds, return them.
        if out:
            return out

    # Fallback to built-in defaults.
    return list(DEFAULT_SEEDS)


# Use YouTube search.list to discover videos in a given publish window.
# Returns a list of candidate dicts and the raw count of results seen.
def search_videos_for_window(seeds, published_after: str, published_before: str,
                             pages_per_query: int, max_results: int, sleep_seconds: float):
    # List of discovered candidate video records.
    discovered = []

    # Count of items returned by the API before local dedupe.
    raw_count = 0

    # Iterate through each seed query.
    for seed in seeds:
        # Extract and clean the query string.
        query = seed.get("query", "").strip()
        if not query:
            continue

        # Keep the category label for downstream analysis.
        category_type = seed.get("category_type", "seed")

        # Track paging token for search.list pagination.
        page_token = None

        # Limit the number of pages per query to control quota.
        for _ in range(pages_per_query):
            # Build the search.list parameters for the current page.
            params = {
                "part": "snippet",
                "type": "video",
                "videoDuration": "short",
                "order": "date",
                "q": query,
                "publishedAfter": published_after,
                "publishedBefore": published_before,
                "maxResults": max_results,
                "key": base.YOUTUBE_API_KEY,
            }

            # If we have a page token, request that page.
            if page_token:
                params["pageToken"] = page_token

            # Execute the search request with retry/backoff.
            resp = request_with_backoff(YT_SEARCH_URL, params)
            if resp is None or resp.status_code != 200:
                # Log failures and stop paging for this query.
                code = resp.status_code if resp is not None else "n/a"
                print(f"WARNING: search.list failed for query '{query}' ({code})")
                break

            # Parse the JSON payload.
            data = resp.json()

            # Extract items list, defaulting to empty.
            items = data.get("items", [])

            # Track raw item count returned by the API.
            raw_count += len(items)

            # Convert API items into our base candidate records.
            for item in items:
                # Each search item contains an id.videoId field.
                vid = (item.get("id") or {}).get("videoId")
                if not vid:
                    continue

                # Store a minimal record used for later metadata fetch.
                discovered.append({
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "query": query,
                    "category_type": category_type,
                })

            # Prepare to request the next page if it exists.
            page_token = data.get("nextPageToken")
            if not page_token:
                # No more pages available.
                break

            # Sleep between pages to be gentle on the API.
            time.sleep(sleep_seconds)

    # Return the list of candidates and total raw count.
    return discovered, raw_count


# Fetch full metadata for a list of video IDs using videos.list.
# Returns a dict keyed by video_id with standardized fields.
def fetch_metadata_for_ids(ids):
    # Short-circuit for empty inputs.
    if not ids:
        return {}

    # Build the videos.list request parameters.
    params = {
        "part": "snippet,contentDetails,statistics,topicDetails,status",
        "id": ",".join(ids),
        "key": base.YOUTUBE_API_KEY,
    }

    # Execute the API call with backoff.
    resp = request_with_backoff(base.YT_VIDEOS_URL, params)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "n/a"
        print(f"WARNING: videos.list failed ({code}) for batch of {len(ids)}")
        return {}

    # Parse the JSON response.
    data = resp.json()

    # Output dictionary keyed by video_id.
    out = {}

    # Iterate through each video item and normalize fields.
    for item in data.get("items", []):
        vid = item.get("id")
        sn = item.get("snippet", {}) or {}
        cd = item.get("contentDetails", {}) or {}
        st = item.get("statistics", {}) or {}
        td = item.get("topicDetails", {}) or {}
        status = item.get("status", {}) or {}

        # Parse duration fields for derived features.
        duration_iso = cd.get("duration", "")
        duration_seconds = base.iso8601_to_seconds(duration_iso)

        # Pull thumbnail metrics (url, width, height, aspect, vertical flag).
        thumb_url, tw, th, aspect, is_vertical = base.parse_thumb_metrics(sn)

        # Determine if like/comment counts are present.
        has_like = "likeCount" in st
        has_comment = "commentCount" in st

        # publicStatsViewable is used to infer hidden stats.
        public_stats_viewable = status.get("publicStatsViewable")

        # Build the normalized metadata record for this video.
        out[vid] = {
            # Snippet fields
            "title": sn.get("title", "") or "",
            "description": sn.get("description", "") or "",
            "tags": "|".join(sn.get("tags", [])) if sn.get("tags") else "",
            "channel_title": sn.get("channelTitle", "") or "",
            "channel_id": sn.get("channelId", "") or "",
            "published_at": sn.get("publishedAt", "") or "",
            "default_language": sn.get("defaultLanguage", "") or "",
            "default_audio_language": sn.get("defaultAudioLanguage", "") or "",

            # Thumbnail fields
            "thumbnail_url": thumb_url or "",
            "thumb_width": tw,
            "thumb_height": th,
            "thumb_aspect_ratio": aspect,
            "is_vertical_thumb": is_vertical,

            # Content details fields
            "duration": duration_iso,
            "duration_seconds": duration_seconds,
            "dimension": cd.get("dimension"),
            "definition": cd.get("definition"),
            "projection": cd.get("projection"),
            "caption_available": str(cd.get("caption", "")).lower() == "true",
            "licensed_content": cd.get("licensedContent"),

            # Statistics fields (nullable if hidden)
            "view_count": int(st["viewCount"]) if "viewCount" in st else None,
            "like_count": int(st["likeCount"]) if has_like else None,
            "comment_count": int(st["commentCount"]) if has_comment else None,

            # Topic details fields
            "topic_categories": "|".join(td.get("topicCategories", [])) if td.get("topicCategories") else "",

            # Status / policy fields
            "privacy_status": status.get("privacyStatus"),
            "upload_status": status.get("uploadStatus"),
            "embeddable": status.get("embeddable"),
            "license": status.get("license"),
            "madeForKids": status.get("madeForKids"),
            "publicStatsViewable": public_stats_viewable,

            # Derived flags for hidden stats
            "likes_hidden": not has_like,
            "comments_disabled": not has_comment,
            "stats_hidden": (public_stats_viewable is False),
        }

    # Return the normalized metadata dictionary.
    return out


# Fetch channel metadata for a set of channel IDs.
# Returns a dict keyed by channel_id.
def fetch_channels_for_ids(channel_ids):
    # Initialize output container.
    out = {}

    # Short-circuit if there is nothing to fetch.
    if not channel_ids:
        return out

    # The API accepts up to 50 channel IDs per request.
    for batch in base.chunked(list(channel_ids), 50):
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": base.YOUTUBE_API_KEY,
        }

        # Call the channels.list endpoint with backoff.
        resp = request_with_backoff(base.YT_CHANNELS_URL, params)
        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "n/a"
            print(f"WARNING: channels.list failed ({code}) for batch of {len(batch)}")
            continue

        # Parse JSON response.
        data = resp.json()

        # Normalize each channel record.
        for item in data.get("items", []):
            cid = item.get("id")
            sn = item.get("snippet", {}) or {}
            st = item.get("statistics", {}) or {}
            out[cid] = {
                "channel_custom_url": sn.get("customUrl", ""),
                "channel_country": sn.get("country", ""),
                "channel_created_at": sn.get("publishedAt", "") or "",
                "channel_description_length": len(sn.get("description", "") or ""),
                "channel_subscriber_count": int(st["subscriberCount"]) if "subscriberCount" in st else None,
                "channel_video_count": int(st["videoCount"]) if "videoCount" in st else None,
                "channel_view_count": int(st["viewCount"]) if "viewCount" in st else None,
                "channel_hidden_subscriber_count": bool(st.get("hiddenSubscriberCount")),
            }

    # Return channel metadata dictionary.
    return out


# Fetch up to N recent shorts video IDs for a given channel using search.list.
def fetch_channel_short_ids(channel_id: str, max_videos: int, sleep_seconds: float):
    if not channel_id or max_videos <= 0:
        return []

    ids = []
    page_token = None

    while len(ids) < max_videos:
        params = {
            "part": "snippet",
            "type": "video",
            "channelId": channel_id,
            "videoDuration": "short",
            "order": "date",
            "maxResults": min(50, max_videos - len(ids)),
            "key": base.YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = request_with_backoff(YT_SEARCH_URL, params)
        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "n/a"
            print(f"WARNING: search.list failed for channel {channel_id} ({code})")
            break

        data = resp.json()
        items = data.get("items", [])

        for item in items:
            vid = (item.get("id") or {}).get("videoId")
            if vid:
                ids.append(vid)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(sleep_seconds)

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for vid in ids:
        if vid in seen:
            continue
        seen.add(vid)
        deduped.append(vid)

    return deduped


# Compute the median view count across recent shorts for a channel.
def compute_channel_median_shorts_view_count(channel_id: str, max_videos: int, sleep_seconds: float):
    short_ids = fetch_channel_short_ids(channel_id, max_videos, sleep_seconds)
    if not short_ids:
        return None

    view_counts = []
    for batch in base.chunked(short_ids, 50):
        meta = fetch_metadata_for_ids(batch)
        for item in meta.values():
            v = item.get("view_count")
            if v is not None:
                view_counts.append(v)

    if not view_counts:
        return None

    return statistics.median(view_counts)


# Initialize the SQLite DB used to dedupe labels.
def init_dedupe_db(path: Path):
    # Make sure the parent folder exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open SQLite connection.
    conn = sqlite3.connect(path)

    # Enable WAL for better concurrent read/write behavior.
    conn.execute("PRAGMA journal_mode=WAL")

    # Create the dedupe table if it does not already exist.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS horizon_labels (
            video_id TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            labeled_at TEXT NOT NULL,
            PRIMARY KEY (video_id, horizon_days)
        )
        """
    )

    # Persist any schema changes.
    conn.commit()

    # Return the open connection.
    return conn


# Check if a (video_id, horizon_days) label has already been recorded.
def is_labeled(conn: sqlite3.Connection, video_id: str, horizon_days: int) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM horizon_labels WHERE video_id = ? AND horizon_days = ?",
        (video_id, horizon_days),
    )
    return cur.fetchone() is not None


# Insert new labels into the dedupe table.
def record_labels(conn: sqlite3.Connection, rows):
    # Skip if there is nothing to record.
    if not rows:
        return

    # Build payload rows for bulk insert.
    payload = [(r["video_id"], int(r["horizon_days"]), r.get("captured_at", "")) for r in rows]

    # Insert new labels, ignoring any duplicates.
    conn.executemany(
        "INSERT OR IGNORE INTO horizon_labels (video_id, horizon_days, labeled_at) VALUES (?, ?, ?)",
        payload,
    )

    # Commit insertions.
    conn.commit()


# Prepare a CSV writer that enforces the expected schema.
def prepare_output_writer(path: Path, fields: list):
    # Make sure output directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # If the file exists, validate its header matches the expected schema.
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:
            raise SystemExit(
                "ERROR: Output schema mismatch. "
                "If you intended to create a new file, delete or move the existing output file."
            )

    # Open in append mode so we never overwrite previous labels.
    f = path.open("a", encoding="utf-8", newline="")

    # DictWriter ensures column alignment and ignores extra fields.
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")

    # If file is empty, write the header first.
    if path.stat().st_size == 0:
        writer.writeheader()

    # Return both the file handle and writer.
    return f, writer


# Main entry point for the daily collector.
def main():
    import argparse

    # Build CLI to allow overrides without editing code.
    parser = argparse.ArgumentParser(description="Daily horizon labeling collector for YouTube Shorts")
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--schema-reference", default=str(SCHEMA_REFERENCE_CSV))
    parser.add_argument("--dedupe-db", default=str(DEDUPE_DB))
    parser.add_argument("--seed-file", default="")
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--tolerance-days", type=float, default=DEFAULT_TOLERANCE_DAYS)
    parser.add_argument("--pages-per-query", type=int, default=DEFAULT_PAGES_PER_QUERY)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--channel-median-sample", type=int, default=DEFAULT_CHANNEL_MEDIAN_SAMPLE)
    args = parser.parse_args()

    # Parse horizons list like "7,30" into [7, 30].
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    # Normalize tolerance and request limits.
    tolerance_days = float(args.tolerance_days)
    pages_per_query = max(1, int(args.pages_per_query))
    max_results = max(1, min(50, int(args.max_results)))
    sleep_seconds = max(0.0, float(args.sleep_seconds))
    channel_median_sample = max(0, int(args.channel_median_sample))
    channel_median_enabled = channel_median_sample > 0

    # Resolve seed sources.
    seed_file = Path(args.seed_file) if args.seed_file else None
    seeds = load_seeds(seed_file=seed_file, seed_overrides=args.seed)

    # Capture current time for labeling.
    now = base.now_utc()
    captured_at_iso = now.isoformat()

    # Log run configuration.
    print(f"Collector version: {base.COLLECTOR_VERSION}")
    print(f"Captured at: {captured_at_iso}")
    print(f"Horizons: {horizons} days (tolerance +/- {tolerance_days} days)")
    print(f"Channel median shorts sample: {channel_median_sample}")

    # Map of video_id -> base discovery record.
    all_candidates = {}

    # Total number of raw search results before dedupe.
    raw_discovered = 0

    # Track which horizon a video was discovered under (useful for debugging).
    discovered_by_horizon = {h: set() for h in horizons}

    # Discovery phase for each horizon window.
    for horizon in horizons:
        # Compute the publish window for this horizon.
        after_dt, before_dt = compute_window(now, horizon, tolerance_days)
        published_after = to_rfc3339(after_dt)
        published_before = to_rfc3339(before_dt)
        print(f"Searching horizon {horizon}d: {published_after} to {published_before}")

        # Run search.list queries with the selected seeds.
        found, raw = search_videos_for_window(
            seeds=seeds,
            published_after=published_after,
            published_before=published_before,
            pages_per_query=pages_per_query,
            max_results=max_results,
            sleep_seconds=sleep_seconds,
        )

        # Track raw and deduped discovery.
        raw_discovered += raw
        for item in found:
            vid = item["video_id"]
            discovered_by_horizon[horizon].add(vid)
            if vid not in all_candidates:
                all_candidates[vid] = item

    # Convert the candidate map into a list of video IDs.
    candidate_ids = list(all_candidates.keys())

    # Log discovery totals.
    print(f"Candidates discovered (raw): {raw_discovered}")
    print(f"Candidates after dedupe: {len(candidate_ids)}")

    # Fetch video metadata for all candidates in batches.
    all_meta = {}
    for batch in base.chunked(candidate_ids, 50):
        all_meta.update(fetch_metadata_for_ids(batch))
        time.sleep(sleep_seconds)

    # Log metadata fetch count.
    print(f"Fetched metadata for {len(all_meta)} videos")

    # Fetch channel metadata for all channels present in video metadata.
    channel_ids = {m.get("channel_id") for m in all_meta.values() if m.get("channel_id")}
    channels = fetch_channels_for_ids(channel_ids)

    # Optionally compute channel-level median shorts view counts.
    channel_median_shorts = {}
    if channel_median_enabled:
        for cid in channel_ids:
            if cid in channel_median_shorts:
                continue
            channel_median_shorts[cid] = compute_channel_median_shorts_view_count(
                cid, channel_median_sample, sleep_seconds
            )

    # Build rows and assign horizon labels.
    rows_to_write = []
    labeled_counts = {h: 0 for h in horizons}

    # Convert each candidate into a full row and attach horizon labels.
    for vid, base_item in all_candidates.items():
        meta = all_meta.get(vid)
        if not meta:
            continue

        # Attach channel metadata (if any) and compute derived fields.
        chan = channels.get(meta.get("channel_id", ""), {})
        row = base.build_row(base_item, meta, chan, captured_at_iso)
        age_days = row.get("age_days")
        view_count = row.get("view_count")

        # Attach channel median shorts view count if computed.
        if channel_median_enabled:
            row["channel_median_shorts_view_count"] = channel_median_shorts.get(
                meta.get("channel_id", ""), None
            )
        else:
            # If disabled, force all rows to NULL so the column is consistent.
            row["channel_median_shorts_view_count"] = None

        # Skip rows missing key features needed for labeling.
        if age_days is None or view_count is None:
            continue

        # Assign row to each horizon window it qualifies for.
        for horizon in horizons:
            if abs(age_days - horizon) <= tolerance_days:
                row_h = dict(row)
                row_h["horizon_days"] = int(horizon)
                row_h["horizon_view_count"] = int(view_count)
                row_h["horizon_label_type"] = "views_at_age"
                rows_to_write.append(row_h)
                labeled_counts[horizon] += 1

    # Initialize dedupe database.
    conn = init_dedupe_db(Path(args.dedupe_db))

    # Filter out rows already labeled for the same horizon.
    filtered_rows = []
    for r in rows_to_write:
        if is_labeled(conn, r["video_id"], r["horizon_days"]):
            continue
        filtered_rows.append(r)

    # Prepare output schema and CSV writer.
    fields = build_output_schema(Path(args.schema_reference), HORIZON_EXTRA_FIELDS)
    f, writer = prepare_output_writer(Path(args.output), fields)

    # Write rows to the output file.
    for r in filtered_rows:
        writer.writerow(r)
    f.close()

    # Record labels so we do not write them again on later runs.
    record_labels(conn, filtered_rows)
    conn.close()

    # Final summary logging.
    for h in horizons:
        print(f"Labeled {h}d (pre-dedupe): {labeled_counts[h]}")
    print(f"Rows written (post-dedupe): {len(filtered_rows)}")
    print(f"Output: {args.output}")


# Standard Python entry point guard.
if __name__ == "__main__":
    main()
