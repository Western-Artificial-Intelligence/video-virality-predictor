import os
import sys
import csv
import time
import random
import sqlite3
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure a clear collector version for horizon labeling runs
if "COLLECTOR_VERSION" not in os.environ:
    os.environ["COLLECTOR_VERSION"] = "v0.4.0-horizon"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Data.Metadata import collect_metadata as base
from Data.Metadata.schema_mapper import build_output_schema

YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

OUTPUT_CSV = REPO_ROOT / "Data" / "Metadata" / "shorts_metadata_horizon.csv"
SCHEMA_REFERENCE_CSV = REPO_ROOT / "Data" / "Metadata" / "shorts_metadata.csv"
DEDUPE_DB = REPO_ROOT / "Data" / "Metadata" / "horizon_labels.sqlite"

DEFAULT_HORIZONS = [7, 30]
DEFAULT_TOLERANCE_DAYS = 0.5
DEFAULT_PAGES_PER_QUERY = 1
DEFAULT_MAX_RESULTS = 25
DEFAULT_SLEEP_SECONDS = 0.2

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

HORIZON_EXTRA_FIELDS = [
    "horizon_days",
    "horizon_view_count",
    "horizon_label_type",
]


def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_window(now: datetime, horizon_days: int, tolerance_days: float):
    after = now - timedelta(days=horizon_days + tolerance_days)
    before = now - timedelta(days=horizon_days - tolerance_days)
    if after >= before:
        raise ValueError("Invalid window: publishedAfter must be earlier than publishedBefore")
    return after, before


def request_with_backoff(url: str, params: dict, max_retries: int = 5, base_delay: float = 1.2):
    last_resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            last_resp = resp
        except Exception:
            resp = None
            last_resp = None

        if resp is not None and resp.status_code == 200:
            return resp

        status = resp.status_code if resp is not None else None
        if status in (403, 429, 500, 503) or resp is None:
            sleep_s = base_delay * (2 ** attempt) + random.random() * 0.25
            time.sleep(sleep_s)
            continue

        return resp

    return last_resp


def load_seeds(seed_file: Path = None, seed_overrides=None):
    if seed_overrides:
        return [{"query": s.strip(), "category_type": "seed"} for s in seed_overrides if s.strip()]
    if seed_file and seed_file.exists():
        out = []
        with seed_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("query") or "").strip()
                if not q:
                    continue
                out.append({
                    "query": q,
                    "category_type": (row.get("category_type") or "seed").strip() or "seed",
                })
        if out:
            return out
    return list(DEFAULT_SEEDS)


def search_videos_for_window(seeds, published_after: str, published_before: str,
                             pages_per_query: int, max_results: int, sleep_seconds: float):
    discovered = []
    raw_count = 0

    for seed in seeds:
        query = seed.get("query", "").strip()
        if not query:
            continue
        category_type = seed.get("category_type", "seed")
        page_token = None

        for _ in range(pages_per_query):
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
            if page_token:
                params["pageToken"] = page_token

            resp = request_with_backoff(YT_SEARCH_URL, params)
            if resp is None or resp.status_code != 200:
                code = resp.status_code if resp is not None else "n/a"
                print(f"WARNING: search.list failed for query '{query}' ({code})")
                break

            data = resp.json()
            items = data.get("items", [])
            raw_count += len(items)

            for item in items:
                vid = (item.get("id") or {}).get("videoId")
                if not vid:
                    continue
                discovered.append({
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "query": query,
                    "category_type": category_type,
                })

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(sleep_seconds)

    return discovered, raw_count


def fetch_metadata_for_ids(ids):
    if not ids:
        return {}
    params = {
        "part": "snippet,contentDetails,statistics,topicDetails,status",
        "id": ",".join(ids),
        "key": base.YOUTUBE_API_KEY,
    }
    resp = request_with_backoff(base.YT_VIDEOS_URL, params)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "n/a"
        print(f"WARNING: videos.list failed ({code}) for batch of {len(ids)}")
        return {}

    data = resp.json()
    out = {}
    for item in data.get("items", []):
        vid = item.get("id")
        sn = item.get("snippet", {}) or {}
        cd = item.get("contentDetails", {}) or {}
        st = item.get("statistics", {}) or {}
        td = item.get("topicDetails", {}) or {}
        status = item.get("status", {}) or {}

        duration_iso = cd.get("duration", "")
        duration_seconds = base.iso8601_to_seconds(duration_iso)
        thumb_url, tw, th, aspect, is_vertical = base.parse_thumb_metrics(sn)

        has_like = "likeCount" in st
        has_comment = "commentCount" in st
        public_stats_viewable = status.get("publicStatsViewable")

        out[vid] = {
            "title": sn.get("title", "") or "",
            "description": sn.get("description", "") or "",
            "tags": "|".join(sn.get("tags", [])) if sn.get("tags") else "",
            "channel_title": sn.get("channelTitle", "") or "",
            "channel_id": sn.get("channelId", "") or "",
            "published_at": sn.get("publishedAt", "") or "",
            "default_language": sn.get("defaultLanguage", "") or "",
            "default_audio_language": sn.get("defaultAudioLanguage", "") or "",
            "thumbnail_url": thumb_url or "",
            "thumb_width": tw,
            "thumb_height": th,
            "thumb_aspect_ratio": aspect,
            "is_vertical_thumb": is_vertical,
            "duration": duration_iso,
            "duration_seconds": duration_seconds,
            "dimension": cd.get("dimension"),
            "definition": cd.get("definition"),
            "projection": cd.get("projection"),
            "caption_available": str(cd.get("caption", "")).lower() == "true",
            "licensed_content": cd.get("licensedContent"),
            "view_count": int(st["viewCount"]) if "viewCount" in st else None,
            "like_count": int(st["likeCount"]) if has_like else None,
            "comment_count": int(st["commentCount"]) if has_comment else None,
            "topic_categories": "|".join(td.get("topicCategories", [])) if td.get("topicCategories") else "",
            "privacy_status": status.get("privacyStatus"),
            "upload_status": status.get("uploadStatus"),
            "embeddable": status.get("embeddable"),
            "license": status.get("license"),
            "madeForKids": status.get("madeForKids"),
            "publicStatsViewable": public_stats_viewable,
            "likes_hidden": not has_like,
            "comments_disabled": not has_comment,
            "stats_hidden": (public_stats_viewable is False),
        }
    return out


def fetch_channels_for_ids(channel_ids):
    out = {}
    if not channel_ids:
        return out
    for batch in base.chunked(list(channel_ids), 50):
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": base.YOUTUBE_API_KEY,
        }
        resp = request_with_backoff(base.YT_CHANNELS_URL, params)
        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "n/a"
            print(f"WARNING: channels.list failed ({code}) for batch of {len(batch)}")
            continue
        data = resp.json()
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
    return out


def init_dedupe_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
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
    conn.commit()
    return conn


def is_labeled(conn: sqlite3.Connection, video_id: str, horizon_days: int) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM horizon_labels WHERE video_id = ? AND horizon_days = ?",
        (video_id, horizon_days),
    )
    return cur.fetchone() is not None


def record_labels(conn: sqlite3.Connection, rows):
    if not rows:
        return
    payload = [(r["video_id"], int(r["horizon_days"]), r.get("captured_at", "")) for r in rows]
    conn.executemany(
        "INSERT OR IGNORE INTO horizon_labels (video_id, horizon_days, labeled_at) VALUES (?, ?, ?)",
        payload,
    )
    conn.commit()


def prepare_output_writer(path: Path, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:
            raise SystemExit(
                "ERROR: Output schema mismatch. "
                "If you intended to create a new file, delete or move the existing output file."
            )
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if path.stat().st_size == 0:
        writer.writeheader()
    return f, writer


def main():
    import argparse

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
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    tolerance_days = float(args.tolerance_days)
    pages_per_query = max(1, int(args.pages_per_query))
    max_results = max(1, min(50, int(args.max_results)))
    sleep_seconds = max(0.0, float(args.sleep_seconds))

    seed_file = Path(args.seed_file) if args.seed_file else None
    seeds = load_seeds(seed_file=seed_file, seed_overrides=args.seed)

    now = base.now_utc()
    captured_at_iso = now.isoformat()

    print(f"Collector version: {base.COLLECTOR_VERSION}")
    print(f"Captured at: {captured_at_iso}")
    print(f"Horizons: {horizons} days (tolerance +/- {tolerance_days} days)")

    all_candidates = {}
    raw_discovered = 0
    discovered_by_horizon = {h: set() for h in horizons}

    for horizon in horizons:
        after_dt, before_dt = compute_window(now, horizon, tolerance_days)
        published_after = to_rfc3339(after_dt)
        published_before = to_rfc3339(before_dt)
        print(f"Searching horizon {horizon}d: {published_after} to {published_before}")

        found, raw = search_videos_for_window(
            seeds=seeds,
            published_after=published_after,
            published_before=published_before,
            pages_per_query=pages_per_query,
            max_results=max_results,
            sleep_seconds=sleep_seconds,
        )

        raw_discovered += raw
        for item in found:
            vid = item["video_id"]
            discovered_by_horizon[horizon].add(vid)
            if vid not in all_candidates:
                all_candidates[vid] = item

    candidate_ids = list(all_candidates.keys())
    print(f"Candidates discovered (raw): {raw_discovered}")
    print(f"Candidates after dedupe: {len(candidate_ids)}")

    # Fetch video metadata
    all_meta = {}
    for batch in base.chunked(candidate_ids, 50):
        all_meta.update(fetch_metadata_for_ids(batch))
        time.sleep(sleep_seconds)

    print(f"Fetched metadata for {len(all_meta)} videos")

    # Fetch channel metadata
    channel_ids = {m.get("channel_id") for m in all_meta.values() if m.get("channel_id")}
    channels = fetch_channels_for_ids(channel_ids)

    # Build rows and assign horizons
    rows_to_write = []
    labeled_counts = {h: 0 for h in horizons}

    for vid, base_item in all_candidates.items():
        meta = all_meta.get(vid)
        if not meta:
            continue
        chan = channels.get(meta.get("channel_id", ""), {})
        row = base.build_row(base_item, meta, chan, captured_at_iso)
        age_days = row.get("age_days")
        view_count = row.get("view_count")

        if age_days is None or view_count is None:
            continue

        for horizon in horizons:
            if abs(age_days - horizon) <= tolerance_days:
                row_h = dict(row)
                row_h["horizon_days"] = int(horizon)
                row_h["horizon_view_count"] = int(view_count)
                row_h["horizon_label_type"] = "views_at_age"
                rows_to_write.append(row_h)
                labeled_counts[horizon] += 1

    conn = init_dedupe_db(Path(args.dedupe_db))

    # Filter out labels already seen
    filtered_rows = []
    for r in rows_to_write:
        if is_labeled(conn, r["video_id"], r["horizon_days"]):
            continue
        filtered_rows.append(r)

    # Prepare output
    fields = build_output_schema(Path(args.schema_reference), HORIZON_EXTRA_FIELDS)
    f, writer = prepare_output_writer(Path(args.output), fields)

    for r in filtered_rows:
        writer.writerow(r)
    f.close()

    record_labels(conn, filtered_rows)
    conn.close()

    # Logging summary
    for h in horizons:
        print(f"Labeled {h}d (pre-dedupe): {labeled_counts[h]}")
    print(f"Rows written (post-dedupe): {len(filtered_rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
