# Data/Metadata/collect_metadata.py
import os, re, csv, time, math, json, requests
from datetime import datetime, timezone
from pathlib import Path
import sys

# ---------- repo import path ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add repo root to import path

# ---------- API key ----------
try:
    from credentials import YOUTUBE_API_KEY
except Exception:
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

if not YOUTUBE_API_KEY:
    raise SystemExit("ERROR: Missing YOUTUBE_API_KEY. Add credentials.py at repo root or set env var.")

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "Data" / "Links" / "shorts_data" / "shorts_links_wide.csv"
OUTPUT_CSV = REPO_ROOT / "Data" / "Metadata" / "shorts_metadata.csv"

# ---------- Config / tags ----------
YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})")
COLLECTOR_VERSION = os.getenv("COLLECTOR_VERSION", "v0.3.1")

# ---------- Helpers ----------
def extract_video_id(url: str):
    m = YOUTUBE_ID_RE.search(url or "")
    return m.group(1) if m else None

def iso8601_to_seconds(iso: str) -> int:
    if not iso or not iso.startswith("PT"):
        return 0
    h = m = s = 0
    H = re.search(r"(\d+)H", iso); M = re.search(r"(\d+)M", iso); S = re.search(r"(\d+)S", iso)
    if H: h = int(H.group(1))
    if M: m = int(M.group(1))
    if S: s = int(S.group(1))
    return h*3600 + m*60 + s

def parse_iso_datetime(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def safe_div(num, den):
    try:
        if num is None or den in (None, 0):
            return None
        return num / den
    except Exception:
        return None

EMOJI_RE = re.compile(
    "["  # selected emoji unicode blocks (broad but safe)
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]", flags=re.UNICODE
)

CLICKBAIT_WORDS = [
    "shocking", "insane", "unbelievable", "you won’t believe", "you won't believe",
    "must see", "crazy", "epic", "gone wrong", "wow", "no way", "wild"
]

def count_emojis(text: str) -> int:
    if not text:
        return 0
    return len(EMOJI_RE.findall(text))

def has_hashtags(text: str) -> bool:
    return bool(re.search(r"#\w+", text or "", flags=re.IGNORECASE))

def hashtag_count(text: str) -> int:
    return len(re.findall(r"#\w+", text or "", flags=re.IGNORECASE))

def has_shorts_hashtag(text: str) -> bool:
    return bool(re.search(r"#shorts?\b", text or "", flags=re.IGNORECASE))

def has_clickbait(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in CLICKBAIT_WORDS)

def parse_thumb_metrics(snippet: dict):
    thumbs = (snippet or {}).get("thumbnails", {}) or {}
    cand = None
    for key in ["maxres", "standard", "high", "medium", "default"]:
        if key in thumbs and isinstance(thumbs[key], dict):
            cand = thumbs[key]; break
    if not cand:
        return None, None, None, None, None
    w = cand.get("width"); h = cand.get("height")
    aspect = None; is_vertical = None
    try:
        if w and h:
            aspect = float(w)/float(h)
            is_vertical = aspect < 1.0
    except Exception:
        pass
    return cand.get("url"), w, h, aspect, is_vertical

# ---------- IO ----------
def read_links(path: Path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "query": (row.get("query") or "").strip(),
                "category_type": (row.get("category_type") or "").strip(),
                "url": (row.get("url") or "").strip(),
            })
    return out

# ---------- API fetch ----------
def fetch_metadata_for_ids(ids):
    params = {
        # liveStreamingDetails & recordingDetails removed
        "part": "snippet,contentDetails,statistics,topicDetails,status",
        "id": ",".join(ids),
        "key": YOUTUBE_API_KEY
    }
    resp = requests.get(YT_VIDEOS_URL, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"WARNING: videos.list failed ({resp.status_code}) for batch of {len(ids)}")
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
        duration_seconds = iso8601_to_seconds(duration_iso)

        thumb_url, tw, th, aspect, is_vertical = parse_thumb_metrics(sn)

        # edge-case flags (video-level)
        has_like = "likeCount" in st
        has_comment = "commentCount" in st
        public_stats_viewable = status.get("publicStatsViewable")

        out[vid] = {
            # snippet
            "title": sn.get("title", "") or "",
            "description": sn.get("description", "") or "",
            "tags": "|".join(sn.get("tags", [])) if sn.get("tags") else "",
            "channel_title": sn.get("channelTitle", "") or "",
            "channel_id": sn.get("channelId", "") or "",
            "published_at": sn.get("publishedAt", "") or "",
            "default_language": sn.get("defaultLanguage", "") or "",
            "default_audio_language": sn.get("defaultAudioLanguage", "") or "",
            # thumbnails proxy
            "thumbnail_url": thumb_url or "",
            "thumb_width": tw,
            "thumb_height": th,
            "thumb_aspect_ratio": aspect,
            "is_vertical_thumb": is_vertical,
            # contentDetails
            "duration": duration_iso,
            "duration_seconds": duration_seconds,
            "dimension": cd.get("dimension"),
            "definition": cd.get("definition"),
            "projection": cd.get("projection"),
            "caption_available": str(cd.get("caption", "")).lower() == "true",
            "licensed_content": cd.get("licensedContent"),
            # statistics (nullable)
            "view_count": int(st["viewCount"]) if "viewCount" in st else None,
            "like_count": int(st["likeCount"]) if has_like else None,
            "comment_count": int(st["commentCount"]) if has_comment else None,
            # topicDetails
            "topic_categories": "|".join(td.get("topicCategories", [])) if td.get("topicCategories") else "",
            # status
            "privacy_status": status.get("privacyStatus"),
            "upload_status": status.get("uploadStatus"),
            "embeddable": status.get("embeddable"),
            "license": status.get("license"),
            "madeForKids": status.get("madeForKids"),
            "publicStatsViewable": public_stats_viewable,
            # NEW: video-level edge-case flags
            "likes_hidden": not has_like,
            "comments_disabled": not has_comment,
            "stats_hidden": (public_stats_viewable is False),
        }
    return out

def fetch_channels_for_ids(channel_ids):
    out = {}
    if not channel_ids:
        return out
    for batch in chunked(list(channel_ids), 50):
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": YOUTUBE_API_KEY
        }
        resp = requests.get(YT_CHANNELS_URL, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"WARNING: channels.list failed ({resp.status_code}) for batch of {len(batch)}")
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
                # NEW: channel-level edge-case flag
                "channel_hidden_subscriber_count": bool(st.get("hiddenSubscriberCount")),
            }
    return out

# ---------- Feature engineering ----------
def shorts_diagnostics(title: str, description: str, duration_seconds: int) -> dict:
    text = f"{title} {description}".lower()
    by_dur = duration_seconds is not None and duration_seconds <= 62  # small tolerance
    by_hash = has_shorts_hashtag(text)
    return {
        "shorts_by_duration": bool(by_dur),
        "shorts_by_hashtag": bool(by_hash),
        "is_short": (by_dur or by_hash)
    }

def build_row(base, meta, chan, captured_at_iso: str):
    # basic fields
    title = meta.get("title","")
    desc = meta.get("description","")
    text_both = f"{title}\n{desc}"

    # times
    published_at = meta.get("published_at","")
    published_dt = parse_iso_datetime(published_at)
    captured_dt = parse_iso_datetime(captured_at_iso)

    # time deltas
    if published_dt and captured_dt:
        age_days = max(1.0, (captured_dt - published_dt).total_seconds() / 86400.0)
        views_per_day = safe_div(meta.get("view_count"), age_days)
        likes_per_day = safe_div(meta.get("like_count"), age_days)
        comments_per_day = safe_div(meta.get("comment_count"), age_days)
        views_per_hour = safe_div(meta.get("view_count"), age_days * 24.0)
    else:
        age_days = views_per_day = likes_per_day = comments_per_day = views_per_hour = None

    too_new_for_rates = (age_days is not None and age_days < 1.0)

    # engagement ratios
    likes_per_view = safe_div(meta.get("like_count"), meta.get("view_count"))
    comments_per_view = safe_div(meta.get("comment_count"), meta.get("view_count"))

    # text / hashtags / emoji
    title_length = len(title or "")
    description_length = len(desc or "")
    emoji_cnt = count_emojis(text_both)
    any_hashtags = has_hashtags(text_both)
    shorts_hash = has_shorts_hashtag(text_both)
    clickbait = has_clickbait(text_both)
    n_hashtags = hashtag_count(text_both)

    # shorts diagnostics
    diag = shorts_diagnostics(title, desc, meta.get("duration_seconds", 0) or 0)

    # channel context
    channel_created_dt = parse_iso_datetime((chan or {}).get("channel_created_at", ""))
    if channel_created_dt and captured_dt:
        channel_age_days = max(1.0, (captured_dt - channel_created_dt).total_seconds()/86400.0)
    else:
        channel_age_days = None

    # optional normalized virality proxy
    virality_score = None
    subs = (chan or {}).get("channel_subscriber_count")
    if views_per_day is not None and subs not in (None, 0):
        try:
            virality_score = math.log10(max(1e-9, views_per_day / subs))
        except Exception:
            virality_score = None

    return {
        # identity / source
        "video_id": base["video_id"],
        "url": base["url"],
        "query": base.get("query",""),
        "category_type": base.get("category_type",""),
        "collector_version": COLLECTOR_VERSION,
        "captured_at": captured_at_iso,

        # channel info
        "channel_id": meta.get("channel_id",""),
        "channel_title": meta.get("channel_title",""),
        "channel_custom_url": (chan or {}).get("channel_custom_url",""),
        "channel_country": (chan or {}).get("channel_country",""),
        "channel_created_at": (chan or {}).get("channel_created_at",""),
        "channel_description_length": (chan or {}).get("channel_description_length"),
        "channel_subscriber_count": (chan or {}).get("channel_subscriber_count"),
        "channel_video_count": (chan or {}).get("channel_video_count"),
        "channel_view_count": (chan or {}).get("channel_view_count"),
        "channel_hidden_subscriber_count": (chan or {}).get("channel_hidden_subscriber_count"),
        "channel_age_days": channel_age_days,

        # video text / nlp proxies
        "title": title,
        "description": desc,
        "title_length": title_length,
        "description_length": description_length,
        "emoji_count": emoji_cnt,
        "has_hashtags": any_hashtags,
        "has_shorts_hashtag": shorts_hash,
        "has_clickbait_words": clickbait,
        "hashtag_count": n_hashtags,

        # video attributes / format
        "published_at": published_at,
        "duration": meta.get("duration",""),
        "duration_seconds": meta.get("duration_seconds"),
        "default_language": meta.get("default_language",""),
        "default_audio_language": meta.get("default_audio_language",""),
        "dimension": meta.get("dimension"),
        "definition": meta.get("definition"),
        "projection": meta.get("projection"),
        "caption_available": meta.get("caption_available"),
        "licensed_content": meta.get("licensed_content"),
        "thumbnail_url": meta.get("thumbnail_url",""),
        "thumb_width": meta.get("thumb_width"),
        "thumb_height": meta.get("thumb_height"),
        "thumb_aspect_ratio": meta.get("thumb_aspect_ratio"),
        "is_vertical_thumb": meta.get("is_vertical_thumb"),

        # engagement (absolute + derived)
        "view_count": meta.get("view_count"),
        "like_count": meta.get("like_count"),
        "comment_count": meta.get("comment_count"),
        "age_days": age_days,
        "views_per_day": views_per_day,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "likes_per_view": likes_per_view,
        "comments_per_view": comments_per_view,
        "views_per_hour": views_per_hour,

        # shorts diagnostics
        "shorts_by_duration": diag["shorts_by_duration"],
        "shorts_by_hashtag": diag["shorts_by_hashtag"],
        "is_short": diag["is_short"],

        # topics & discovery
        "topic_categories": meta.get("topic_categories",""),

        # status / policy + edge-case flags
        "privacy_status": meta.get("privacy_status"),
        "upload_status": meta.get("upload_status"),
        "embeddable": meta.get("embeddable"),
        "license": meta.get("license",""),
        "madeForKids": meta.get("madeForKids"),
        "publicStatsViewable": meta.get("publicStatsViewable"),
        "likes_hidden": meta.get("likes_hidden"),
        "comments_disabled": meta.get("comments_disabled"),
        "stats_hidden": meta.get("stats_hidden"),
        "too_new_for_rates": too_new_for_rates,

        # optional modeling target
        "virality_score": virality_score,
    }

# ---------- Main ----------
def main():
    print(f"Reading links from: {INPUT_CSV}")
    rows = read_links(INPUT_CSV)

    # Normalize & dedupe by video_id
    seen = {}
    for r in rows:
        vid = extract_video_id(r["url"])
        if not vid:
            continue
        if vid not in seen:
            seen[vid] = {
                "video_id": vid,
                "url": r["url"],
                "query": r.get("query", ""),
                "category_type": r.get("category_type", ""),
            }

    items = list(seen.values())
    ids = [x["video_id"] for x in items]
    print(f"Found {len(ids)} unique video IDs.")

    # Fetch video metadata in batches
    all_meta = {}
    for batch in chunked(ids, 50):
        all_meta.update(fetch_metadata_for_ids(batch))
        time.sleep(0.2)

    # Fetch channel metadata (unique channel IDs)
    channel_ids = {m.get("channel_id") for m in all_meta.values() if m.get("channel_id")}
    channels = fetch_channels_for_ids(channel_ids)

    # Output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        # identity & source
        "video_id","url","query","category_type","collector_version","captured_at",

        # channel
        "channel_id","channel_title","channel_custom_url","channel_country","channel_created_at",
        "channel_description_length","channel_subscriber_count","channel_video_count","channel_view_count",
        "channel_hidden_subscriber_count","channel_age_days",

        # text / nlp proxies
        "title","description","title_length","description_length","emoji_count",
        "has_hashtags","has_shorts_hashtag","has_clickbait_words","hashtag_count",

        # video attrs
        "published_at","duration","duration_seconds","default_language","default_audio_language",
        "dimension","definition","projection","caption_available","licensed_content",
        "thumbnail_url","thumb_width","thumb_height","thumb_aspect_ratio","is_vertical_thumb",

        # engagement
        "view_count","like_count","comment_count","age_days","views_per_day","likes_per_day",
        "comments_per_day","likes_per_view","comments_per_view","views_per_hour",

        # shorts diagnostics
        "shorts_by_duration","shorts_by_hashtag","is_short",

        # topics & discovery
        "topic_categories",

        # status / policy + edge-case flags
        "privacy_status","upload_status","embeddable","license","madeForKids","publicStatsViewable",
        "likes_hidden","comments_disabled","stats_hidden","too_new_for_rates",

        # optional modeling target
        "virality_score",
    ]

    captured_at_iso = now_utc().isoformat()

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        for base in items:
            vid = base["video_id"]
            meta = all_meta.get(vid, {})
            chan = channels.get(meta.get("channel_id",""), {})
            row = build_row(base, meta, chan, captured_at_iso)
            w.writerow(row)

    print(f"Saved metadata → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
