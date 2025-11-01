# Data/Metadata/collect_metadata.py
import os, re, csv, time, requests
from pathlib import Path

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add repo root to import path

# Load API key (from credentials.py or env var)
try:
    from credentials import YOUTUBE_API_KEY
except Exception:
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

if not YOUTUBE_API_KEY:
    raise SystemExit("ERROR: Missing YOUTUBE_API_KEY. Add credentials.py at repo root or set env var.")

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "Data" / "Links" / "shorts_data" / "shorts_links_wide.csv"
OUTPUT_CSV = REPO_ROOT / "Data" / "Metadata" / "shorts_metadata.csv"

YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})")

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

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

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

def fetch_metadata_for_ids(ids):
    params = {
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

        out[vid] = {
            "title": sn.get("title", ""),
            "description": sn.get("description", "") or "",
            "tags": "|".join(sn.get("tags", [])) if sn.get("tags") else "",
            "channel_title": sn.get("channelTitle", ""),
            "channel_id": sn.get("channelId", ""),
            "published_at": sn.get("publishedAt", ""),
            "duration": duration_iso,
            "duration_seconds": duration_seconds,
            "default_audio_language": sn.get("defaultAudioLanguage", "") or sn.get("defaultLanguage", ""),
            "caption_available": str(cd.get("caption", "")).lower() == "true",
            "view_count": int(st["viewCount"]) if "viewCount" in st else None,
            "like_count": int(st["likeCount"]) if "likeCount" in st else None,
            "comment_count": int(st["commentCount"]) if "commentCount" in st else None,
            "topic_categories": "|".join(td.get("topicCategories", [])) if td.get("topicCategories") else "",
            "embeddable": status.get("embeddable"),
            "license": status.get("license"),
            "madeForKids": status.get("madeForKids"),
            "publicStatsViewable": status.get("publicStatsViewable"),
            
        }
    return out

def derive_is_short(title: str, description: str, duration_seconds: int) -> bool:
    text = f"{title} {description}".lower()
    return (duration_seconds <= 60) or ("#short" in text or "#shorts" in text)

def main():
    print(f"Reading links from: {INPUT_CSV}")
    rows = read_links(INPUT_CSV)

    # Normalize & dedupe
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

    all_meta = {}
    for batch in chunked(ids, 50):
        all_meta.update(fetch_metadata_for_ids(batch))
        time.sleep(0.2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id","url","query","category_type",
        "title","channel_title","channel_id","published_at",
        "duration","duration_seconds","default_audio_language","caption_available",
        "tags","description",
        "view_count","like_count","comment_count",
        "topic_categories","is_short",
        "embeddable","license","madeForKids","publicStatsViewable"
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for base in items:
            vid = base["video_id"]; meta = all_meta.get(vid, {})
            title = meta.get("title",""); desc = meta.get("description",""); dur_s = meta.get("duration_seconds",0)
            w.writerow({
                **base,
                "title": title,
                "channel_title": meta.get("channel_title",""),
                "channel_id": meta.get("channel_id",""),
                "published_at": meta.get("published_at",""),
                "duration": meta.get("duration",""),
                "duration_seconds": dur_s,
                "default_audio_language": meta.get("default_audio_language",""),
                "caption_available": meta.get("caption_available", False),
                "tags": meta.get("tags",""),
                "description": desc,
                "view_count": meta.get("view_count"),
                "like_count": meta.get("like_count"),
                "comment_count": meta.get("comment_count"),
                "topic_categories": meta.get("topic_categories",""),
                "is_short": derive_is_short(title, desc, dur_s),
		        "embeddable": meta.get("embeddable"),
		        "license": meta.get("license",""),
		        "madeForKids": meta.get("madeForKids"),
		        "publicStatsViewable": meta.get("publicStatsViewable"),
		

            })

    print(f"Saved metadata → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
