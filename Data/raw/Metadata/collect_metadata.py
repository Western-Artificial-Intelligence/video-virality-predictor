"""Compatibility helpers for daily_horizon_collector.

This module provides the subset of constants/functions used by
Data/raw/Metadata/daily_horizon_collector.py.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Tuple

try:
    from credentials import YOUTUBE_API_KEY  # type: ignore
except Exception:
    YOUTUBE_API_KEY = ""

COLLECTOR_VERSION = os.environ.get("COLLECTOR_VERSION", "v0.4.0-horizon")

YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def chunked(values: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def iso8601_to_seconds(value: str) -> int:
    if not value:
        return 0
    # Supports common YouTube ISO8601 durations like PT54S, PT2M13S, PT1H3M2S
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s


def parse_thumb_metrics(snippet: Dict) -> Tuple[str, int, int, float, int]:
    thumbs = (snippet or {}).get("thumbnails") or {}
    best = None
    best_area = -1
    for _, t in thumbs.items():
        w = int(t.get("width") or 0)
        h = int(t.get("height") or 0)
        area = w * h
        if area > best_area:
            best_area = area
            best = t

    if not best:
        return "", 0, 0, 0.0, 0

    url = best.get("url") or ""
    w = int(best.get("width") or 0)
    h = int(best.get("height") or 0)
    ratio = (w / h) if w > 0 and h > 0 else 0.0
    is_vertical = 1 if h > w else 0
    return url, w, h, ratio, is_vertical


def _to_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def build_row(base_item: Dict, meta: Dict, chan: Dict, captured_at_iso: str) -> Dict:
    snippet = meta.get("snippet") or {}
    stats = meta.get("statistics") or {}

    view_count = _to_int(stats.get("viewCount"))
    like_count = _to_int(stats.get("likeCount"))
    comment_count = _to_int(stats.get("commentCount"))

    age_days = _to_float(base_item.get("age_days"))
    age_days_safe = age_days if age_days > 0 else 0.0
    age_hours_safe = age_days_safe * 24 if age_days_safe > 0 else 0.0

    chan_subs = _to_int(chan.get("subscriberCount"))
    chan_videos = _to_int(chan.get("videoCount"))
    chan_views = _to_int(chan.get("viewCount"))

    row = {
        "video_id": base_item.get("video_id") or meta.get("id") or "",
        "url": base_item.get("url") or "",
        "query": base_item.get("query") or "",
        "category_type": base_item.get("category_type") or "",
        "collector_version": COLLECTOR_VERSION,
        "captured_at": captured_at_iso,
        "channel_id": snippet.get("channelId") or base_item.get("channel_id") or "",
        "published_at": snippet.get("publishedAt") or base_item.get("published_at") or "",
        "view_count": view_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "age_days": age_days,
        "views_per_day": (view_count / age_days_safe) if age_days_safe > 0 else 0.0,
        "views_per_hour": (view_count / age_hours_safe) if age_hours_safe > 0 else 0.0,
        "likes_per_day": (like_count / age_days_safe) if age_days_safe > 0 else 0.0,
        "comments_per_day": (comment_count / age_days_safe) if age_days_safe > 0 else 0.0,
        "likes_per_view": (like_count / view_count) if view_count > 0 else 0.0,
        "comments_per_view": (comment_count / view_count) if view_count > 0 else 0.0,
        "channel_subscriber_count": chan_subs,
        "channel_video_count": chan_videos,
        "channel_view_count": chan_views,
        "channel_created_at": chan.get("publishedAt") or "",
        "channel_country": chan.get("country") or "",
        "channel_description_length": len((chan.get("description") or "")),
    }
    return row
