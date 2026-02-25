"""Compatibility helpers for daily_horizon_collector.

This module provides the subset of constants/functions used by
Data/raw/Metadata/daily_horizon_collector.py.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

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


def _to_optional_int(v) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _safe_div(num, den) -> Optional[float]:
    try:
        if num is None or den in (None, 0):
            return None
        return float(num) / float(den)
    except Exception:
        return None


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _infer_age_days(base_item: Dict, snippet: Dict, meta: Dict, captured_at_iso: str) -> float:
    age_days = _to_float(base_item.get("age_days"))
    if age_days > 0:
        return age_days

    published_at = (
        (snippet or {}).get("publishedAt")
        or meta.get("published_at")
        or base_item.get("published_at")
        or ""
    )
    published_dt = _parse_iso_datetime(published_at)
    captured_dt = _parse_iso_datetime(captured_at_iso)
    if not published_dt or not captured_dt:
        return 0.0

    delta_seconds = (captured_dt - published_dt).total_seconds()
    return (delta_seconds / 86400.0) if delta_seconds > 0 else 0.0


EMOJI_RE = re.compile(
    "["
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
    "]",
    flags=re.UNICODE,
)


CLICKBAIT_WORDS = [
    "shocking",
    "insane",
    "unbelievable",
    "you won’t believe",
    "you won't believe",
    "must see",
    "crazy",
    "epic",
    "gone wrong",
    "wow",
    "no way",
    "wild",
]


def _count_emojis(text: str) -> int:
    if not text:
        return 0
    return len(EMOJI_RE.findall(text))


def _has_hashtags(text: str) -> bool:
    return bool(re.search(r"#\w+", text or "", flags=re.IGNORECASE))


def _hashtag_count(text: str) -> int:
    return len(re.findall(r"#\w+", text or "", flags=re.IGNORECASE))


def _has_shorts_hashtag(text: str) -> bool:
    return bool(re.search(r"#shorts?\b", text or "", flags=re.IGNORECASE))


def _has_clickbait(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in CLICKBAIT_WORDS)


def build_row(base_item: Dict, meta: Dict, chan: Dict, captured_at_iso: str) -> Dict:
    snippet = meta.get("snippet") or {}
    content = meta.get("contentDetails") or {}
    stats = meta.get("statistics") or {}
    status = meta.get("status") or {}

    title = (meta.get("title") or snippet.get("title") or "").strip()
    description = (meta.get("description") or snippet.get("description") or "").strip()
    text_both = f"{title}\n{description}"

    # Support both raw YouTube API shape and normalized collector shape.
    view_count = _to_optional_int(
        stats.get("viewCount") if "viewCount" in stats else meta.get("view_count")
    )
    like_count = _to_optional_int(stats.get("likeCount")) if "likeCount" in stats else _to_optional_int(meta.get("like_count"))
    comment_count = (
        _to_optional_int(stats.get("commentCount"))
        if "commentCount" in stats
        else _to_optional_int(meta.get("comment_count"))
    )
    has_like = ("likeCount" in stats) if stats else (like_count is not None)
    has_comment = ("commentCount" in stats) if stats else (comment_count is not None)

    age_days = _infer_age_days(base_item, snippet, meta, captured_at_iso)
    age_days_safe = age_days if age_days > 0 else None
    age_hours_safe = (age_days * 24.0) if age_days > 0 else None

    channel_created_at = chan.get("channel_created_at") or chan.get("publishedAt") or ""
    captured_dt = _parse_iso_datetime(captured_at_iso)
    channel_created_dt = _parse_iso_datetime(channel_created_at)
    if channel_created_dt and captured_dt:
        channel_age_days = max(1.0, (captured_dt - channel_created_dt).total_seconds() / 86400.0)
    else:
        channel_age_days = None

    chan_subs = (
        _to_optional_int(chan.get("subscriberCount"))
        if "subscriberCount" in chan
        else _to_optional_int(chan.get("channel_subscriber_count"))
    )
    chan_videos = (
        _to_optional_int(chan.get("videoCount"))
        if "videoCount" in chan
        else _to_optional_int(chan.get("channel_video_count"))
    )
    chan_views = (
        _to_optional_int(chan.get("viewCount"))
        if "viewCount" in chan
        else _to_optional_int(chan.get("channel_view_count"))
    )

    channel_description_length = chan.get("channel_description_length")
    if channel_description_length is None:
        channel_description_length = len(chan.get("description") or "")
    else:
        channel_description_length = _to_int(channel_description_length)

    duration_iso = meta.get("duration") or content.get("duration") or ""
    duration_seconds = _to_optional_int(meta.get("duration_seconds"))
    if duration_seconds is None:
        duration_seconds = iso8601_to_seconds(duration_iso)

    thumb_url = meta.get("thumbnail_url") or ""
    thumb_width = meta.get("thumb_width")
    thumb_height = meta.get("thumb_height")
    thumb_aspect_ratio = meta.get("thumb_aspect_ratio")
    is_vertical_thumb = meta.get("is_vertical_thumb")
    if not thumb_url and snippet:
        thumb_url, thumb_width, thumb_height, thumb_aspect_ratio, is_vertical_thumb = parse_thumb_metrics(
            snippet
        )

    shorts_by_duration = bool(duration_seconds is not None and duration_seconds <= 62)
    shorts_by_hashtag = _has_shorts_hashtag(text_both)
    is_short = shorts_by_duration or shorts_by_hashtag

    published_at = (
        snippet.get("publishedAt")
        or meta.get("published_at")
        or base_item.get("published_at")
        or ""
    )

    views_per_day = _safe_div(view_count, age_days_safe)
    views_per_hour = _safe_div(view_count, age_hours_safe)
    likes_per_day = _safe_div(like_count, age_days_safe)
    comments_per_day = _safe_div(comment_count, age_days_safe)
    likes_per_view = _safe_div(like_count, view_count)
    comments_per_view = _safe_div(comment_count, view_count)

    too_new_for_rates = bool(age_days_safe is not None and age_days_safe < 1.0)
    virality_score = None
    if views_per_day is not None and chan_subs not in (None, 0):
        try:
            virality_score = math.log10(max(1e-9, views_per_day / chan_subs))
        except Exception:
            virality_score = None

    public_stats_viewable = meta.get("publicStatsViewable")
    if public_stats_viewable is None:
        public_stats_viewable = status.get("publicStatsViewable")

    row = {
        "video_id": base_item.get("video_id") or meta.get("id") or "",
        "url": base_item.get("url") or "",
        "query": base_item.get("query") or "",
        "category_type": base_item.get("category_type") or "",
        "collector_version": COLLECTOR_VERSION,
        "captured_at": captured_at_iso,
        "channel_id": (
            snippet.get("channelId")
            or meta.get("channel_id")
            or base_item.get("channel_id")
            or ""
        ),
        "channel_title": snippet.get("channelTitle") or meta.get("channel_title") or "",
        "channel_custom_url": chan.get("channel_custom_url") or chan.get("customUrl") or "",
        "channel_country": chan.get("channel_country") or chan.get("country") or "",
        "channel_created_at": channel_created_at,
        "channel_description_length": channel_description_length,
        "channel_subscriber_count": chan_subs,
        "channel_video_count": chan_videos,
        "channel_view_count": chan_views,
        "channel_hidden_subscriber_count": bool(
            chan.get("channel_hidden_subscriber_count")
            if chan.get("channel_hidden_subscriber_count") is not None
            else chan.get("hiddenSubscriberCount")
        ),
        "channel_age_days": channel_age_days,
        "title": title,
        "description": description,
        "title_length": len(title),
        "description_length": len(description),
        "emoji_count": _count_emojis(text_both),
        "has_hashtags": _has_hashtags(text_both),
        "has_shorts_hashtag": shorts_by_hashtag,
        "has_clickbait_words": _has_clickbait(text_both),
        "hashtag_count": _hashtag_count(text_both),
        "published_at": published_at,
        "duration": duration_iso,
        "duration_seconds": duration_seconds,
        "default_language": meta.get("default_language") or snippet.get("defaultLanguage") or "",
        "default_audio_language": meta.get("default_audio_language")
        or snippet.get("defaultAudioLanguage")
        or "",
        "dimension": meta.get("dimension") if meta.get("dimension") is not None else content.get("dimension"),
        "definition": meta.get("definition") if meta.get("definition") is not None else content.get("definition"),
        "projection": meta.get("projection") if meta.get("projection") is not None else content.get("projection"),
        "caption_available": (
            meta.get("caption_available")
            if meta.get("caption_available") is not None
            else str(content.get("caption", "")).lower() == "true"
        ),
        "licensed_content": (
            meta.get("licensed_content")
            if meta.get("licensed_content") is not None
            else content.get("licensedContent")
        ),
        "thumbnail_url": thumb_url,
        "thumb_width": _to_optional_int(thumb_width),
        "thumb_height": _to_optional_int(thumb_height),
        "thumb_aspect_ratio": _to_float(thumb_aspect_ratio)
        if thumb_aspect_ratio not in (None, "")
        else None,
        "is_vertical_thumb": (
            bool(is_vertical_thumb)
            if is_vertical_thumb not in (None, "")
            else None
        ),
        "view_count": view_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "age_days": age_days_safe,
        "views_per_day": views_per_day,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "likes_per_view": likes_per_view,
        "comments_per_view": comments_per_view,
        "views_per_hour": views_per_hour,
        "shorts_by_duration": shorts_by_duration,
        "shorts_by_hashtag": shorts_by_hashtag,
        "is_short": is_short,
        "topic_categories": meta.get("topic_categories") or "",
        "privacy_status": meta.get("privacy_status")
        if meta.get("privacy_status") is not None
        else status.get("privacyStatus"),
        "upload_status": meta.get("upload_status")
        if meta.get("upload_status") is not None
        else status.get("uploadStatus"),
        "embeddable": meta.get("embeddable")
        if meta.get("embeddable") is not None
        else status.get("embeddable"),
        "license": meta.get("license")
        if meta.get("license") is not None
        else status.get("license"),
        "madeForKids": meta.get("madeForKids")
        if meta.get("madeForKids") is not None
        else status.get("madeForKids"),
        "publicStatsViewable": public_stats_viewable,
        "likes_hidden": (
            bool(meta.get("likes_hidden"))
            if meta.get("likes_hidden") is not None
            else (not has_like)
        ),
        "comments_disabled": (
            bool(meta.get("comments_disabled"))
            if meta.get("comments_disabled") is not None
            else (not has_comment)
        ),
        "stats_hidden": (
            bool(meta.get("stats_hidden"))
            if meta.get("stats_hidden") is not None
            else (public_stats_viewable is False)
        ),
        "too_new_for_rates": too_new_for_rates,
        "virality_score": virality_score,
    }
    return row
