import csv
from pathlib import Path

# Fallback schema (matches Data/raw/Metadata/shorts_metadata.csv header)
DEFAULT_SCHEMA = [
    "video_id",
    "url",
    "query",
    "category_type",
    "collector_version",
    "captured_at",
    "channel_id",
    "channel_title",
    "channel_custom_url",
    "channel_country",
    "channel_created_at",
    "channel_description_length",
    "channel_subscriber_count",
    "channel_video_count",
    "channel_view_count",
    "channel_hidden_subscriber_count",
    "channel_age_days",
    "title",
    "description",
    "title_length",
    "description_length",
    "emoji_count",
    "has_hashtags",
    "has_shorts_hashtag",
    "has_clickbait_words",
    "hashtag_count",
    "published_at",
    "duration",
    "duration_seconds",
    "default_language",
    "default_audio_language",
    "dimension",
    "definition",
    "projection",
    "caption_available",
    "licensed_content",
    "thumbnail_url",
    "thumb_width",
    "thumb_height",
    "thumb_aspect_ratio",
    "is_vertical_thumb",
    "view_count",
    "like_count",
    "comment_count",
    "age_days",
    "views_per_day",
    "likes_per_day",
    "comments_per_day",
    "likes_per_view",
    "comments_per_view",
    "views_per_hour",
    "shorts_by_duration",
    "shorts_by_hashtag",
    "is_short",
    "topic_categories",
    "privacy_status",
    "upload_status",
    "embeddable",
    "license",
    "madeForKids",
    "publicStatsViewable",
    "likes_hidden",
    "comments_disabled",
    "stats_hidden",
    "too_new_for_rates",
    "virality_score",
]


def load_reference_schema(reference_csv: Path) -> list:
    if reference_csv and reference_csv.exists():
        with reference_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                return [h.strip() for h in header if h.strip()]
    return list(DEFAULT_SCHEMA)


def build_output_schema(reference_csv: Path, extra_fields: list) -> list:
    fields = load_reference_schema(reference_csv)
    for f in extra_fields:
        if f not in fields:
            fields.append(f)
    return fields
