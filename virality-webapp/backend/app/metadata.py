from __future__ import annotations

from dataclasses import dataclass
from typing import Any


KNOWN_BOOLEAN_FIELDS = {
    "has_hashtags",
    "has_shorts_hashtag",
    "has_clickbait_words",
    "caption_available",
    "is_vertical_thumb",
    "shorts_by_duration",
    "shorts_by_hashtag",
    "is_short",
    "licensed_content",
    "embeddable",
    "madeForKids",
    "publicStatsViewable",
    "likes_hidden",
    "comments_disabled",
    "stats_hidden",
    "channel_hidden_subscriber_count",
    "too_new_for_rates",
}


@dataclass
class FieldDescriptor:
    name: str
    type: str
    required: bool = False
    default: Any = None
    options: list[str] | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "options": self.options,
            "description": self.description,
        }


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def coerce_value(value: Any, field_type: str) -> Any:
    if field_type == "boolean":
        parsed = parse_bool(value)
        if parsed is None:
            return None
        return 1 if parsed else 0
    if field_type == "number":
        return parse_number(value)
    return parse_string(value)


def canonicalize_metadata(raw_metadata: dict[str, Any], fields: dict[str, FieldDescriptor]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Normalize all known schema keys.
    for name, descriptor in fields.items():
        raw_value = raw_metadata.get(name, descriptor.default)
        coerced = coerce_value(raw_value, descriptor.type)
        if coerced is None and descriptor.default is not None:
            coerced = coerce_value(descriptor.default, descriptor.type)
        out[name] = coerced

    # Preserve additional keys as strings for traceability/debugging.
    for name, value in raw_metadata.items():
        if name in out:
            continue
        out[name] = parse_string(value)

    # Text fields are always present for meta-text construction.
    for text_key in ("title", "description", "query"):
        val = out.get(text_key)
        out[text_key] = "" if val is None else str(val)

    return out


def merge_field_maps(field_maps: list[dict[str, FieldDescriptor]]) -> dict[str, FieldDescriptor]:
    merged: dict[str, FieldDescriptor] = {}
    for fmap in field_maps:
        for name, desc in fmap.items():
            if name not in merged:
                merged[name] = FieldDescriptor(
                    name=name,
                    type=desc.type,
                    required=desc.required,
                    default=desc.default,
                    options=list(desc.options) if desc.options else None,
                    description=desc.description,
                )
                continue

            current = merged[name]
            if current.type != desc.type:
                # Number can safely absorb boolean/string by coercion if needed.
                if current.type == "number" or desc.type == "number":
                    current.type = "number"
                elif current.type == "string" or desc.type == "string":
                    current.type = "string"
                else:
                    current.type = "string"

            current.required = current.required or desc.required
            if current.default is None and desc.default is not None:
                current.default = desc.default

            if current.options is None or desc.options is None:
                current.options = None
            else:
                joined = sorted(set(current.options) | set(desc.options))
                current.options = joined if len(joined) <= 100 else None

    # Ensure core text fields are present.
    for key, description in (
        ("title", "Video title"),
        ("description", "Video description"),
        ("query", "Seed/query label used during collection"),
    ):
        if key not in merged:
            merged[key] = FieldDescriptor(name=key, type="string", required=False, default="", description=description)

    preferred_order = [
        "title",
        "description",
        "query",
        "duration_seconds",
        "channel_country",
        "default_language",
        "default_audio_language",
        "published_hour",
        "published_dayofweek",
    ]

    def _sort_key(item: tuple[str, FieldDescriptor]) -> tuple[int, str]:
        name = item[0]
        if name in preferred_order:
            return (preferred_order.index(name), name)
        return (len(preferred_order) + 1, name)

    ordered = dict(sorted(merged.items(), key=_sort_key))
    return ordered
