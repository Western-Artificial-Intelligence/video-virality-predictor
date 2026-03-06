from app.metadata import FieldDescriptor, canonicalize_metadata, merge_field_maps


def test_canonicalize_metadata_coerces_types_and_defaults():
    fields = {
        "duration_seconds": FieldDescriptor(name="duration_seconds", type="number", default=None),
        "has_hashtags": FieldDescriptor(name="has_hashtags", type="boolean", default=None),
        "channel_country": FieldDescriptor(name="channel_country", type="string", default="US"),
        "title": FieldDescriptor(name="title", type="string", default=""),
        "description": FieldDescriptor(name="description", type="string", default=""),
        "query": FieldDescriptor(name="query", type="string", default=""),
    }

    out = canonicalize_metadata(
        {
            "duration_seconds": "31.5",
            "has_hashtags": "true",
            "channel_country": "CA",
            "title": "hello",
        },
        fields,
    )

    assert out["duration_seconds"] == 31.5
    assert out["has_hashtags"] == 1
    assert out["channel_country"] == "CA"
    assert out["title"] == "hello"
    assert out["description"] == ""
    assert out["query"] == ""


def test_merge_field_maps_keeps_core_text_fields():
    fmap = {
        "duration_seconds": FieldDescriptor(name="duration_seconds", type="number"),
    }

    merged = merge_field_maps([fmap])

    assert "title" in merged
    assert "description" in merged
    assert "query" in merged
    assert merged["duration_seconds"].type == "number"
