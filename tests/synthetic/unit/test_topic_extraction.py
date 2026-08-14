"""Topic extraction uses descriptive fields before identifier-like titles."""

import json
from pathlib import Path

import pytest

from cogniverse_synthetic.generators.base import (
    CANONICAL_TOPIC_FIELDS,
    extract_topic,
    is_content_hash_topic,
)
from cogniverse_synthetic.generators.profile import ProfileGenerator
from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)
from cogniverse_synthetic.generators.workflow import WorkflowGenerator

HASH_VALUE = "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"
RODEO_TEXT = (
    "This video frame captures an outdoor event, likely a rodeo or similar "
    "competition, viewed through a wire mesh fence..."
)


def _profile_topic(item):
    return ProfileGenerator()._extract_topic(item)


def _query_enhancement_topic(item):
    return QueryEnhancementGenerator._extract_topic(item)


def _workflow_topic(item):
    return WorkflowGenerator()._extract_topic(item)


@pytest.mark.parametrize(
    ("name", "extract_topic", "expected"),
    [
        ("profile", _profile_topic, RODEO_TEXT),
        ("query_enhancement", _query_enhancement_topic, "This video frame captures"),
        ("workflow", _workflow_topic, RODEO_TEXT),
    ],
)
def test_extract_topic_prefers_segment_description_over_video_title_hash(
    name, extract_topic, expected
):
    item = {
        "video_title": HASH_VALUE,
        "segment_description": RODEO_TEXT,
    }

    assert extract_topic(item) == expected, name


@pytest.mark.parametrize(
    ("name", "extract_topic", "expected"),
    [
        ("profile", _profile_topic, RODEO_TEXT),
        ("query_enhancement", _query_enhancement_topic, "This video frame captures"),
        ("workflow", _workflow_topic, RODEO_TEXT),
    ],
)
def test_extract_topic_rejects_bare_hash_and_uses_next_field(
    name, extract_topic, expected
):
    item = {
        "description": HASH_VALUE,
        "segment_description": RODEO_TEXT,
    }

    assert extract_topic(item) == expected, name


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (HASH_VALUE, True),
        (f"{HASH_VALUE}_seg_7", True),
        ("animal rodeo", False),
    ],
)
def test_content_hash_predicate_distinguishes_hashes_from_titles(value, expected):
    assert is_content_hash_topic(value) is expected


@pytest.mark.parametrize(
    ("extract_topic", "expected"),
    [
        (_profile_topic, None),
        (_query_enhancement_topic, None),
    ],
)
def test_extract_topic_returns_none_when_no_descriptive_field_exists(
    extract_topic, expected
):
    item = {"video_title": HASH_VALUE}

    assert extract_topic(item) == expected


def test_workflow_extract_topic_rejects_hash_only_item():
    with pytest.raises(
        ValueError, match="sampled workflow content requires a non-empty topic"
    ):
        _workflow_topic({"video_title": HASH_VALUE})


def test_extract_topic_uses_document_body_and_strips_bom():
    item = {
        "schema_name": "document_text",
        "document_title": "Annual report",
        "full_text": "\ufeffThe video is of people applaud in the arena",
    }

    assert extract_topic(item, max_words=4) == "The video is of"


def test_canonical_topic_fields_cover_all_shipped_schema_fieldsets():
    canonical_fields = set(CANONICAL_TOPIC_FIELDS)
    for schema_path in sorted(Path("configs/schemas").glob("*_schema.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        default_fieldset = next(
            fieldset
            for fieldset in schema["fieldsets"]
            if fieldset["name"] == "default"
        )
        missing_fields = [
            field_name
            for field_name in default_fieldset["fields"]
            if field_name not in canonical_fields
        ]
        assert missing_fields == [], schema_path.name
