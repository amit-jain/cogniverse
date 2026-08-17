"""Topic extraction uses descriptive fields before identifier-like titles."""

import json
from pathlib import Path

import pytest

from cogniverse_synthetic.generators.base import (
    CANONICAL_TOPIC_FIELDS,
    extract_topic,
    is_identifier_topic,
    is_non_speech_annotation,
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
DOCUMENT_TOPIC = "v_-6dz6tBH77I.txt"
DOCUMENT_BODY = (
    "\ufeffThe video is of a man in athletic clothes standing in a net. He is "
    "holding a disk in his right hand. There are people sitting and standing "
    "right outside the net. There are also several people sitting on bleachers. "
    "It is sunny. There are several hills visible in the distance. The man spins "
    "a few times and throws the disk from the open end of the net. He looks "
    "toward the direction where he threw the disk. People applaude"
)
LIVE_DOCUMENT_SAMPLE = {
    "topic": DOCUMENT_TOPIC,
    "description": DOCUMENT_BODY,
    "start_time": 0.0,
    "end_time": 0.0,
    "video_id": "",
    "segment_id": 0,
    "creation_timestamp": 1786726986553,
    "schema_name": "document_text",
    "profile_name": "document_text_semantic",
    "embedding_type": "multi_vector",
    "profile_type": "document",
    "modality": "DOCUMENT",
    "profile_metadata": {
        "schema_name": "document_text",
        "embedding_model": "lightonai/LateOn",
        "embedding_type": "multi_vector",
        "type": "document",
    },
}


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


def test_identifier_predicate_verdict_for_every_shipped_id_form():
    verdicts = {
        value: is_identifier_topic(value)
        for value in [
            HASH_VALUE,
            f"{HASH_VALUE}_seg_7",
            "550e8400-e29b-41d4-a716-446655440000",
            "v_-6dz6tBH77I",
            "v_-6dz6tBH77I.txt",
            "yt_dQw4w9WgXcQ",
            "doc_7f3a91",
            "IMG_20240113_154522",
            "IMG_20240113_154522.jpg",
            "sha256:9f86d081884c7d65",
            "The video is of",
            "animal rodeo",
            "t-shirt",
            "COVID-19",
            "Apollo-11",
            "report.pdf",
            "transformer attention mechanism",
        ]
    }

    assert verdicts == {
        HASH_VALUE: True,
        f"{HASH_VALUE}_seg_7": True,
        "550e8400-e29b-41d4-a716-446655440000": True,
        "v_-6dz6tBH77I": True,
        "v_-6dz6tBH77I.txt": True,
        "yt_dQw4w9WgXcQ": True,
        "doc_7f3a91": True,
        "IMG_20240113_154522": True,
        "IMG_20240113_154522.jpg": True,
        "sha256:9f86d081884c7d65": True,
        "The video is of": False,
        "animal rodeo": False,
        "t-shirt": False,
        "COVID-19": False,
        "Apollo-11": False,
        "report.pdf": False,
        "transformer attention mechanism": False,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("*Screaming*", True),
        ("[Music]", True),
        ("(applause)", True),
        ("[Music] (applause) *Screaming*", True),
        ("[Music] hello there", False),
        ("hello [Music] there", False),
        ("hello there", False),
    ],
)
def test_non_speech_annotation_predicate_detects_only_annotation_tokens(
    value, expected
):
    assert is_non_speech_annotation(value) is expected


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
    item = LIVE_DOCUMENT_SAMPLE

    assert extract_topic(item, max_words=4) == "The video is of"


def test_extract_topic_rejects_annotation_only_transcript():
    item = {"audio_transcript": "*Screaming*"}

    assert extract_topic(item) is None


def test_extract_topic_preserves_mixed_transcript_annotations():
    item = {"audio_transcript": "[Music] hello there"}

    assert extract_topic(item) == "[Music] hello there"


def test_extract_topic_refuses_metadata_only_fallback():
    item = {
        "tenant_id": "tenant-123",
        "org_id": "org-456",
        "org_name": "org-name",
        "status": "active",
        "config_id": "cfg-789",
        "config_key": "query_enhancement",
        "scope": "tenant",
        "service": "optimizer",
        "adapter_id": "adapter-1",
        "derivation_kind": "derived",
        "written_by": "system",
        "tenant_full_id": "tenant-123:prod",
        "tenant_name": "tenant-name",
        "signature": "sig-1",
        "name": "metadata-only",
        "agent_type": "query_enhancement",
    }

    assert extract_topic(item) is None


def test_canonical_topic_fields_cover_all_shipped_schema_text_roles():
    canonical_fields = set(CANONICAL_TOPIC_FIELDS)
    assert canonical_fields.isdisjoint(
        {
            "tenant_id",
            "org_id",
            "org_name",
            "status",
            "config_id",
            "config_key",
            "scope",
            "service",
            "adapter_id",
            "derivation_kind",
            "written_by",
            "tenant_full_id",
            "tenant_name",
            "signature",
            "name",
            "agent_type",
        }
    )
    for schema_path in sorted(Path("configs/schemas").glob("*_schema.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        document_mapping = schema.get("document_mapping") or {}
        missing_fields = [
            field_name
            for field_name in (
                document_mapping.get("title"),
                document_mapping.get("description"),
                document_mapping.get("content"),
                document_mapping.get("text_content"),
                document_mapping.get("transcript"),
            )
            if isinstance(field_name, str) and field_name not in canonical_fields
        ]
        assert missing_fields == [], schema_path.name
