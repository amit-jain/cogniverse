"""Exact content and timestamp behavior for synthetic pattern extraction."""

from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_synthetic.utils.pattern_extraction import PatternExtractor

pytestmark = pytest.mark.unit


def test_normalized_semantic_fields_take_part_in_pattern_extraction() -> None:
    patterns = PatternExtractor().extract(
        [
            {
                "topic": "Quantum Robotics",
                "description": "TensorFlow Navigation Tutorial",
                "transcript": "PyTorch Motion Planning",
            }
        ]
    )

    assert "quantum robotics" in patterns["topics"]
    assert {
        "Quantum Robotics",
        "TensorFlow",
        "Navigation Tutorial",
        "PyTorch",
        "Motion Planning",
    } <= set(patterns["entities"])
    assert patterns["content_types"] == ["tutorial"]


def test_sentence_leading_stopword_is_not_a_named_entity() -> None:
    entities = PatternExtractor().extract_entities(
        [{"topic": "In 2026, Marie Curie met Pierre Curie."}]
    )

    assert entities == [
        "Curie",
        "Marie",
        "Marie Curie",
        "Pierre",
        "Pierre Curie",
    ]


def test_millisecond_timestamp_is_interpreted_as_milliseconds() -> None:
    old_date = datetime.now(timezone.utc) - timedelta(days=400)
    timestamp_ms = int(old_date.timestamp() * 1000)

    temporal = PatternExtractor().extract_temporal_patterns(
        [{"creation_timestamp": timestamp_ms}]
    )

    assert temporal == [f"from {old_date.year}"]
    assert "recent" not in temporal
    assert "latest" not in temporal


def test_timezone_aware_timestamp_uses_utc_recency_window() -> None:
    recent_date = datetime.now(timezone.utc) - timedelta(days=5)

    temporal = PatternExtractor().extract_temporal_patterns(
        [{"timestamp": recent_date.isoformat()}]
    )

    assert temporal == ["latest", "recent"]


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-05T01:02:03",
        datetime(2026, 8, 5, 1, 2, 3),
    ],
    ids=["iso-string", "datetime"],
)
def test_naive_timestamp_is_rejected_instead_of_assuming_utc(timestamp) -> None:
    with pytest.raises(ValueError) as error:
        PatternExtractor().extract_temporal_patterns([{"timestamp": timestamp}])

    assert str(error.value) == f"invalid content timestamp {timestamp!r}"


def test_invalid_timestamp_is_rejected_with_the_supplied_value() -> None:
    with pytest.raises(
        ValueError,
        match="^invalid content timestamp 'not-a-timestamp'$",
    ):
        PatternExtractor().extract_temporal_patterns([{"timestamp": "not-a-timestamp"}])


@pytest.mark.parametrize("timestamp", [True, False])
def test_boolean_timestamp_is_rejected(timestamp) -> None:
    with pytest.raises(ValueError) as error:
        PatternExtractor().extract_temporal_patterns([{"timestamp": timestamp}])

    assert str(error.value) == f"invalid content timestamp {timestamp!r}"


def test_epoch_zero_is_a_valid_supplied_timestamp() -> None:
    temporal = PatternExtractor().extract_temporal_patterns([{"creation_timestamp": 0}])

    assert temporal == ["from 1970"]


def test_empty_extraction_does_not_invent_patterns() -> None:
    patterns = PatternExtractor().extract([])

    assert patterns == {
        "topics": [],
        "entities": [],
        "temporal": [],
        "content_types": [],
    }


def test_content_without_temporal_or_type_evidence_stays_empty() -> None:
    patterns = PatternExtractor().extract([{"topic": "Marie Curie discovered radium"}])

    assert patterns["temporal"] == []
    assert patterns["content_types"] == []


def test_entity_names_alone_do_not_create_relationships() -> None:
    relationships = PatternExtractor().extract_relationships(
        [
            {"text": "Marie Curie", "type": "PERSON"},
            {"text": "Radium", "type": "MATERIAL"},
        ]
    )

    assert relationships == []
