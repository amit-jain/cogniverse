"""Holdout family predicates keep ineligible ids train-only."""

import pytest

from cogniverse_runtime.optimization_cli import (
    _entity_extraction_is_scoreable,
    _profile_selection_is_scoreable,
    _split_holdout,
    _split_served_holdout,
    is_scoreable,
)


def _served_record(index: int, query: str, example_id: str) -> dict[str, object]:
    return {
        "query": query,
        "source_text": f"served text {index}",
        "grounding_context": f"served context {index}",
        "selected_profile": "video_colpali_smol500_mv_frame",
        "example_id": example_id,
    }


def _truth_record(index: int, query: str, example_id: str) -> dict[str, object]:
    return {
        "query": query,
        "entities": [{"text": f"entity{index}", "type": "CONCEPT"}],
        "example_id": example_id,
    }


def _assert_served_span_split(scoreable_predicate) -> None:
    records = [
        _served_record(0, "alpha", "span:simba:0"),
        _served_record(1, "beta", "span:simba:1"),
        _served_record(2, "gamma", "span:simba:2"),
        _served_record(3, "gamma", "approved:simba:0"),
    ]

    split = _split_served_holdout(records, 1, scoreable_predicate=scoreable_predicate)

    assert split.distinct_queries == 3
    assert split.holdout_queries == 1
    assert tuple(record["example_id"] for record in split.train) == (
        "span:simba:0",
        "span:simba:1",
        "approved:simba:0",
    )
    assert tuple(record["example_id"] for record in split.holdout) == ("span:simba:2",)


def test_simba_holdout_keeps_approved_rows_train_only() -> None:
    _assert_served_span_split(is_scoreable)


def test_profile_holdout_keeps_approved_rows_train_only() -> None:
    _assert_served_span_split(_profile_selection_is_scoreable)


def test_entity_holdout_keeps_approved_rows_train_only() -> None:
    records = [
        _truth_record(0, "alpha", "truth:entity:0"),
        _truth_record(1, "beta", "truth:entity:1"),
        _truth_record(2, "gamma", "truth:entity:2"),
        _truth_record(3, "gamma", "approved:entity:0"),
    ]

    split = _split_holdout(
        records, 1, scoreable_predicate=_entity_extraction_is_scoreable
    )

    assert split.distinct_queries == 3
    assert split.holdout_queries == 1
    assert tuple(record["example_id"] for record in split.train) == (
        "truth:entity:0",
        "truth:entity:1",
        "approved:entity:0",
    )
    assert tuple(record["example_id"] for record in split.holdout) == (
        "truth:entity:2",
    )


@pytest.mark.parametrize(
    ("splitter", "scoreable_predicate"),
    [
        (_split_served_holdout, is_scoreable),
        (_split_served_holdout, _profile_selection_is_scoreable),
        (_split_holdout, _entity_extraction_is_scoreable),
    ],
    ids=("simba", "profile", "entity"),
)
def test_unknown_example_id_family_raises(splitter, scoreable_predicate) -> None:
    record = {
        "query": "shadow query",
        "source_text": "shadow text",
        "grounding_context": "shadow context",
        "selected_profile": "video_colpali_smol500_mv_frame",
        "entities": [{"text": "shadow", "type": "CONCEPT"}],
        "example_id": "shadow:0",
    }

    with pytest.raises(ValueError, match="shadow:0"):
        splitter([record], 1, scoreable_predicate=scoreable_predicate)
