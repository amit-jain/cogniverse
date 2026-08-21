from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_agents.optimizer.example_selection import (
    ExampleStats,
    SelectionReport,
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
    embed_texts,
    select_training_records,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _selection_block(
    pool: int,
    deduped: int,
    *,
    cap: int = 300,
    mmr_applied: bool = False,
    decayed_count: int = 0,
) -> dict[str, dict[str, int | bool]]:
    return {
        "selection": {
            "pool": pool,
            "deduped": deduped,
            "cap": cap,
            "mmr_applied": mmr_applied,
            "decayed_count": decayed_count,
        }
    }


LEDGER = [
    {
        "consumed_example_ids": ["span:a", "approved:b"],
        "decision": "promote",
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:a", "span:c"],
        "decision": "keep",
        "created_at": "2026-08-10T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:a"],
        "decision": "promote",
        "created_at": "2026-08-15T00:00:00+00:00",
    },
]

LEDGER_CONFIRMED_OLD = [
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-10T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-15T00:00:00+00:00",
    },
]

LEDGER_FRESH_UNCONFIRMED = [
    {
        "consumed_example_ids": ["span:e"],
        "decision": "keep",
        "created_at": "2026-08-25T00:00:00+00:00",
    }
]


def test_confirmation_stats_complete_golden():
    stats = confirmation_stats(LEDGER)

    assert stats == {
        "span:a": ExampleStats(2, datetime(2026, 8, 1, tzinfo=timezone.utc)),
        "approved:b": ExampleStats(1, datetime(2026, 8, 1, tzinfo=timezone.utc)),
        "span:c": ExampleStats(0, datetime(2026, 8, 10, tzinfo=timezone.utc)),
    }


def test_decay_weight_old_unconfirmed_halves_when_under_threshold():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER)

    assert decay_weight(stats, "span:c", now=now, knobs=knobs) == 0.5
    assert decay_weight(stats, "span:a", now=now, knobs=knobs) == 0.5
    assert decay_weight(stats, "approved:b", now=now, knobs=knobs) == 0.5


def test_decay_weight_unknown_id_is_fresh():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER)

    assert decay_weight(stats, "span:missing", now=now, knobs=knobs) == 1.0


def test_decay_weight_confirmed_old_remains_full_weight():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER_CONFIRMED_OLD)

    assert decay_weight(stats, "span:d", now=now, knobs=knobs) == 1.0


def test_decay_weight_fresh_unconfirmed_remains_full_weight():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER_FRESH_UNCONFIRMED)

    assert decay_weight(stats, "span:e", now=now, knobs=knobs) == 1.0


def test_mmr_prefers_diverse_over_duplicate_direction():
    calls: list[list[str]] = []

    def _fake_embed(texts):
        calls.append(list(texts))
        vectors = {
            "alpha one": [1.0, 0.0],
            "alpha two": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    records = [
        {"example_id": "span:a", "query": "alpha one"},
        {"example_id": "span:b", "query": "alpha two"},
        {"example_id": "span:c", "query": "beta"},
    ]
    selected, report = select_training_records(
        records,
        weights={"span:a": 1.0, "span:b": 1.0, "span:c": 0.5},
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=_fake_embed,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:b"]
    assert report == SelectionReport(
        pool=3,
        deduped=3,
        cap=2,
        mmr_applied=True,
        decayed_count=1,
        selected_ids=["span:a", "span:b"],
    )
    assert calls == [["alpha one", "alpha two", "beta"]]


def test_below_cap_never_embeds():
    def boom(_):
        raise AssertionError("embed_fn called below cap")

    selected, report = select_training_records(
        [{"example_id": "span:a", "query": "q"}],
        weights={"span:a": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert selected == [{"example_id": "span:a", "query": "q"}]
    assert report == SelectionReport(
        pool=1,
        deduped=1,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a"],
    )


def test_dedup_casefold_first_wins():
    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Find Cats"},
            {"example_id": "span:b", "query": "find cats"},
        ],
        weights={"span:a": 1.0, "span:b": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=lambda texts: [],
    )

    assert [record["example_id"] for record in selected] == ["span:a"]
    assert report == SelectionReport(
        pool=2,
        deduped=1,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a"],
    )


def test_zero_norm_embedding_raises_value_error_named_example_id():
    def _fake_embed(_):
        return [[0.0, 0.0], [1.0, 0.0]]

    with pytest.raises(ValueError, match=r"embedding for span:a has zero norm"):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:b", "query": "beta"},
            ],
            weights={"span:a": 1.0, "span:b": 1.0},
            knobs=TrainingSelectionKnobs(1, 0.7, 3, 14, 0.5),
            embed_fn=_fake_embed,
        )


def test_at_cap_boundary_returns_all_without_embedding():
    records = [
        {"example_id": "span:a", "query": "alpha"},
        {"example_id": "span:b", "query": "beta"},
    ]

    def boom(_):
        raise AssertionError("embed_fn called at cap")

    selected, report = select_training_records(
        records,
        weights={"span:a": 1.0, "span:b": 1.0},
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert selected == records
    assert report == SelectionReport(
        pool=2,
        deduped=2,
        cap=2,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a", "span:b"],
    )


def test_missing_weight_raises_key_error_for_missing_example_id():
    with pytest.raises(KeyError, match=r"span:missing"):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:missing", "query": "beta"},
            ],
            weights={"span:a": 1.0},
            knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
            embed_fn=lambda texts: [],
        )


def test_decayed_count_ignores_removed_duplicate_weight():
    def boom(_):
        raise AssertionError("embed_fn called below cap")

    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Alpha"},
            {"example_id": "span:b", "query": "alpha"},
            {"example_id": "span:c", "query": "Gamma"},
        ],
        weights={"span:a": 1.0, "span:b": 0.5, "span:c": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:c"]
    assert report == SelectionReport(
        pool=3,
        deduped=2,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a", "span:c"],
    )


def test_mmr_embeds_deduped_queries_once_in_order():
    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        assert texts == ["Alpha", "beta", "Gamma"]
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Alpha"},
            {"example_id": "span:b", "query": "beta"},
            {"example_id": "span:c", "query": "ALPHA"},
            {"example_id": "span:d", "query": "Gamma"},
        ],
        weights={
            "span:a": 1.0,
            "span:b": 1.0,
            "span:c": 0.5,
            "span:d": 1.0,
        },
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=fake_embed,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:d"]
    assert report == SelectionReport(
        pool=4,
        deduped=3,
        cap=2,
        mmr_applied=True,
        decayed_count=0,
        selected_ids=["span:a", "span:d"],
    )
    assert calls == [["Alpha", "beta", "Gamma"]]


def test_dead_port_embedder_raises_runtime_error():
    with pytest.raises(
        RuntimeError,
        match=r"training-selection embedder at http://127\.0\.0\.1:29071 failed:",
    ):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:b", "query": "beta"},
            ],
            weights={"span:a": 1.0, "span:b": 1.0},
            knobs=TrainingSelectionKnobs(1, 0.7, 3, 14, 0.5),
            embed_fn=lambda texts: embed_texts(
                "http://127.0.0.1:29071",
                texts,
                timeout=1.0,
            ),
        )
