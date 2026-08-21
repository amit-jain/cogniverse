from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_agents.optimizer.example_selection import (
    ExampleStats,
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


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
