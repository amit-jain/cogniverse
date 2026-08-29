"""Ledger versions name the metric that produced their score.

A score is only comparable against ``confirmation_score_threshold`` when the
metric that produced it is the metric in force now. Versions scored under a
superseded metric, and versions from before the ledger recorded a metric at
all, are unknown evidence: they neither confirm nor count against an example.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.example_selection import (
    ExampleStats,
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
)
from tests.evaluation.fakes import InMemoryDatasetStore, StubTelemetryProvider

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CURRENT = "entity_extraction.pair_set_f1.v1"
SUPERSEDED = "entity_extraction.token_set_f1.v1"
NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)
OLD_DAY = 1
OLD = datetime(2026, 8, OLD_DAY, tzinfo=timezone.utc)
THRESHOLD = 0.7
KNOBS = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5, THRESHOLD)
KNOBS_UNSET = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5, None)
EXAMPLE = "span:e-1"
TENANT = "metricid_tenant:metricid_tenant"


def _row(*, score: float | None, metric_id: str | None, day: int = OLD_DAY) -> dict:
    """One promoted lineage entry as ``get_version_lineage`` returns it."""
    row = {
        "consumed_example_ids": [EXAMPLE],
        "decision": "promote",
        "scored": score is not None,
        "score": score,
        "candidate_score": score,
        "created_at": f"2026-08-{day:02d}T00:00:00+00:00",
    }
    if metric_id is not None:
        row["metric_id"] = metric_id
    return row


@pytest.fixture
def manager() -> ArtifactManager:
    return ArtifactManager(
        telemetry_provider=StubTelemetryProvider(InMemoryDatasetStore()),
        tenant_id=TENANT,
    )


@pytest.mark.asyncio
async def test_saved_version_carries_metric_id_into_lineage(manager):
    """The metric id stamped at write time is what the lineage reads back."""
    await manager.save_blob_versioned(
        "model",
        "entity_extraction",
        '{"extract.predict": {"demos": []}}',
        consumed_example_ids=["span:e-1"],
        decision="promote",
        scored=True,
        score=0.81,
        base_score=0.6,
        candidate_score=0.81,
        metric_id=CURRENT,
    )

    lineage = await manager.get_version_lineage("model", "entity_extraction")

    assert lineage == [
        {
            "version": 1,
            "name": f"dspy-model-{TENANT}-entity_extraction-v1",
            "row_count": 1,
            "consumed_example_ids": ["span:e-1"],
            "decision": "promote",
            "scored": True,
            "score": 0.81,
            "base_score": 0.6,
            "candidate_score": 0.81,
            "created_at": lineage[0]["created_at"],
            "metric_id": CURRENT,
        }
    ]


@pytest.mark.asyncio
async def test_unscored_version_needs_no_metric_id(manager):
    """A version that records no score names no metric; the field reads None."""
    await manager.save_blob_versioned(
        "model",
        "entity_extraction",
        "{}",
        consumed_example_ids=["span:e-1"],
        decision="insufficient_population",
        scored=False,
        score=None,
        base_score=None,
        candidate_score=None,
    )

    lineage = await manager.get_version_lineage("model", "entity_extraction")

    assert lineage[0]["metric_id"] is None
    assert lineage[0]["scored"] is False
    assert lineage[0]["score"] is None


@pytest.mark.asyncio
async def test_scored_version_without_metric_id_is_refused(manager):
    """A comparable number may not enter the ledger anonymously."""
    with pytest.raises(ValueError) as excinfo:
        await manager.save_blob_versioned(
            "model",
            "entity_extraction",
            "{}",
            consumed_example_ids=["span:e-1"],
            decision="promote",
            scored=True,
            score=0.81,
            base_score=0.6,
            candidate_score=0.81,
        )

    assert str(excinfo.value) == (
        "metric_id is required when a version records a score "
        "(kind='model', key='entity_extraction')"
    )


def test_current_metric_id_above_threshold_confirms():
    """One confirmation counts but does not yet clear the confirmation floor."""
    stats = confirmation_stats(
        [_row(score=0.81, metric_id=CURRENT)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(1, 0, 1, OLD)}
    assert decay_weight(stats, EXAMPLE, now=NOW, knobs=KNOBS) == 0.5


def test_three_current_metric_confirmations_hold_off_decay():
    stats = confirmation_stats(
        [_row(score=0.81, metric_id=CURRENT) for _ in range(3)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(3, 0, 3, OLD)}
    assert decay_weight(stats, EXAMPLE, now=NOW, knobs=KNOBS) == 1.0


def test_current_metric_id_below_threshold_does_not_confirm():
    stats = confirmation_stats(
        [_row(score=0.62, metric_id=CURRENT)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(0, 0, 1, OLD)}
    assert decay_weight(stats, EXAMPLE, now=NOW, knobs=KNOBS) == 0.5


def test_superseded_metric_id_is_unknown_not_unconfirmed():
    """0.62 under the old metric is not evidence against the example."""
    stats = confirmation_stats(
        [_row(score=0.62, metric_id=SUPERSEDED)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(0, 1, 0, OLD)}


def test_superseded_metric_id_above_threshold_is_unknown_not_confirmed():
    """0.81 under the old metric is not evidence for the example either."""
    stats = confirmation_stats(
        [_row(score=0.81, metric_id=SUPERSEDED)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(0, 1, 0, OLD)}


def test_absent_metric_id_is_unknown():
    """Every version written before the ledger recorded a metric."""
    stats = confirmation_stats(
        [_row(score=0.81, metric_id=None), _row(score=0.62, metric_id=None)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert stats == {EXAMPLE: ExampleStats(0, 2, 0, OLD)}


def test_unknown_promotions_protect_an_old_example_from_decay():
    """Three unknown promotions hold decay off exactly as three unscored ones do."""
    superseded = confirmation_stats(
        [_row(score=0.62, metric_id=SUPERSEDED) for _ in range(3)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )
    unscored = confirmation_stats(
        [_row(score=None, metric_id=None) for _ in range(3)],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )

    assert superseded == unscored == {EXAMPLE: ExampleStats(0, 3, 0, OLD)}
    assert decay_weight(superseded, EXAMPLE, now=NOW, knobs=KNOBS) == 1.0


def test_metric_change_never_introduces_new_decay():
    """Reclassifying confirmations as unknown cannot start downweighting.

    The thresholded decay guard tests ``confirmations + unscored_promotions``,
    so moving a promotion between those two buckets leaves the sum fixed.
    """
    lineage = [_row(score=0.81, metric_id=SUPERSEDED) for _ in range(3)]

    before = confirmation_stats(
        [dict(row, metric_id=CURRENT) for row in lineage],
        score_threshold=THRESHOLD,
        metric_id=CURRENT,
    )
    after = confirmation_stats(lineage, score_threshold=THRESHOLD, metric_id=CURRENT)

    assert before == {EXAMPLE: ExampleStats(3, 0, 3, OLD)}
    assert after == {EXAMPLE: ExampleStats(0, 3, 0, OLD)}
    assert decay_weight(before, EXAMPLE, now=NOW, knobs=KNOBS) == 1.0
    assert decay_weight(after, EXAMPLE, now=NOW, knobs=KNOBS) == 1.0


def test_mixed_metric_ids_split_into_confirmed_and_unknown():
    lineage = [
        _row(score=0.81, metric_id=CURRENT),
        _row(score=0.90, metric_id=SUPERSEDED, day=2),
        _row(score=0.55, metric_id=CURRENT, day=3),
        _row(score=None, metric_id=None, day=4),
    ]

    stats = confirmation_stats(lineage, score_threshold=THRESHOLD, metric_id=CURRENT)

    assert stats == {EXAMPLE: ExampleStats(1, 2, 2, OLD)}


def test_thresholded_stats_refuse_an_unnamed_metric():
    with pytest.raises(ValueError) as excinfo:
        confirmation_stats([_row(score=0.81, metric_id=CURRENT)], score_threshold=0.7)

    assert str(excinfo.value) == (
        "confirmation_stats requires metric_id when score_threshold is set"
    )


def test_threshold_unset_ignores_metric_identity_entirely():
    """The two optimizers with no threshold behave exactly as before.

    Same lineage, once carrying mixed metric ids and once carrying none:
    identical stats and identical weights, whatever metric_id is passed.
    """
    stamped = [
        _row(score=0.81, metric_id=CURRENT),
        _row(score=0.90, metric_id=SUPERSEDED, day=2),
        _row(score=None, metric_id=None, day=3),
    ]
    stripped = [{k: v for k, v in row.items() if k != "metric_id"} for row in stamped]
    expected = {EXAMPLE: ExampleStats(3, 1, 2, OLD)}

    assert confirmation_stats(stamped, score_threshold=None) == expected
    assert confirmation_stats(stripped, score_threshold=None) == expected
    assert (
        confirmation_stats(stamped, score_threshold=None, metric_id=CURRENT) == expected
    )
    assert (
        decay_weight(
            confirmation_stats(stamped, score_threshold=None, metric_id=CURRENT),
            EXAMPLE,
            now=NOW,
            knobs=KNOBS_UNSET,
        )
        == 1.0
    )
