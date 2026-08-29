from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_agents.optimizer.example_selection import (
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

METRIC_ID = "entity_extraction.pair_set_f1.v1"
NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
OLD = datetime(2026, 8, 1, tzinfo=timezone.utc)
FRESH = datetime(2026, 8, 25, tzinfo=timezone.utc)
EXPECTED_FIELDS = (
    "confirmations",
    "unscored_promotions",
    "scored_promotions",
    "first_seen",
)


def _row(
    day: int,
    *,
    example_id: str,
    decision: str,
    scored: bool,
    score: float | None,
) -> dict:
    return {
        "consumed_example_ids": [example_id],
        "decision": decision,
        "scored": scored,
        "score": score,
        "candidate_score": score,
        "metric_id": METRIC_ID,
        "created_at": f"2026-08-{day:02d}T00:00:00+00:00",
    }


def _assert_example_stats(
    stats_row,
    *,
    confirmations: int,
    unscored_promotions: int,
    scored_promotions: int,
    first_seen: datetime,
) -> None:
    assert stats_row._fields == EXPECTED_FIELDS
    assert stats_row._asdict() == {
        "confirmations": confirmations,
        "unscored_promotions": unscored_promotions,
        "scored_promotions": scored_promotions,
        "first_seen": first_seen,
    }
    assert stats_row.confirmations == confirmations
    assert stats_row.unscored_promotions == unscored_promotions
    assert stats_row.scored_promotions == scored_promotions
    assert stats_row.first_seen == first_seen


CASES = [
    pytest.param(
        "old + 3 scored-above-threshold promotions -> no decay",
        "span:confirmed",
        [
            _row(
                1,
                example_id="span:confirmed",
                decision="promote",
                scored=True,
                score=0.8,
            ),
            _row(
                10,
                example_id="span:confirmed",
                decision="promote",
                scored=True,
                score=0.8,
            ),
            _row(
                15,
                example_id="span:confirmed",
                decision="promote",
                scored=True,
                score=0.8,
            ),
        ],
        0.7,
        1.0,
        {
            "confirmations": 3,
            "unscored_promotions": 0,
            "scored_promotions": 3,
            "first_seen": OLD,
        },
        id="confirmed-above",
    ),
    pytest.param(
        "old + 3 scored-below-threshold promotions -> decays",
        "span:below",
        [
            _row(
                1, example_id="span:below", decision="promote", scored=True, score=0.6
            ),
            _row(
                10, example_id="span:below", decision="promote", scored=True, score=0.6
            ),
            _row(
                15, example_id="span:below", decision="promote", scored=True, score=0.6
            ),
        ],
        0.7,
        0.5,
        {
            "confirmations": 0,
            "unscored_promotions": 0,
            "scored_promotions": 3,
            "first_seen": OLD,
        },
        id="confirmed-below",
    ),
    pytest.param(
        "old + 3 unscored promotions -> no decay",
        "span:unknown",
        [
            _row(
                1,
                example_id="span:unknown",
                decision="promote",
                scored=False,
                score=None,
            ),
            _row(
                10,
                example_id="span:unknown",
                decision="promote",
                scored=False,
                score=None,
            ),
            _row(
                15,
                example_id="span:unknown",
                decision="promote",
                scored=False,
                score=None,
            ),
        ],
        0.7,
        1.0,
        {
            "confirmations": 0,
            "unscored_promotions": 3,
            "scored_promotions": 0,
            "first_seen": OLD,
        },
        id="unknown-only",
    ),
    pytest.param(
        "old + mixed: 1 scored-above, 2 unscored -> no decay",
        "span:mixed",
        [
            _row(
                1, example_id="span:mixed", decision="promote", scored=True, score=0.8
            ),
            _row(
                10,
                example_id="span:mixed",
                decision="promote",
                scored=False,
                score=None,
            ),
            _row(
                15,
                example_id="span:mixed",
                decision="promote",
                scored=False,
                score=None,
            ),
        ],
        0.7,
        1.0,
        {
            "confirmations": 1,
            "unscored_promotions": 2,
            "scored_promotions": 1,
            "first_seen": OLD,
        },
        id="mixed-unknown",
    ),
    pytest.param(
        "old + zero promotions at all -> decays",
        "span:none",
        [
            _row(1, example_id="span:none", decision="keep", scored=False, score=None),
            _row(10, example_id="span:none", decision="keep", scored=False, score=None),
            _row(15, example_id="span:none", decision="keep", scored=False, score=None),
        ],
        0.7,
        0.5,
        {
            "confirmations": 0,
            "unscored_promotions": 0,
            "scored_promotions": 0,
            "first_seen": OLD,
        },
        id="no-promotions",
    ),
    pytest.param(
        "fresh, any history -> no decay",
        "span:fresh",
        [
            _row(
                25, example_id="span:fresh", decision="promote", scored=True, score=0.6
            ),
            _row(
                25, example_id="span:fresh", decision="promote", scored=True, score=0.6
            ),
            _row(
                25, example_id="span:fresh", decision="promote", scored=True, score=0.6
            ),
        ],
        0.7,
        1.0,
        {
            "confirmations": 0,
            "unscored_promotions": 0,
            "scored_promotions": 3,
            "first_seen": FRESH,
        },
        id="fresh-history",
    ),
    pytest.param(
        "threshold unset -> current behavior preserved exactly",
        "span:legacy",
        [
            _row(
                1,
                example_id="span:legacy",
                decision="promote",
                scored=False,
                score=None,
            ),
            _row(
                10,
                example_id="span:legacy",
                decision="promote",
                scored=False,
                score=None,
            ),
        ],
        None,
        0.5,
        {
            "confirmations": 2,
            "unscored_promotions": 2,
            "scored_promotions": 0,
            "first_seen": OLD,
        },
        id="threshold-none",
    ),
]


@pytest.mark.parametrize(
    "_label, example_id, lineage, score_threshold, expected_weight, expected_stats",
    CASES,
)
def test_decay_state_matrix(
    _label,
    example_id,
    lineage,
    score_threshold,
    expected_weight,
    expected_stats,
):
    now = NOW
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5, score_threshold)

    stats = confirmation_stats(
        lineage, score_threshold=score_threshold, metric_id=METRIC_ID
    )
    assert set(stats) == {example_id}
    _assert_example_stats(stats[example_id], **expected_stats)
    if _label == "old + 3 scored-above-threshold promotions -> no decay":
        assert stats[example_id].confirmations == 3
        assert stats[example_id].unscored_promotions == 0
        assert stats[example_id].scored_promotions == 3
    elif _label == "old + 3 scored-below-threshold promotions -> decays":
        assert stats[example_id].confirmations == 0
        assert stats[example_id].unscored_promotions == 0
        assert stats[example_id].scored_promotions == 3
    elif _label == "old + 3 unscored promotions -> no decay":
        assert stats[example_id].confirmations == 0
        assert stats[example_id].unscored_promotions == 3
        assert stats[example_id].scored_promotions == 0
    elif _label == "old + mixed: 1 scored-above, 2 unscored -> no decay":
        assert stats[example_id].confirmations == 1
        assert stats[example_id].unscored_promotions == 2
        assert stats[example_id].scored_promotions == 1
    elif _label == "old + zero promotions at all -> decays":
        assert stats[example_id].confirmations == 0
        assert stats[example_id].unscored_promotions == 0
        assert stats[example_id].scored_promotions == 0
    elif _label == "fresh, any history -> no decay":
        assert stats[example_id].confirmations == 0
        assert stats[example_id].unscored_promotions == 0
        assert stats[example_id].scored_promotions == 3
    else:
        assert _label == "threshold unset -> current behavior preserved exactly"
        assert stats[example_id].confirmations == 2
        assert stats[example_id].unscored_promotions == 2
        assert stats[example_id].scored_promotions == 0
    assert decay_weight(stats, example_id, now=now, knobs=knobs) == expected_weight
