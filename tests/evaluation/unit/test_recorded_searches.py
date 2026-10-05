"""Scoring a tenant's recorded searches against its golden set."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from cogniverse_evaluation.recorded_searches import (
    SEARCH_SPAN_NAME,
    score_recorded_searches,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

SUNSET, RED_CAR, DOG = "sunset over the sea", "a red car", "dog on a beach"
# Golden order is not alphabetical, so a sort by query text shows.
GOLDEN = [
    {"query": SUNSET, "expected_videos": ["sunset"]},
    {"query": RED_CAR, "expected_videos": ["red_car", "garage"]},
    {"query": DOG, "expected_videos": ["dog"]},
]


def _span(query, profile, strategy, titles, *, at, status="OK", trace="t"):
    rows = [
        {"document_id": f"doc{i}", "source_title": title}
        for i, title in enumerate(titles)
    ]
    return {
        "name": SEARCH_SPAN_NAME,
        "start_time": pd.Timestamp(f"2026-10-05T10:{at:02d}:00Z"),
        "status_code": status,
        "context.trace_id": trace,
        "attributes.query": query,
        "attributes.profile": profile,
        "attributes.strategy": strategy,
        "attributes.output.value": json.dumps(rows),
    }


def _ndcg(relevances, ideal):
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))
    return dcg / sum(1 / math.log2(rank + 2) for rank in range(ideal))


def _scores(spans):
    return score_recorded_searches(pd.DataFrame(spans), GOLDEN)


def test_latest_search_per_profile_strategy_and_query_is_scored():
    scored = _scores(
        [
            # The newest search arrives neither first nor last.
            _span(SUNSET, "colpali", "hybrid", ["sunset.mp4"], at=3, trace="mid"),
            _span(
                SUNSET, "colpali", "hybrid", ["x.mp4", "sunset.mp4"], at=5, trace="new"
            ),
            _span(SUNSET, "colpali", "hybrid", ["sunset.mp4"], at=1, trace="old"),
            # Two segments of one source count once, at the best rank.
            _span(
                RED_CAR,
                "colpali",
                "hybrid",
                ["red_car.mp4", "red_car.mp4", "x.mp4", "garage.mov"],
                at=2,
            ),
            _span(f"  {SUNSET} ", "colpali", "bm25", ["y.mp4"], at=3),
            _span("not a golden query", "colpali", "bm25", ["sunset.mp4"], at=4),
        ]
    )
    sunset_ndcg = _ndcg([0, 1], 1)
    red_car_ndcg = _ndcg([1, 0, 1], 2)
    assert scored["queries"] == [
        {
            "profile": "colpali",
            "strategy": "bm25",
            "query": SUNSET,
            "expected": ["sunset"],
            "retrieved": ["y"],
            "searched_at": "2026-10-05T10:03:00+00:00",
            "trace_id": "t",
            "mrr": 0.0,
            "ndcg": 0.0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "precision_at_5": 0.0,
        },
        {
            "profile": "colpali",
            "strategy": "hybrid",
            "query": SUNSET,
            "expected": ["sunset"],
            "retrieved": ["x", "sunset"],
            "searched_at": "2026-10-05T10:05:00+00:00",
            "trace_id": "new",
            "mrr": 0.5,
            "ndcg": pytest.approx(sunset_ndcg),
            "recall_at_1": 0.0,
            "recall_at_5": 1.0,
            "precision_at_5": 0.5,
        },
        {
            "profile": "colpali",
            "strategy": "hybrid",
            "query": RED_CAR,
            "expected": ["red_car", "garage"],
            "retrieved": ["red_car", "x", "garage"],
            "searched_at": "2026-10-05T10:02:00+00:00",
            "trace_id": "t",
            "mrr": 1.0,
            "ndcg": pytest.approx(red_car_ndcg),
            "recall_at_1": 0.5,
            "recall_at_5": 1.0,
            "precision_at_5": pytest.approx(2 / 3),
        },
    ]
    assert scored["strategies"] == [
        {
            "profile": "colpali",
            "strategy": "bm25",
            "queries": 1,
            "mrr": 0.0,
            "ndcg": 0.0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "precision_at_5": 0.0,
            "success_rate": 0.0,
        },
        {
            "profile": "colpali",
            "strategy": "hybrid",
            "queries": 2,
            "mrr": 0.75,
            "ndcg": pytest.approx((sunset_ndcg + red_car_ndcg) / 2),
            "recall_at_1": 0.25,
            "recall_at_5": 1.0,
            "precision_at_5": pytest.approx((0.5 + 2 / 3) / 2),
            "success_rate": 0.5,
        },
    ]
    assert scored["golden_queries"] == 3
    assert scored["unsearched_queries"] == [DOG]
    assert (scored["failed_searches"], scored["unscored_searches"]) == (0, 0)


def test_failed_and_untitled_searches_are_counted_not_scored():
    untitled = _span(RED_CAR, "colpali", "bm25", ["red_car.mp4"], at=2)
    untitled["attributes.output.value"] = json.dumps(
        [{"document_id": "doc0", "source_title": None}]
    )
    no_rows = _span(DOG, "colpali", "bm25", [], at=3)
    no_rows["attributes.output.value"] = None
    scored = _scores(
        [
            _span(SUNSET, "colpali", "bm25", [], at=1, status="ERROR"),
            untitled,
            no_rows,
            _span("not a golden query", "colpali", "bm25", [], at=4, status="ERROR"),
        ]
    )
    assert scored == {
        "golden_queries": 3,
        "strategies": [],
        "queries": [],
        "unsearched_queries": [SUNSET, RED_CAR, DOG],
        "failed_searches": 1,
        "unscored_searches": 2,
    }


def test_a_search_with_no_results_scores_zero():
    scored = _scores([_span(DOG, "colpali", "bm25", [], at=1)])
    assert [(q["query"], q["retrieved"], q["mrr"]) for q in scored["queries"]] == [
        (DOG, [], 0.0)
    ]
    assert scored["unsearched_queries"] == [SUNSET, RED_CAR]


def test_no_spans_scores_nothing():
    scored = score_recorded_searches(pd.DataFrame(), GOLDEN)
    assert scored == {
        "golden_queries": 3,
        "strategies": [],
        "queries": [],
        "unsearched_queries": [SUNSET, RED_CAR, DOG],
        "failed_searches": 0,
        "unscored_searches": 0,
    }
