"""Aggregate-metric computation for the evaluation tab.

Regression (HIGH/PERF): the per-profile/strategy aggregate was recomputed
INSIDE the per-experiment loop (O(experiments^2) over all accumulated queries).
It is now a single pure pass (`_aggregate_experiment_metrics`) run once after
all experiments load. These pin its correctness + empty-query safety.
"""

import pytest

from cogniverse_dashboard.tabs.evaluation import _aggregate_experiment_metrics


def test_aggregate_computes_means_over_all_queries():
    data = {
        "frame_based_colpali": {
            "binary_binary": {
                "queries": [
                    {"metrics": {"mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0}},
                    {"metrics": {"mrr": 0.5, "recall@1": 0.0, "recall@5": 0.5}},
                    {"metrics": {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0}},
                ],
                "aggregate_metrics": {"mrr": {"mean": 0}},  # stale placeholder
            }
        }
    }

    _aggregate_experiment_metrics(data)

    agg = data["frame_based_colpali"]["binary_binary"]["aggregate_metrics"]
    assert agg["mrr"]["mean"] == pytest.approx(0.5)  # (1.0+0.5+0.0)/3
    assert agg["recall@1"]["mean"] == pytest.approx(1 / 3)  # (1+0+0)/3
    assert agg["recall@5"]["mean"] == pytest.approx(0.5)  # (1.0+0.5+0.0)/3


def test_aggregate_handles_multiple_profiles_and_strategies():
    data = {
        "p1": {
            "s1": {
                "queries": [
                    {"metrics": {"mrr": 0.8, "recall@1": 1.0, "recall@5": 1.0}}
                ],
                "aggregate_metrics": {},
            }
        },
        "p2": {
            "s2": {
                "queries": [
                    {"metrics": {"mrr": 0.2, "recall@1": 0.0, "recall@5": 0.4}}
                ],
                "aggregate_metrics": {},
            }
        },
    }

    _aggregate_experiment_metrics(data)

    assert data["p1"]["s1"]["aggregate_metrics"]["mrr"]["mean"] == pytest.approx(0.8)
    assert data["p2"]["s2"]["aggregate_metrics"]["recall@5"]["mean"] == pytest.approx(
        0.4
    )


def test_aggregate_skips_empty_queries_no_zero_division():
    data = {"p": {"s": {"queries": [], "aggregate_metrics": {"mrr": {"mean": 0}}}}}
    _aggregate_experiment_metrics(data)  # must not raise ZeroDivisionError
    assert data["p"]["s"]["aggregate_metrics"]["mrr"]["mean"] == 0
