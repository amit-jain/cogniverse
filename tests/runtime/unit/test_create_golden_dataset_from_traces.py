"""Coverage for the golden-dataset-from-traces CLI script.

``scripts/create_golden_dataset_from_traces.py`` mines Phoenix traces +
evaluations into a golden eval dataset of challenging (low-scoring) queries.
It had zero test coverage. These exercise the real analysis pipeline:
per-query aggregation (avg score + majority-vote expected videos), the
low-score challenging filter/sort/top-N, golden-dataset assembly, and the
UTC-aware fetch window.
"""

from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "create_golden_dataset_from_traces.py"
_spec = importlib.util.spec_from_file_location(
    "create_golden_dataset_from_traces", _SCRIPT
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GoldenDatasetGenerator = _mod.GoldenDatasetGenerator


def _bare_generator(
    *, min_occurrences=2, score_threshold=0.5, top_n_queries=20, hours_back=48
):
    """Construct without the telemetry get_provider init."""
    g = GoldenDatasetGenerator.__new__(GoldenDatasetGenerator)
    g.tenant_id = "golden_rt"
    g.hours_back = hours_back
    g.min_occurrences = min_occurrences
    g.score_threshold = score_threshold
    g.top_n_queries = top_n_queries
    g.provider = MagicMock()
    return g


def _trace_row(query, score, videos, profile="frame_based_colpali"):
    return {
        "input": {"query": query},
        "score": score,
        "output": {"results": [{"video_id": v} for v in videos]},
        "attributes": {"profile": profile, "ranking_strategy": "binary_binary"},
        "start_time": "2026-06-05T00:00:00+00:00",
    }


@pytest.mark.unit
class TestAnalyzeQueryPerformance:
    def test_aggregates_scores_and_majority_vote_videos(self):
        g = _bare_generator(min_occurrences=2)
        # "barbell" appears 3x: v1 in all 3 (majority), v2 in 1 (minority).
        df = pd.DataFrame(
            [
                _trace_row("barbell", 0.2, ["v1", "v2"]),
                _trace_row("barbell", 0.4, ["v1"]),
                _trace_row("barbell", 0.3, ["v1"]),
                _trace_row("ocean", 0.9, ["v9"]),  # only 1 occurrence -> filtered
            ]
        )

        stats = g.analyze_query_performance(df)

        assert "ocean" not in stats, "single-occurrence query must be filtered out"
        assert set(stats) == {"barbell"}
        bar = stats["barbell"]
        assert bar["occurrences"] == 3
        assert bar["avg_score"] == pytest.approx((0.2 + 0.4 + 0.3) / 3)
        assert bar["min_score"] == pytest.approx(0.2)
        assert bar["max_score"] == pytest.approx(0.4)
        # v1 appears in 3/3 (>= threshold 1.5); v2 in 1/3 (< 1.5) -> excluded.
        assert bar["expected_videos"] == ["v1"]
        assert bar["profiles_tested"] == ["frame_based_colpali"]
        assert bar["strategies_tested"] == ["binary_binary"]

    def test_query_without_scores_is_dropped(self):
        g = _bare_generator(min_occurrences=1)
        df = pd.DataFrame(
            [
                {"input": {"query": "noscore"}, "output": {"results": []}},
                {"input": {"query": "noscore"}, "output": {"results": []}},
            ]
        )
        assert g.analyze_query_performance(df) == {}


@pytest.mark.unit
class TestChallengingQueriesAndDataset:
    def test_identify_filters_and_sorts_by_lowest_score(self):
        g = _bare_generator(score_threshold=0.5, top_n_queries=2)
        query_stats = {
            "good": {"avg_score": 0.8, "expected_videos": ["a"]},
            "bad": {"avg_score": 0.2, "expected_videos": ["b"]},
            "mid": {"avg_score": 0.45, "expected_videos": ["c"]},
            "worst": {"avg_score": 0.1, "expected_videos": ["d"]},
        }
        challenging = g.identify_challenging_queries(query_stats)
        # "good" (0.8 > 0.5) excluded; remaining sorted ascending, top 2.
        assert [q for q, _ in challenging] == ["worst", "bad"]

    def test_create_golden_dataset_shape(self):
        g = _bare_generator()
        challenging = [
            (
                "bad query",
                {
                    "expected_videos": ["v1", "v2"],
                    "avg_score": 0.2,
                    "occurrences": 4,
                    "profiles_tested": ["frame_based_colpali"],
                    "strategies_tested": ["binary_binary"],
                },
            )
        ]
        ds = g.create_golden_dataset(challenging)
        assert set(ds) == {"bad query"}
        entry = ds["bad query"]
        assert entry["expected_videos"] == ["v1", "v2"]
        assert entry["difficulty"] == "challenging"
        assert entry["auto_generated"] is True
        assert entry["avg_score"] == pytest.approx(0.2)
        assert entry["occurrences"] == 4
        # Each expected video gets a default relevance of 1.0.
        assert entry["relevance_scores"] == {"v1": 1.0, "v2": 1.0}


@pytest.mark.unit
class TestFetchWindow:
    @pytest.mark.asyncio
    async def test_fetch_window_is_utc_aware(self):
        g = _bare_generator(hours_back=48)
        captured = {}

        async def fake_get_spans(**kwargs):
            captured.update(kwargs)
            return pd.DataFrame([_trace_row("q", 0.5, ["v1"])])

        g.provider.traces.get_spans = fake_get_spans
        g.provider.annotations.get_annotations = AsyncMock(return_value=pd.DataFrame())

        await g.fetch_traces_with_evaluations()

        assert captured["start_time"].tzinfo is not None
        assert captured["end_time"].tzinfo is not None
        assert captured["end_time"] - captured["start_time"] == timedelta(hours=48)
        assert captured["project"] == "cogniverse-golden_rt"

    @pytest.mark.asyncio
    async def test_empty_spans_returns_empty_frame(self):
        g = _bare_generator()
        g.provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())
        result = await g.fetch_traces_with_evaluations()
        assert result.empty
