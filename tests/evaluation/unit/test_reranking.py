"""Tests for applying the live rerankers to evaluation traces.

Uses ``multi_modal`` (pure heuristic, no external model) so the reranking runs
in-process and exercises the real shared rerank_result_dicts service.
"""

import pytest

from cogniverse_evaluation.core.reranking import apply_reranking_to_traces


@pytest.mark.asyncio
async def test_apply_reranking_reorders_trace_results():
    """multi_modal reranks each trace's results in place, preserving the id set."""
    traces = [
        {
            "trace_id": "t1",
            "query": "cat",
            "results": [
                {
                    "id": "a",
                    "title": "cat",
                    "content": "a cat plays",
                    "modality": "video",
                    "score": 0.3,
                },
                {
                    "id": "b",
                    "title": "dog",
                    "content": "a dog runs",
                    "modality": "video",
                    "score": 0.9,
                },
                {
                    "id": "c",
                    "title": "bird",
                    "content": "a bird sings",
                    "modality": "video",
                    "score": 0.5,
                },
            ],
        }
    ]
    out = await apply_reranking_to_traces(traces, "multi_modal", {})
    assert len(out) == 1
    reranked = out[0]["results"]
    assert sorted(r["id"] for r in reranked) == ["a", "b", "c"]
    assert all("score" in r for r in reranked)


@pytest.mark.asyncio
async def test_apply_reranking_skips_traces_without_results():
    """Traces with empty/absent results are left untouched (no crash)."""
    traces = [
        {"trace_id": "t2", "query": "x", "results": []},
        {"trace_id": "t3", "query": "y"},
    ]
    out = await apply_reranking_to_traces(traces, "multi_modal", {})
    assert out[0]["results"] == []
    assert "results" not in out[1]


@pytest.mark.asyncio
async def test_apply_reranking_unknown_strategy_leaves_results_unchanged():
    """Unknown strategy raises inside the service; the bridge logs and leaves
    that trace's results in their original order rather than crashing the run."""
    original = [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.2}]
    traces = [{"trace_id": "t4", "query": "q", "results": list(original)}]
    out = await apply_reranking_to_traces(traces, "nonexistent", {})
    assert out[0]["results"] == original


def test_to_rsr_parses_creation_timestamp_for_temporal_reranking():
    """_to_rsr must populate timestamp from creation_timestamp (epoch ms) so
    MultiModalReranker temporal scoring uses a real value instead of always
    returning the neutral 0.5 (the timestamp was never set before)."""
    from datetime import timezone

    from cogniverse_agents.search.rerank_service import _to_rsr

    # 1704067200000 ms == 2024-01-01T00:00:00Z
    rsr = _to_rsr({"id": "d1", "metadata": {"creation_timestamp": 1704067200000}})
    assert rsr.timestamp is not None
    assert rsr.timestamp.year == 2024
    assert rsr.timestamp.tzinfo == timezone.utc

    # Top-level creation_timestamp is honored too.
    assert (
        _to_rsr({"id": "d2", "creation_timestamp": 1704067200000}).timestamp is not None
    )

    # No timestamp data → None (temporal scoring falls back to neutral, by design).
    assert _to_rsr({"id": "d3"}).timestamp is None
