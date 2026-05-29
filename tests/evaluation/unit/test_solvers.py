"""
Unit tests for evaluation solvers.
"""

from unittest.mock import Mock, patch

import pytest

from cogniverse_evaluation.core.solvers import (
    create_batch_solver,
    create_live_solver,
    create_retrieval_solver,
)


def _mock_httpx_response(results_count: int = 3, status_code: int = 200):
    """Create a mock httpx.Response with search results."""
    results = [
        {
            "document_id": f"video_{i}_frame_{i * 10}",
            "source_id": f"video_{i}",
            "score": 0.9 - i * 0.1,
            "content": f"Test result {i}",
        }
        for i in range(results_count)
    ]
    response = Mock()
    response.status_code = status_code
    response.json.return_value = {"results": results, "count": results_count}
    response.raise_for_status.return_value = None
    return response


class TestRetrievalSolver:
    """Test retrieval solver.

    The solver uses httpx.post to call the runtime search API directly.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_basic(self):
        """Test basic retrieval solver functionality."""
        mock_response = _mock_httpx_response(results_count=3)

        with patch("httpx.post", return_value=mock_response) as mock_post:
            solver = create_retrieval_solver(
                profiles=["profile1"], strategies=["strategy1"], config={"top_k": 5}
            )

            state = Mock()
            state.input = {"query": "test query 1"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            assert "search_results" in result.metadata
            assert len(result.metadata["search_results"]) == 1
            assert "profile1_strategy1" in result.metadata["search_results"]
            entry = result.metadata["search_results"]["profile1_strategy1"]
            assert entry["success"] is True
            assert entry["count"] == 3
            assert entry["profile"] == "profile1"
            assert entry["strategy"] == "strategy1"

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert request_body["query"] == "test query 1"
            assert request_body["profile"] == "profile1"
            assert request_body["ranking_strategy"] == "strategy1"
            assert request_body["top_k"] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_multiple_configs(self):
        """Test retrieval solver with multiple profiles and strategies."""
        mock_response = _mock_httpx_response(results_count=2)

        with patch("httpx.post", return_value=mock_response) as mock_post:
            solver = create_retrieval_solver(
                profiles=["profile1", "profile2"],
                strategies=["strategy1", "strategy2"],
                config={},
            )

            state = Mock()
            state.input = {"query": "test query"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert "search_results" in result.metadata
            assert len(result.metadata["search_results"]) == 4

            expected_keys = [
                "profile1_strategy1",
                "profile1_strategy2",
                "profile2_strategy1",
                "profile2_strategy2",
            ]
            for key in expected_keys:
                assert key in result.metadata["search_results"]
                assert result.metadata["search_results"][key]["success"] is True

            assert mock_post.call_count == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_with_tracing_config(self):
        """Test retrieval solver with tracing config."""
        mock_response = _mock_httpx_response(results_count=1)

        with patch("httpx.post", return_value=mock_response):
            solver = create_retrieval_solver(
                profiles=["profile1"],
                strategies=["strategy1"],
                config={"enable_tracing": True},
            )

            state = Mock()
            state.input = {"query": "test"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            assert "search_results" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_error_handling(self):
        """Test retrieval solver error handling when httpx.post fails."""
        with patch("httpx.post", side_effect=Exception("Connection refused")):
            solver = create_retrieval_solver(
                profiles=["profile1"], strategies=["strategy1"], config={}
            )

            state = Mock()
            state.input = {"query": "test"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)
            assert result is not None
            assert "search_results" in result.metadata
            entry = result.metadata["search_results"]["profile1_strategy1"]
            assert entry["success"] is False
            assert "Connection refused" in entry["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_string_input(self):
        """Test retrieval solver with string input instead of dict."""
        mock_response = _mock_httpx_response(results_count=1)

        with patch("httpx.post", return_value=mock_response):
            solver = create_retrieval_solver(
                profiles=["p1"], strategies=["s1"], config={}
            )

            state = Mock()
            state.input = "direct string query"
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)
            assert result is not None
            assert "search_results" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_empty_query(self):
        """Test retrieval solver with empty query returns early."""
        solver = create_retrieval_solver(profiles=["p1"], strategies=["s1"], config={})

        state = Mock()
        state.input = {"query": ""}
        state.outputs = {}
        state.metadata = {}
        generate = Mock()

        result = await solver(state, generate)
        assert result is not None


def _populated_traces_df():
    """A two-row spans DataFrame matching the columns the solver reads."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "trace_id": "trace-a",
                "attributes.input.value": "what is a quark",
                "attributes.output.value": ["v1", "v2"],
                "attributes.metadata.profile": "frame_based_colpali",
                "attributes.metadata.strategy": "binary_binary",
                "attributes.metadata": {"profile": "frame_based_colpali"},
                "timestamp": "2026-01-01T00:00:00Z",
                "duration_ms": 1100,
            },
            {
                "trace_id": "trace-b",
                "attributes.input.value": "explain entanglement",
                "attributes.output.value": ["v3"],
                "attributes.metadata.profile": "videoprism_global",
                "attributes.metadata.strategy": "float_float",
                "attributes.metadata": {"profile": "videoprism_global"},
                "timestamp": "2026-01-01T00:00:30Z",
                "duration_ms": 900,
            },
        ]
    )


def _seed_solver_provider(monkeypatch, df):
    """Patch the solver's provider + ground-truth + search-service so the
    happy path executes against a populated DataFrame instead of the
    autouse empty-DF mock from conftest."""
    from unittest.mock import AsyncMock, MagicMock

    mock_provider = MagicMock()
    mock_provider.telemetry = MagicMock()
    mock_provider.telemetry.traces = MagicMock()

    async def _spans(**_kwargs):
        return df

    mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=_spans)

    monkeypatch.setattr(
        "cogniverse_evaluation.providers.get_evaluation_provider",
        lambda: mock_provider,
    )

    fake_strategy = MagicMock()

    async def _extract(trace_data, _backend):
        return {
            "expected_items": [f"gt-{trace_data['trace_id']}"],
            "confidence": 0.9,
            "source": "fixture",
        }

    fake_strategy.extract_ground_truth = AsyncMock(side_effect=_extract)
    monkeypatch.setattr(
        "cogniverse_evaluation.core.ground_truth.get_ground_truth_strategy",
        lambda *_a, **_kw: fake_strategy,
        raising=False,
    )
    monkeypatch.setattr(
        "cogniverse_agents.search.service.SearchService",
        lambda *_a, **_kw: MagicMock(backend=MagicMock()),
        raising=False,
    )
    return mock_provider


class TestBatchSolver:
    """Batch solver loads + ground-truth-enriches traces from Phoenix."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_with_ids_loads_exact_matches(self, monkeypatch):
        _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(trace_ids=["trace-a"], config={})

        state = Mock()
        state.outputs = {}
        state.metadata = {}
        result = await solver(state, Mock())

        assert [t["trace_id"] for t in result.metadata["loaded_traces"]] == ["trace-a"]
        loaded = result.metadata["loaded_traces"][0]
        for key in (
            "trace_id",
            "query",
            "results",
            "profile",
            "strategy",
            "timestamp",
            "duration_ms",
            "metadata",
            "ground_truth",
            "ground_truth_confidence",
            "ground_truth_source",
        ):
            assert key in loaded
        assert loaded["ground_truth"] == ["gt-trace-a"]
        assert loaded["ground_truth_confidence"] == 0.9
        assert loaded["ground_truth_source"] == "fixture"
        stats = result.metadata["ground_truth_stats"]
        assert stats["total_traces"] == 1
        assert stats["traces_with_ground_truth"] == 1
        assert stats["average_confidence"] == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_recent_traces_loads_full_set(self, monkeypatch):
        _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(trace_ids=None, config={"hours_back": 1})

        state = Mock()
        state.outputs = {}
        state.metadata = {}
        result = await solver(state, Mock())

        ids = sorted(t["trace_id"] for t in result.metadata["loaded_traces"])
        assert ids == ["trace-a", "trace-b"]
        stats = result.metadata["ground_truth_stats"]
        assert stats["total_traces"] == 2
        assert stats["traces_with_ground_truth"] == 2
        assert stats["average_confidence"] == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_empty_result_short_circuits_with_no_data(
        self, monkeypatch
    ):
        import pandas as pd

        _seed_solver_provider(monkeypatch, pd.DataFrame())
        solver = create_batch_solver(trace_ids=None, config={"hours_back": 1})

        state = Mock()
        state.outputs = {}
        state.metadata = {}
        result = await solver(state, Mock())

        assert result.output.completion == "No traces found"
        assert "loaded_traces" not in result.metadata


class TestLiveSolver:
    """Live solver collects bounded iterations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_poll_collects_documented_trace_data_shape(self, monkeypatch):
        _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_live_solver(
            config={"continuous": False, "max_iterations": 1, "poll_interval": 0}
        )

        state = Mock()
        state.outputs = {}
        state.metadata = {}
        result = await solver(state, Mock())

        traces = result.metadata["live_traces"]
        assert [t["trace_id"] for t in traces] == ["trace-a", "trace-b"]
        for t in traces:
            assert set(t.keys()) >= {"trace_id", "query", "results", "timestamp"}
        assert "2" in result.output.completion

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bounded_iteration_count_does_not_hang_on_empty_polls(
        self, monkeypatch
    ):
        import pandas as pd

        _seed_solver_provider(monkeypatch, pd.DataFrame())
        solver = create_live_solver(
            config={"continuous": False, "max_iterations": 3, "poll_interval": 0}
        )

        state = Mock()
        state.outputs = {}
        state.metadata = {}
        result = await solver(state, Mock())
        assert result.metadata["live_traces"] == []
