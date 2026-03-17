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


class TestBatchSolver:
    """Test batch solver for Phoenix traces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_with_ids(self, mock_phoenix_client):
        """Test loading specific trace IDs."""
        with patch("phoenix.Client", return_value=mock_phoenix_client):
            solver = create_batch_solver(trace_ids=["trace1", "trace2"], config={})

            state = Mock()
            state.outputs = {}
            state.metadata = {}
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_recent_traces(self, mock_phoenix_client):
        """Test loading recent traces."""
        with patch("phoenix.Client", return_value=mock_phoenix_client):
            solver = create_batch_solver(trace_ids=None, config={"hours_back": 1})

            state = Mock()
            state.outputs = {}
            state.metadata = {}
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_empty_result(self, mock_phoenix_client):
        """Test handling empty trace results."""

        with patch("phoenix.Client", return_value=mock_phoenix_client):
            solver = create_batch_solver(trace_ids=None, config={"hours_back": 1})

            state = Mock()
            state.outputs = {}
            state.metadata = {}
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            assert len(result.outputs) == 0


class TestLiveSolver:
    """Test live solver for real-time traces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_live_trace_solver_continuous(self, mock_phoenix_client):
        """Test continuous polling for new traces."""
        with patch("phoenix.Client", return_value=mock_phoenix_client):
            with patch("asyncio.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                solver = create_live_solver(
                    config={"poll_interval": 1, "continuous": True}
                )

                state = Mock()
                state.outputs = {}
                state.metadata = {}
                generate = Mock()

                try:
                    await solver(state, generate)
                except KeyboardInterrupt:
                    pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_live_trace_solver_single_poll(self, mock_phoenix_client):
        """Test single poll mode."""
        with patch("phoenix.Client", return_value=mock_phoenix_client):
            solver = create_live_solver(config={"continuous": False})

            state = Mock()
            state.outputs = {}
            state.metadata = {}
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
