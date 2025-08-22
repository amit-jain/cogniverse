"""
Unit tests for evaluation solvers.
"""

from unittest.mock import Mock, patch

import pytest

from src.evaluation.core.solvers import (
    create_batch_solver,
    create_live_solver,
    create_retrieval_solver,
)


class TestRetrievalSolver:
    """Test retrieval solver."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_basic(self, mock_search_service):
        """Test basic retrieval solver functionality."""
        with patch(
            "src.app.search.service.SearchService", return_value=mock_search_service
        ):
            with patch(
                "src.common.config.get_config",
                return_value={"vespa_url": "http://localhost"},
            ):
                solver = create_retrieval_solver(
                    profiles=["profile1"], strategies=["strategy1"], config={"top_k": 5}
                )

                # Create mock state
                state = Mock()
                state.input = {"query": "test query 1"}
                state.outputs = {}
                state.metadata = {}  # Real dict for metadata
                generate = Mock()

                # Run solver
                result = await solver(state, generate)

                assert result is not None
                # Should have results in metadata
                assert "search_results" in result.metadata
                assert len(result.metadata["search_results"]) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_multiple_configs(self, mock_search_service):
        """Test retrieval solver with multiple profiles and strategies."""
        with patch(
            "src.app.search.service.SearchService", return_value=mock_search_service
        ):
            with patch(
                "src.common.config.get_config",
                return_value={"vespa_url": "http://localhost"},
            ):
                solver = create_retrieval_solver(
                    profiles=["profile1", "profile2"],
                    strategies=["strategy1", "strategy2"],
                    config={},
                )

                state = Mock()
                state.input = {"query": "test query"}
                state.outputs = {}
                state.metadata = {}  # Real dict for metadata
                generate = Mock()

                result = await solver(state, generate)

                # Should have 4 results (2 profiles * 2 strategies) in metadata
                assert "search_results" in result.metadata
                assert len(result.metadata["search_results"]) == 4

                # Check output keys
                expected_keys = [
                    "profile1_strategy1",
                    "profile1_strategy2",
                    "profile2_strategy1",
                    "profile2_strategy2",
                ]
                for key in expected_keys:
                    assert key in result.metadata["search_results"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_with_tracing_config(self, mock_search_service):
        """Test retrieval solver with tracing config (tracing not actually implemented)."""
        with patch(
            "src.app.search.service.SearchService", return_value=mock_search_service
        ):
            with patch(
                "src.common.config.get_config",
                return_value={"vespa_url": "http://localhost"},
            ):
                # Note: enable_tracing config is accepted but not used in current implementation
                solver = create_retrieval_solver(
                    profiles=["profile1"],
                    strategies=["strategy1"],
                    config={"enable_tracing": True},
                )

                state = Mock()
                state.input = {"query": "test"}
                state.outputs = {}
                state.metadata = {}  # Real dict for metadata
                generate = Mock()

                result = await solver(state, generate)

                # Solver should still work even with tracing config
                assert result is not None
                assert "search_results" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_error_handling(self):
        """Test retrieval solver error handling."""
        with patch("src.app.search.service.SearchService") as mock_service:
            with patch(
                "src.common.config.get_config",
                return_value={"vespa_url": "http://localhost"},
            ):
                # Simulate search failure
                mock_service.return_value.search.side_effect = Exception(
                    "Search failed"
                )

                solver = create_retrieval_solver(
                    profiles=["profile1"], strategies=["strategy1"], config={}
                )

                state = Mock()
                state.input = {"query": "test"}
                state.outputs = {}
                state.metadata = {}
                generate = Mock()

                # Should handle error gracefully
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
            # Should have loaded traces
            mock_phoenix_client.get_spans_dataframe.assert_called()

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
            mock_phoenix_client.get_spans_dataframe.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_empty_result(self, mock_phoenix_client):
        """Test handling empty trace results."""
        import pandas as pd

        mock_phoenix_client.get_spans_dataframe.return_value = pd.DataFrame()

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
                # Make it stop after one iteration
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

                # Should have polled at least once
                mock_phoenix_client.get_spans_dataframe.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_live_trace_solver_single_poll(self, mock_phoenix_client):
        """Test single poll mode."""
        with patch("phoenix.Client", return_value=mock_phoenix_client):
            solver = create_live_solver(config={"continuous": False})

            state = Mock()
            state.outputs = {}
            state.metadata = {}  # Make metadata a real dict
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            # Live solver polls continuously, so may be called multiple times
            assert mock_phoenix_client.get_spans_dataframe.called
