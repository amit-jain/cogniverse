"""
Unit tests for Inspect AI tools.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from cogniverse_core.evaluation.core.tools import phoenix_query_tool, video_search_tool


class TestVideoSearchTool:
    """Test video search tool functionality."""

    @pytest.mark.unit
    def test_video_search_tool_creation(self):
        """Test video search tool creation."""
        tool_func = video_search_tool()

        assert tool_func is not None
        assert callable(tool_func)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_video_search_tool_execution(self):
        """Test video search tool execution."""
        with (
            patch("cogniverse_runtime.search.service.SearchService") as mock_service_cls,
            patch("cogniverse_core.config.utils.get_config") as mock_config,
        ):

            # Setup mocks
            mock_config.return_value = {"test": "config"}
            mock_service = Mock()
            mock_service.search.return_value = [
                Mock(
                    to_dict=lambda: {
                        "document_id": "video1_frame_0",
                        "source_id": "video1",
                        "score": 0.9,
                        "content": "test content",
                        "metadata": {},
                    }
                ),
                Mock(
                    to_dict=lambda: {
                        "document_id": "video2_frame_0",
                        "source_id": "video2",
                        "score": 0.8,
                        "content": "test content 2",
                        "metadata": {},
                    }
                ),
            ]
            mock_service_cls.return_value = mock_service

            tool_func = video_search_tool()
            results = await tool_func(
                query="test query",
                profile="test_profile",
                strategy="test_strategy",
                top_k=10,
            )

            assert len(results) == 2
            assert results[0]["video_id"] == "video1"
            assert results[0]["score"] == 0.9
            assert results[0]["rank"] == 1
            assert results[1]["video_id"] == "video2"
            assert results[1]["score"] == 0.8
            assert results[1]["rank"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_video_search_tool_no_source_id(self):
        """Test video search tool when source_id is missing."""
        with (
            patch("cogniverse_runtime.search.service.SearchService") as mock_service_cls,
            patch("cogniverse_core.config.utils.get_config") as mock_config,
        ):

            mock_config.return_value = {"test": "config"}
            mock_service = Mock()
            mock_service.search.return_value = [
                Mock(to_dict=lambda: {"document_id": "video1_frame_5", "score": 0.9})
            ]
            mock_service_cls.return_value = mock_service

            tool_func = video_search_tool()
            results = await tool_func(
                query="test", profile="profile", strategy="strategy"
            )

            # Should extract video_id from document_id
            assert results[0]["video_id"] == "video1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_video_search_tool_no_score(self):
        """Test video search tool when score is missing."""
        with (
            patch("cogniverse_runtime.search.service.SearchService") as mock_service_cls,
            patch("cogniverse_core.config.utils.get_config") as mock_config,
        ):

            mock_config.return_value = {"test": "config"}
            mock_service = Mock()
            mock_service.search.return_value = [
                Mock(to_dict=lambda: {"document_id": "video1", "source_id": "video1"}),
                Mock(
                    to_dict=lambda: {
                        "document_id": "video2",
                        "source_id": "video2",
                        "score": 0.0,  # Zero score
                    }
                ),
            ]
            mock_service_cls.return_value = mock_service

            tool_func = video_search_tool()
            results = await tool_func(
                query="test", profile="profile", strategy="strategy"
            )

            # Should use rank-based score
            assert results[0]["score"] == 1.0  # 1/(0+1)
            assert results[1]["score"] == 0.5  # 1/(1+1)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_video_search_tool_error_handling(self):
        """Test video search tool error handling."""
        with (
            patch("cogniverse_runtime.search.service.SearchService") as mock_service_cls,
            patch("cogniverse_core.config.utils.get_config") as mock_config,
        ):

            mock_config.return_value = {"test": "config"}
            mock_service_cls.side_effect = Exception("Search service error")

            tool_func = video_search_tool()

            with pytest.raises(RuntimeError) as exc_info:
                await tool_func(query="test", profile="profile", strategy="strategy")

            assert "Search tool failed" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_video_search_tool_config_fallback(self):
        """Test config import fallback."""
        # Simply test that tool creation works
        tool_func = video_search_tool()
        assert tool_func is not None
        assert callable(tool_func)


class TestPhoenixQueryTool:
    """Test Phoenix query tool functionality."""

    @pytest.mark.unit
    def test_phoenix_query_tool_creation(self):
        """Test Phoenix query tool creation."""
        tool_func = phoenix_query_tool()

        assert tool_func is not None
        assert callable(tool_func)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_traces(self):
        """Test querying traces from Phoenix."""
        mock_px = MagicMock()
        mock_client = Mock()
        mock_df = pd.DataFrame(
            {
                "span_id": ["span1", "span2"],
                "name": ["search", "ranking"],
                "duration": [100, 50],
            }
        )
        mock_client.get_spans_dataframe.return_value = mock_df
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()
            results = await tool_func(
                query_type="traces", filter="name == 'search'", limit=10
            )

            assert isinstance(results, list)
            assert len(results) == 2
            assert results[0]["span_id"] == "span1"
            assert results[1]["span_id"] == "span2"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_datasets(self):
        """Test querying datasets from Phoenix."""
        mock_px = MagicMock()
        mock_client = Mock()
        mock_dataset = Mock()
        mock_dataset.name = "test_dataset"
        mock_example = Mock()
        mock_example.to_dict.return_value = {"input": "test", "output": "result"}
        mock_dataset.examples = [mock_example] * 5
        mock_client.get_dataset.return_value = mock_dataset
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()
            result = await tool_func(query_type="datasets", name="test_dataset")

            assert result["name"] == "test_dataset"
            assert result["num_examples"] == 5
            assert len(result["examples"]) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_datasets_no_name(self):
        """Test querying datasets without name."""
        mock_px = MagicMock()
        mock_client = Mock()
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()

            with pytest.raises(RuntimeError) as exc_info:
                await tool_func(query_type="datasets")

            assert "Dataset name is required" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_experiments(self):
        """Test querying experiments through spans."""
        mock_px = MagicMock()
        mock_client = Mock()

        # Test listing all experiments
        mock_df = Mock()
        mock_df.empty = False
        mock_df.columns = ["attributes.metadata.experiment_name", "other_column"]
        mock_df.__getitem__ = lambda self, key: Mock(
            dropna=lambda: Mock(
                unique=lambda: Mock(tolist=lambda: ["exp1", "exp2", "exp3"])
            )
        )
        mock_client.get_spans_dataframe.return_value = mock_df
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()

            # Test listing experiments
            result = await tool_func(query_type="experiments")
            assert result == {"experiments": ["exp1", "exp2", "exp3"], "count": 3}

            # Test getting specific experiment
            mock_df.to_dict.return_value = [{"trace": "data"}]
            mock_df.__len__ = lambda self: 5

            result = await tool_func(query_type="experiments", name="exp1")
            assert result == {
                "experiment_name": "exp1",
                "traces": [{"trace": "data"}],
                "count": 5,
            }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_unknown_type(self):
        """Test querying with unknown type."""
        mock_px = MagicMock()
        mock_client = Mock()
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()

            with pytest.raises(RuntimeError) as exc_info:
                await tool_func(query_type="unknown")

            assert "Unknown query type" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_error_handling(self):
        """Test Phoenix query error handling."""
        mock_px = MagicMock()
        mock_px.Client.side_effect = Exception("Phoenix connection failed")

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()

            # The function raises the original exception, not RuntimeError
            with pytest.raises(Exception) as exc_info:
                await tool_func(query_type="traces")

            assert "Phoenix connection failed" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_traces_with_params(self):
        """Test querying traces with all parameters."""
        from datetime import datetime

        mock_px = MagicMock()
        mock_client = Mock()
        mock_df = pd.DataFrame({"span_id": ["span1"]})
        mock_client.get_spans_dataframe.return_value = mock_df
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):
            tool_func = phoenix_query_tool()

            start_time = datetime.now()
            end_time = datetime.now()

            await tool_func(
                query_type="traces",
                filter="status == 'success'",
                start_time=start_time,
                end_time=end_time,
                limit=50,
            )

            # Check that parameters were passed
            mock_client.get_spans_dataframe.assert_called_once_with(
                filter_condition="status == 'success'",
                start_time=start_time,
                end_time=end_time,
                limit=50,
            )
