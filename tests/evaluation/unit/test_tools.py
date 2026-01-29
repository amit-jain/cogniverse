"""
Unit tests for Inspect AI tools.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from cogniverse_evaluation.core.tools import phoenix_query_tool, video_search_tool


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
            patch(
                "cogniverse_runtime.search.service.SearchService"
            ) as mock_service_cls,
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
            patch(
                "cogniverse_runtime.search.service.SearchService"
            ) as mock_service_cls,
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
            patch(
                "cogniverse_runtime.search.service.SearchService"
            ) as mock_service_cls,
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
            patch(
                "cogniverse_runtime.search.service.SearchService"
            ) as mock_service_cls,
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
        from unittest.mock import AsyncMock

        mock_df = pd.DataFrame(
            {
                "span_id": ["span1", "span2"],
                "name": ["search", "ranking"],
                "duration": [100, 50],
            }
        )

        mock_provider = Mock()
        mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=mock_df)

        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider",
            return_value=mock_provider,
        ):
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
        from unittest.mock import AsyncMock

        mock_dataset_data = {
            "name": "test_dataset",
            "num_examples": 5,
            "examples": [{"input": "test", "output": "result"}] * 5,
        }

        mock_provider = Mock()
        mock_provider.telemetry.datasets.get_dataset = AsyncMock(
            return_value=mock_dataset_data
        )

        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider",
            return_value=mock_provider,
        ):
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
        from unittest.mock import AsyncMock

        # Test listing all experiments
        mock_df_list = pd.DataFrame(
            {
                "attributes.metadata.experiment_name": ["exp1", "exp2", "exp3", "exp1"],
                "other_column": [1, 2, 3, 4],
            }
        )

        # Test getting specific experiment
        mock_df_specific = pd.DataFrame(
            {"trace": ["data"] * 5, "attributes.metadata.experiment_name": ["exp1"] * 5}
        )

        mock_provider = Mock()

        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider",
            return_value=mock_provider,
        ):
            tool_func = phoenix_query_tool()

            # Test listing experiments
            mock_provider.telemetry.traces.get_spans = AsyncMock(
                return_value=mock_df_list
            )
            result = await tool_func(query_type="experiments")
            assert result == {"experiments": ["exp1", "exp2", "exp3"], "count": 3}

            # Test getting specific experiment
            mock_provider.telemetry.traces.get_spans = AsyncMock(
                return_value=mock_df_specific
            )
            result = await tool_func(query_type="experiments", name="exp1")
            assert result == {
                "experiment_name": "exp1",
                "traces": mock_df_specific.to_dict(orient="records"),
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
        from unittest.mock import AsyncMock

        mock_provider = Mock()
        mock_provider.telemetry.traces.get_spans = AsyncMock(
            side_effect=Exception("Phoenix connection failed")
        )

        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider",
            return_value=mock_provider,
        ):
            tool_func = phoenix_query_tool()

            # The function raises RuntimeError wrapping the original exception
            with pytest.raises(RuntimeError) as exc_info:
                await tool_func(query_type="traces")

            assert "Phoenix query tool failed" in str(exc_info.value)
            assert "Phoenix connection failed" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_query_traces_with_params(self):
        """Test querying traces with all parameters."""
        from datetime import datetime
        from unittest.mock import AsyncMock

        mock_df = pd.DataFrame({"span_id": ["span1"]})
        mock_provider = Mock()
        mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=mock_df)

        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider",
            return_value=mock_provider,
        ):
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

            # Check that parameters were passed to the async method
            mock_provider.telemetry.traces.get_spans.assert_called_once()
            call_kwargs = mock_provider.telemetry.traces.get_spans.call_args[1]
            assert call_kwargs["start_time"] == start_time
            assert call_kwargs["end_time"] == end_time
            assert call_kwargs["limit"] == 50
