"""
Unit tests for ground truth extraction.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from cogniverse_core.evaluation.core.ground_truth import (
    BackendGroundTruthStrategy,
    DatasetGroundTruthStrategy,
    HybridGroundTruthStrategy,
    SchemaAwareGroundTruthStrategy,
    get_ground_truth_strategy,
)


class TestGroundTruthStrategies:
    """Test ground truth strategy selection and base functionality."""

    @pytest.mark.unit
    def test_get_ground_truth_strategy_schema_aware(self):
        """Test getting schema-aware strategy."""
        config = {"ground_truth_strategy": "schema_aware"}
        strategy = get_ground_truth_strategy(config)
        assert isinstance(strategy, SchemaAwareGroundTruthStrategy)

    @pytest.mark.unit
    def test_get_ground_truth_strategy_dataset(self):
        """Test getting dataset strategy."""
        config = {"ground_truth_strategy": "dataset"}
        strategy = get_ground_truth_strategy(config)
        assert isinstance(strategy, DatasetGroundTruthStrategy)

    @pytest.mark.unit
    def test_get_ground_truth_strategy_backend(self):
        """Test getting backend strategy."""
        config = {"ground_truth_strategy": "backend"}
        strategy = get_ground_truth_strategy(config)
        assert isinstance(strategy, BackendGroundTruthStrategy)

    @pytest.mark.unit
    def test_get_ground_truth_strategy_hybrid(self):
        """Test getting hybrid strategy."""
        config = {"ground_truth_strategy": "hybrid"}
        strategy = get_ground_truth_strategy(config)
        assert isinstance(strategy, HybridGroundTruthStrategy)

    @pytest.mark.unit
    def test_get_ground_truth_strategy_default(self):
        """Test getting default strategy."""
        config = {}
        strategy = get_ground_truth_strategy(config)
        assert isinstance(strategy, SchemaAwareGroundTruthStrategy)

    @pytest.mark.unit
    def test_get_ground_truth_strategy_unknown(self):
        """Test getting strategy with unknown type."""
        config = {"ground_truth_strategy": "unknown_type"}
        strategy = get_ground_truth_strategy(config)
        # Should default to schema-aware
        assert isinstance(strategy, SchemaAwareGroundTruthStrategy)


class TestSchemaAwareGroundTruthStrategy:
    """Test schema-aware ground truth extraction."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return SchemaAwareGroundTruthStrategy()

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = Mock()
        backend.schema_name = "test_schema"
        backend.search = AsyncMock(
            return_value=[
                {"id": "item1", "content": "test content 1"},
                {"id": "item2", "content": "test content 2"},
            ]
        )
        return backend

    @pytest.fixture
    def mock_analyzer(self):
        """Create mock schema analyzer."""
        analyzer = Mock()
        analyzer.analyze_query.return_value = {
            "query_type": "keyword",
            "constraints": {"keywords": ["test"]},
        }
        analyzer.extract_item_id.side_effect = lambda x: x.get("id")
        analyzer.get_expected_field_name.return_value = "expected_items"
        return analyzer

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_ground_truth_no_backend(self, strategy):
        """Test extraction without backend."""
        trace_data = {"query": "test query"}

        result = await strategy.extract_ground_truth(trace_data, backend=None)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "no_backend"
        assert "error" in result["metadata"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_ground_truth_no_query(self, strategy, mock_backend):
        """Test extraction without query."""
        trace_data = {}

        result = await strategy.extract_ground_truth(trace_data, backend=mock_backend)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "no_query"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_ground_truth_success(
        self, strategy, mock_backend, mock_analyzer
    ):
        """Test successful ground truth extraction."""
        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_schema_analyzer",
            return_value=mock_analyzer,
        ):
            trace_data = {
                "query": "test query",
                "metadata": {
                    "schema": "test_schema",
                    "fields": {"id_fields": ["id"], "content_fields": ["content"]},
                },
            }

            result = await strategy.extract_ground_truth(
                trace_data, backend=mock_backend
            )

            assert result["expected_items"] == ["item1", "item2"]
            assert result["confidence"] > 0.0
            assert result["source"] == "schema_aware_backend"
            assert result["metadata"]["schema"] == "test_schema"
            assert result["metadata"]["query_type"] == "keyword"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_ground_truth_schema_discovery_error(self, strategy):
        """Test extraction with schema discovery error."""
        backend = Mock()
        backend.schema_name = None
        backend.get_schema_name = Mock(side_effect=Exception("Schema error"))

        trace_data = {"query": "test query"}

        result = await strategy.extract_ground_truth(trace_data, backend=backend)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "schema_discovery_error"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_ground_truth_backend_error(self, strategy, mock_analyzer):
        """Test extraction with backend error."""
        backend = Mock()
        backend.schema_name = "test"
        backend.search = AsyncMock(side_effect=Exception("Backend error"))

        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_schema_analyzer",
            return_value=mock_analyzer,
        ):
            trace_data = {
                "query": "test query",
                "metadata": {"schema": "test", "fields": {"id_fields": ["id"]}},
            }

            result = await strategy.extract_ground_truth(trace_data, backend=backend)

            assert result["expected_items"] == []
            assert result["source"] in ["backend_error", "extraction_error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_with_constraints(self, strategy, mock_backend):
        """Test searching with constraints."""
        query_constraints = {
            "query_type": "keyword",
            "constraints": {"keywords": ["test"]},
        }
        schema_fields = {"id_fields": ["id"]}

        results = await strategy._search_with_constraints(
            mock_backend, "test query", query_constraints, schema_fields
        )

        assert len(results) == 2
        mock_backend.search.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_safe_async_call(self, strategy):
        """Test safe async call wrapper."""

        # Test with coroutine
        async def async_func():
            return "async_result"

        result = await strategy._safe_async_call(async_func())
        assert result == "async_result"

        # Test with regular value
        result = await strategy._safe_async_call("sync_result")
        assert result == "sync_result"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_discover_schema_fields(self, strategy):
        """Test schema field discovery."""
        backend = Mock()
        backend.get_schema_info = AsyncMock(
            return_value={
                "fields": {
                    "id": {"type": "keyword", "is_id": True},
                    "content": {"type": "text"},
                    "timestamp": {"type": "date"},
                }
            }
        )

        fields = await strategy._discover_schema_fields("test_schema", backend)

        assert "id" in fields["id_fields"]
        assert (
            "content" in fields["content_fields"] or "content" in fields["text_fields"]
        )
        assert "timestamp" in fields["temporal_fields"]


class TestDatasetGroundTruthStrategy:
    """Test dataset-based ground truth extraction."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return DatasetGroundTruthStrategy()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_from_dataset(self, strategy):
        """Test extraction from dataset."""
        # Mock phoenix module
        mock_px = MagicMock()
        mock_client = Mock()
        mock_dataset = Mock()

        # Create mock example with matching query
        mock_example = Mock()
        mock_example.input = {"query": "test query"}
        mock_example.output = {"expected_items": ["item1", "item2", "item3"]}

        mock_dataset.examples = [mock_example]
        mock_client.get_dataset.return_value = mock_dataset
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):

            trace_data = {
                "query": "test query",
                "metadata": {"dataset": "test_dataset"},
            }

            result = await strategy.extract_ground_truth(trace_data)

            assert result["expected_items"] == ["item1", "item2", "item3"]
            assert result["confidence"] > 0.9
            assert result["source"] == "dataset"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_no_dataset_match(self, strategy):
        """Test extraction when no dataset match found."""
        trace_data = {"query": "unknown query", "metadata": {}}

        result = await strategy.extract_ground_truth(trace_data)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "no_dataset"


class TestBackendGroundTruthStrategy:
    """Test backend-based ground truth extraction."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return BackendGroundTruthStrategy()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_from_backend(self, strategy):
        """Test extraction directly from backend."""
        backend = Mock()
        backend.schema_name = "test"
        backend.search = AsyncMock(
            return_value=[
                {"id": "item1", "content": "test content 1"},
                {"id": "item2", "content": "test content 2"},
            ]
        )

        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_schema_analyzer"
        ) as mock_analyzer:
            analyzer = Mock()
            analyzer.analyze_query.return_value = {"query_type": "keyword"}
            analyzer.extract_item_id.side_effect = lambda x: x.get("id")
            analyzer.get_expected_field_name.return_value = "expected_items"
            mock_analyzer.return_value = analyzer

            trace_data = {
                "query": "test query",
                "metadata": {
                    "high_precision": True,
                    "schema": "test",
                    "fields": {"id_fields": ["id"]},
                },
            }

            result = await strategy.extract_ground_truth(trace_data, backend=backend)

            assert result["expected_items"] == ["item1", "item2"]
            assert result["confidence"] > 0
            assert "backend" in result["source"] or "schema_aware" in result["source"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_backend_not_supported(self, strategy):
        """Test extraction when backend doesn't support ground truth."""
        backend = Mock()
        backend.schema_name = None
        backend.get_schema_name = Mock(side_effect=Exception("Not supported"))

        trace_data = {"query": "test query"}

        result = await strategy.extract_ground_truth(trace_data, backend=backend)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert "error" in result["source"] or "discovery" in result["source"]


class TestHybridGroundTruthStrategy:
    """Test hybrid ground truth extraction."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return HybridGroundTruthStrategy()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_with_high_confidence_dataset(self, strategy):
        """Test extraction when dataset has high confidence."""
        # The hybrid strategy will try multiple strategies
        backend = Mock()
        backend.schema_name = "test"
        backend.search = AsyncMock(
            return_value=[
                {"id": "item1", "content": "test content 1"},
                {"id": "item2", "content": "test content 2"},
            ]
        )

        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_schema_analyzer"
        ) as mock_analyzer:
            analyzer = Mock()
            analyzer.analyze_query.return_value = {"query_type": "keyword"}
            analyzer.extract_item_id.side_effect = lambda x: x.get("id")
            analyzer.get_expected_field_name.return_value = "expected_items"
            mock_analyzer.return_value = analyzer

            trace_data = {
                "query": "test",
                "metadata": {"schema": "test", "fields": {"id_fields": ["id"]}},
            }

            result = await strategy.extract_ground_truth(trace_data, backend)

            # Should get results from one of the strategies
            assert "expected_items" in result
            assert result["confidence"] >= 0
            assert "hybrid" in result["source"] or "strategies_tried" in result.get(
                "metadata", {}
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_merge_results(self, strategy):
        """Test merging results from multiple sources."""
        # Create sub-strategies
        strategy.dataset_strategy = DatasetGroundTruthStrategy()
        strategy.backend_strategy = BackendGroundTruthStrategy()
        strategy.schema_strategy = SchemaAwareGroundTruthStrategy()

        # No dataset match, will use schema strategy fallback
        strategy.dataset_strategy.dataset_cache = {}

        # Mock backend without ground truth support
        backend = Mock()

        trace_data = {"query": "test"}

        result = await strategy.extract_ground_truth(trace_data, backend)

        # Should try multiple strategies and merge
        assert "expected_items" in result
        assert "source" in result


class TestAdditionalGroundTruthMethods:
    """Test additional ground truth methods for better coverage."""

    @pytest.fixture
    def strategy(self):
        """Create schema-aware strategy instance."""
        return SchemaAwareGroundTruthStrategy()

    @pytest.mark.unit
    def test_parse_field_mappings_with_existing_format(self, strategy):
        """Test parsing field mappings that are already in correct format."""
        mappings = {
            "id_fields": ["id", "_id"],
            "content_fields": ["content", "body"],
            "metadata_fields": ["meta"],
        }

        result = strategy._parse_field_mappings(mappings)

        assert result == mappings

    @pytest.mark.unit
    def test_parse_field_mappings_from_flat_list(self, strategy):
        """Test parsing field mappings from flat list."""
        mappings = {"_id": "string", "content": "text", "timestamp": "date"}

        result = strategy._parse_field_mappings(mappings)

        assert "_id" in result["id_fields"]
        assert "content" in result["content_fields"]
        assert "timestamp" in result["temporal_fields"]

    @pytest.mark.unit
    def test_categorize_fields(self, strategy):
        """Test field categorization."""
        field_list = [
            "video_id",
            "title",
            "description",
            "created_at",
            "score",
            "metadata_json",
        ]

        result = strategy._categorize_fields(field_list)

        assert "video_id" in result["id_fields"]
        assert "title" in result["text_fields"]
        assert "description" in result["content_fields"]
        assert "created_at" in result["temporal_fields"]
        assert "score" in result["numeric_fields"]
        assert "metadata_json" in result["metadata_fields"]

    @pytest.mark.unit
    def test_infer_fields_from_dict_result(self, strategy):
        """Test inferring fields from dictionary result."""
        result = {
            "_id": "123",
            "title": "Test Video",
            "description": "A long description that is over 100 characters to test content field detection. This should be categorized as content.",
            "timestamp": "2024-01-01",
            "score": 0.95,
            "metadata": {"tags": ["test"]},
        }

        fields = strategy._infer_fields_from_results(result)

        assert "_id" in fields["id_fields"]
        assert "title" in fields["text_fields"]
        assert "description" in fields["content_fields"]
        assert "timestamp" in fields["temporal_fields"]
        assert "score" in fields["numeric_fields"]
        assert "metadata" in fields["metadata_fields"]

    @pytest.mark.unit
    def test_infer_fields_from_object_result(self, strategy):
        """Test inferring fields from object with attributes."""
        result = Mock()
        result.__dict__ = {
            "id": "456",
            "content": "Short text",
            "date_created": "2024-01-01",
            "rank": 1,
        }

        fields = strategy._infer_fields_from_results(result)

        assert "id" in fields["id_fields"]
        assert "content" in fields["text_fields"]  # Short text goes to text_fields
        assert "date_created" in fields["temporal_fields"]
        assert "rank" in fields["numeric_fields"]

    @pytest.mark.unit
    def test_calculate_confidence_various_scenarios(self, strategy):
        """Test confidence calculation with various scenarios."""
        # Test with no results
        conf = strategy._calculate_confidence({"query_type": "generic"}, 0, 0, 0)
        assert conf == 0.0

        # Test with perfect extraction
        conf = strategy._calculate_confidence({"query_type": "exact"}, 10, 0, 10)
        assert conf > 0.8

        # Test with errors
        conf = strategy._calculate_confidence({"query_type": "generic"}, 5, 2, 10)
        assert conf < 0.5

        # Test with too many results
        conf = strategy._calculate_confidence({"query_type": "generic"}, 60, 0, 60)
        assert conf < 0.3

        # Test with field constraints
        conf = strategy._calculate_confidence(
            {
                "query_type": "generic",
                "field_constraints": {"id": "123", "type": "video", "status": "active"},
            },
            5,
            0,
            5,
        )
        assert conf > 0.5  # Higher confidence due to structured query

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_info_from_metadata(self, strategy):
        """Test getting schema info from trace metadata."""
        trace_data = {
            "query": "test",
            "metadata": {
                "schema": "video_schema",
                "fields": {"id_fields": ["video_id"], "content_fields": ["transcript"]},
            },
        }

        backend = Mock()

        schema_info = await strategy._get_schema_info(trace_data, backend)

        assert schema_info["name"] == "video_schema"
        assert schema_info["fields"]["id_fields"] == ["video_id"]
        assert schema_info["fields"]["content_fields"] == ["transcript"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_discover_schema_fields_with_field_mappings(self, strategy):
        """Test discovering schema fields using field mappings."""
        backend = Mock()
        backend.get_schema_info = AsyncMock(side_effect=Exception("Not available"))
        backend.get_field_mappings = AsyncMock(
            return_value={"_id": "keyword", "title": "text", "created": "date"}
        )

        fields = await strategy._discover_schema_fields("test_schema", backend)

        assert "_id" in fields["id_fields"]
        assert "title" in fields["text_fields"]
        assert "created" in fields["temporal_fields"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_discover_schema_fields_with_sample_query(self, strategy):
        """Test discovering schema fields using sample query."""
        backend = Mock()
        backend.get_schema_info = AsyncMock(side_effect=Exception("Not available"))
        backend.get_field_mappings = AsyncMock(side_effect=Exception("Not available"))
        backend.search = AsyncMock(
            return_value=[
                {"id": "123", "content": "Test content", "timestamp": "2024-01-01"}
            ]
        )

        fields = await strategy._discover_schema_fields("test_schema", backend)

        assert "id" in fields["id_fields"]
        assert "content" in fields["text_fields"]
        assert "timestamp" in fields["temporal_fields"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_discover_schema_fields_with_list_fields(self, strategy):
        """Test discovering schema fields using list_fields method."""
        backend = Mock()
        backend.get_schema_info = AsyncMock(side_effect=Exception("Not available"))
        backend.get_field_mappings = AsyncMock(side_effect=Exception("Not available"))
        backend.search = AsyncMock(side_effect=Exception("Not available"))
        backend.list_fields = AsyncMock(
            return_value=[
                "document_id",
                "text_content",
                "updated_time",
                "relevance_score",
            ]
        )

        fields = await strategy._discover_schema_fields("test_schema", backend)

        assert "document_id" in fields["id_fields"]
        assert "text_content" in fields["content_fields"]
        assert "updated_time" in fields["temporal_fields"]
        assert "relevance_score" in fields["numeric_fields"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_with_constraints_with_all_params(self, strategy):
        """Test search with all constraint types."""
        backend = Mock()
        backend.search = AsyncMock(
            return_value=[
                {"id": "1", "content": "Result 1"},
                {"id": "2", "content": "Result 2"},
            ]
        )

        constraints = {
            "max_results": 20,
            "field_constraints": {"type": "video"},
            "temporal_constraints": {"after": "2024-01-01"},
        }

        results = await strategy._search_with_constraints(
            backend, "test query", constraints, {"id_fields": ["id"]}
        )

        assert len(results) == 2
        backend.search.assert_called_once()
        call_kwargs = backend.search.call_args[1]
        assert call_kwargs["query_text"] == "test query"
        assert call_kwargs["top_k"] == 20
        assert call_kwargs["field_constraints"] == {"type": "video"}
        assert call_kwargs["temporal_constraints"] == {"after": "2024-01-01"}


class TestDatasetGroundTruthStrategyExtended:
    """Extended tests for DatasetGroundTruthStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return DatasetGroundTruthStrategy()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_from_local_file(self, strategy):
        """Test extraction from local JSON file."""
        # Mock phoenix module to fail
        mock_px = MagicMock()
        mock_px.Client.side_effect = Exception("Phoenix not available")

        with patch.dict("sys.modules", {"phoenix": mock_px}):

            # Mock file operations
            dataset_content = {
                "queries": [
                    {"query": "test query", "expected_items": ["item1", "item2"]}
                ]
            }

            with (
                patch("os.path.exists", return_value=True),
                patch(
                    "builtins.open", mock_open(read_data=json.dumps(dataset_content))
                ),
            ):

                trace_data = {
                    "query": "test query",
                    "metadata": {"dataset": "test_dataset"},
                }

                result = await strategy.extract_ground_truth(trace_data)

                assert result["expected_items"] == ["item1", "item2"]
                assert result["confidence"] == 0.9
                assert result["source"] == "dataset_file"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_with_alternative_field_names(self, strategy):
        """Test extraction trying alternative field names."""
        # Mock phoenix module
        mock_px = MagicMock()
        mock_client = Mock()
        mock_dataset = Mock()

        # Create mock example with alternative field names
        mock_example = Mock()
        mock_example.input = {"query": "test query"}
        mock_example.output = {"expected_videos": ["video1", "video2"]}

        mock_dataset.examples = [mock_example]
        mock_client.get_dataset.return_value = mock_dataset
        mock_px.Client.return_value = mock_client

        with patch.dict("sys.modules", {"phoenix": mock_px}):

            trace_data = {
                "query": "test query",
                "metadata": {"dataset": "test_dataset"},
            }

            result = await strategy.extract_ground_truth(trace_data)

            assert result["expected_items"] == ["video1", "video2"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_dataset_error(self, strategy):
        """Test extraction with dataset error."""
        # Mock phoenix module to fail
        mock_px = MagicMock()
        mock_px.Client.side_effect = Exception("Connection failed")

        with patch.dict("sys.modules", {"phoenix": mock_px}):

            with patch("os.path.exists", return_value=False):
                trace_data = {
                    "query": "test query",
                    "metadata": {"dataset": "test_dataset"},
                }

                result = await strategy.extract_ground_truth(trace_data)

                assert result["expected_items"] == []
                assert result["confidence"] == 0.0
                assert result["source"] == "dataset_no_match"


class TestHybridGroundTruthStrategyExtended:
    """Extended tests for HybridGroundTruthStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return HybridGroundTruthStrategy()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_strategies_fail(self, strategy):
        """Test when all strategies fail."""
        backend = Mock()
        backend.schema_name = None
        backend.get_schema_name = Mock(side_effect=Exception("Error"))

        trace_data = {"query": "test"}

        result = await strategy.extract_ground_truth(trace_data, backend)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "all_strategies_failed"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_best_confidence_selection(self, strategy):
        """Test that hybrid selects result with best confidence."""
        backend = Mock()
        backend.schema_name = "test"
        backend.search = AsyncMock(
            return_value=[{"id": "item1"}, {"id": "item2"}, {"id": "item3"}]
        )

        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_schema_analyzer"
        ) as mock_analyzer:
            analyzer = Mock()
            analyzer.analyze_query.return_value = {"query_type": "exact"}
            analyzer.extract_item_id.side_effect = lambda x: x.get("id")
            analyzer.get_expected_field_name.return_value = "expected_items"
            mock_analyzer.return_value = analyzer

            trace_data = {
                "query": "test",
                "metadata": {"schema": "test", "fields": {"id_fields": ["id"]}},
            }

            result = await strategy.extract_ground_truth(trace_data, backend)

            # Should select best result
            assert len(result["expected_items"]) > 0
            assert result["confidence"] > 0
            assert "hybrid" in result["source"]
            assert result["metadata"]["strategies_tried"] >= 1
