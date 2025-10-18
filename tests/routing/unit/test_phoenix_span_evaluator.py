"""
Unit tests for PhoenixSpanEvaluator

Tests the extraction of routing experiences from Phoenix spans and
feeding them to the AdvancedRoutingOptimizer.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from cogniverse_agents.routing.phoenix_span_evaluator import PhoenixSpanEvaluator


@pytest.fixture
def mock_optimizer():
    """Create mock AdvancedRoutingOptimizer"""
    optimizer = Mock()
    optimizer.record_routing_experience = AsyncMock(return_value=0.8)
    return optimizer


@pytest.fixture
def mock_phoenix_client():
    """Create mock Phoenix client"""
    client = Mock()
    return client


@pytest.fixture
def span_evaluator(mock_optimizer, mock_phoenix_client):
    """Create PhoenixSpanEvaluator with mocked dependencies"""
    with patch(
        "cogniverse_agents.routing.phoenix_span_evaluator.px.Client",
        return_value=mock_phoenix_client,
    ):
        evaluator = PhoenixSpanEvaluator(
            optimizer=mock_optimizer, tenant_id="test-tenant"
        )
    return evaluator


class TestPhoenixSpanEvaluatorInit:
    """Test PhoenixSpanEvaluator initialization"""

    def test_initialization_default_tenant(self, mock_optimizer, mock_phoenix_client):
        """Test initialization with default tenant"""
        with patch(
            "cogniverse_agents.routing.phoenix_span_evaluator.px.Client",
            return_value=mock_phoenix_client,
        ):
            evaluator = PhoenixSpanEvaluator(optimizer=mock_optimizer)

        assert evaluator.optimizer == mock_optimizer
        assert evaluator.tenant_id == "default"
        assert evaluator.phoenix_client == mock_phoenix_client
        assert evaluator.project_name.startswith("cogniverse-default-")

    def test_initialization_custom_tenant(self, mock_optimizer, mock_phoenix_client):
        """Test initialization with custom tenant"""
        with patch(
            "cogniverse_agents.routing.phoenix_span_evaluator.px.Client",
            return_value=mock_phoenix_client,
        ):
            evaluator = PhoenixSpanEvaluator(
                optimizer=mock_optimizer, tenant_id="custom-tenant"
            )

        assert evaluator.tenant_id == "custom-tenant"
        assert "custom-tenant" in evaluator.project_name


class TestExtractRoutingExperience:
    """Test routing experience extraction from span data"""

    def test_extract_experience_phoenix_flattened_format(self, span_evaluator):
        """Test extraction from Phoenix flattened attribute format"""
        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-1",
                "start_time": datetime.now(),
                "status": "OK",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.95,
                    "processing_time": 123.45,
                    "query": "show me basketball videos",
                    "context": None,
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is not None
        assert experience.query == "show me basketball videos"
        assert experience.chosen_agent == "video_search_agent"
        assert experience.routing_confidence == 0.95
        assert experience.processing_time == 123.45

    def test_extract_experience_nested_format(self, span_evaluator):
        """Test extraction from nested attribute format (unit test format)"""
        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-2",
                "start_time": datetime.now(),
                "status": "OK",
                "attributes": {
                    "routing.chosen_agent": "summarizer_agent",
                    "routing.confidence": 0.87,
                    "routing.processing_time": 200.0,
                    "routing.query": "summarize the game",
                    "routing.context": None,
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is not None
        assert experience.query == "summarize the game"
        assert experience.chosen_agent == "summarizer_agent"
        assert experience.routing_confidence == 0.87
        assert experience.processing_time == 200.0

    def test_extract_experience_with_context(self, span_evaluator):
        """Test extraction with context containing entities and relationships"""
        import json

        context = {
            "entities": [{"text": "basketball", "type": "SPORT"}],
            "relationships": [
                {"source": "player", "relation": "plays", "target": "basketball"}
            ],
            "enhanced_query": "show me professional basketball highlights",
        }

        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-3",
                "start_time": datetime.now(),
                "status": "OK",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.92,
                    "processing_time": 150.0,
                    "query": "show me basketball",
                    "context": json.dumps(context),
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is not None
        assert len(experience.entities) == 1
        assert experience.entities[0]["text"] == "basketball"
        assert len(experience.relationships) == 1
        assert experience.enhanced_query == "show me professional basketball highlights"

    def test_extract_experience_wrong_span_name(self, span_evaluator):
        """Test extraction rejects non-routing spans"""
        span_row = pd.Series(
            {
                "name": "cogniverse.request",  # Wrong span name
                "span_id": "test-span-4",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.95,
                    "query": "show me videos",
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is None

    def test_extract_experience_missing_required_fields(self, span_evaluator):
        """Test extraction fails gracefully with missing required fields"""
        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-5",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    # Missing confidence and query
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is None

    def test_extract_experience_successful_span(self, span_evaluator):
        """Test agent_success is True for OK status"""
        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-6",
                "status": "OK",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.95,
                    "query": "show me videos",
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is not None
        assert experience.agent_success is True

    def test_extract_experience_failed_span(self, span_evaluator):
        """Test agent_success is False for ERROR status"""
        span_row = pd.Series(
            {
                "name": "cogniverse.routing",
                "span_id": "test-span-7",
                "status": "ERROR",
                "attributes.routing": {
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.95,
                    "query": "show me videos",
                },
            }
        )

        experience = span_evaluator._extract_routing_experience_from_df_row(span_row)

        assert experience is not None
        assert experience.agent_success is False


class TestEvaluateRoutingSpans:
    """Test span evaluation and experience creation"""

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_success(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test successful evaluation of routing spans"""
        # Mock Phoenix client to return routing spans
        routing_spans = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "start_time": datetime.now(),
                    "status": "OK",
                    "attributes.routing": {
                        "chosen_agent": "video_search_agent",
                        "confidence": 0.95,
                        "query": "show me basketball",
                    },
                },
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-2",
                    "start_time": datetime.now(),
                    "status": "OK",
                    "attributes.routing": {
                        "chosen_agent": "summarizer_agent",
                        "confidence": 0.88,
                        "query": "summarize the game",
                    },
                },
            ]
        )

        mock_phoenix_client.get_spans_dataframe = Mock(return_value=routing_spans)

        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] == 2
        assert results["experiences_created"] == 2
        assert span_evaluator.optimizer.record_routing_experience.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_no_spans(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test evaluation when no spans are found"""
        mock_phoenix_client.get_spans_dataframe = Mock(return_value=pd.DataFrame())

        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] == 0
        assert results["experiences_created"] == 0
        assert span_evaluator.optimizer.record_routing_experience.call_count == 0

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_no_routing_spans(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test evaluation when spans exist but none are routing spans"""
        non_routing_spans = pd.DataFrame(
            [
                {
                    "name": "cogniverse.request",  # Parent span, not routing
                    "span_id": "span-1",
                    "start_time": datetime.now(),
                },
                {
                    "name": "search",  # Search span, not routing
                    "span_id": "span-2",
                    "start_time": datetime.now(),
                },
            ]
        )

        mock_phoenix_client.get_spans_dataframe = Mock(return_value=non_routing_spans)

        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] == 0
        assert results["experiences_created"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_filters_duplicates(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test that duplicate span IDs are not processed twice"""
        routing_spans = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "start_time": datetime.now(),
                    "status": "OK",
                    "attributes.routing": {
                        "chosen_agent": "video_search_agent",
                        "confidence": 0.95,
                        "query": "show me basketball",
                    },
                }
            ]
        )

        mock_phoenix_client.get_spans_dataframe = Mock(return_value=routing_spans)

        # Evaluate twice
        results1 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)
        results2 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results1["experiences_created"] == 1
        assert results2["experiences_created"] == 0  # Duplicate filtered out

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_handles_errors(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test error handling during span evaluation"""
        mock_phoenix_client.get_spans_dataframe = Mock(
            side_effect=Exception("Phoenix connection error")
        )

        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] == 0
        assert results["experiences_created"] == 0
        assert len(results["errors"]) == 1
        assert "Phoenix connection error" in results["errors"][0]

    @pytest.mark.asyncio
    async def test_evaluate_routing_spans_batch_size(
        self, span_evaluator, mock_phoenix_client
    ):
        """Test batch size limiting"""
        # Create 100 routing spans
        routing_spans = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": f"span-{i}",
                    "start_time": datetime.now() - timedelta(seconds=i),
                    "status": "OK",
                    "attributes.routing": {
                        "chosen_agent": "video_search_agent",
                        "confidence": 0.95,
                        "query": f"query {i}",
                    },
                }
                for i in range(100)
            ]
        )

        mock_phoenix_client.get_spans_dataframe = Mock(return_value=routing_spans)

        results = await span_evaluator.evaluate_routing_spans(
            lookback_hours=1, batch_size=10
        )

        # Should only process 10 spans due to batch size limit
        assert results["spans_processed"] == 10
        assert results["experiences_created"] == 10


class TestHelperMethods:
    """Test helper methods"""

    def test_is_routing_span_df(self, span_evaluator):
        """Test routing span identification"""
        routing_span = pd.Series({"name": "cogniverse.routing"})
        non_routing_span = pd.Series({"name": "cogniverse.request"})

        assert span_evaluator._is_routing_span_df(routing_span) is True
        assert span_evaluator._is_routing_span_df(non_routing_span) is False

    def test_compute_search_quality_ok_status(self, span_evaluator):
        """Test search quality computation with OK status"""
        span_row = pd.Series({"status": "OK", "attributes": {"agent.result_count": 10}})

        quality = span_evaluator._compute_search_quality_df(span_row)

        assert quality > 0.5  # OK status should boost quality

    def test_compute_search_quality_error_status(self, span_evaluator):
        """Test search quality computation with ERROR status"""
        span_row = pd.Series({"status": "ERROR"})

        quality = span_evaluator._compute_search_quality_df(span_row)

        assert quality == 0.1  # Error should result in low quality

    def test_determine_agent_success(self, span_evaluator):
        """Test agent success determination"""
        success_span = pd.Series({"status": "OK"})
        error_span = pd.Series({"status": "ERROR"})

        assert span_evaluator._determine_agent_success_df(success_span) is True
        assert span_evaluator._determine_agent_success_df(error_span) is False
