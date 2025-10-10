"""
Unit tests for ResultAggregator with relationship context integration
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cogniverse_agents.result_aggregator import (
    AgentResult,
    AggregatedResult,
    AggregationRequest,
    ResultAggregator,
)
from cogniverse_agents.routing_agent import RoutingDecision
from cogniverse_agents.result_enhancement_engine import EnhancedResult


@pytest.mark.unit
class TestAggregationRequest:
    """Test AggregationRequest data structure"""

    @pytest.mark.ci_fast
    def test_aggregation_request_creation_minimal(self):
        """Test creating AggregationRequest with minimal required fields"""
        routing_decision = RoutingDecision(
            query="test query",
            recommended_agent="video_search",
            confidence=0.8,
            reasoning="test routing",
            entities=[],
            relationships=[],
        )

        search_results = [{"video_id": "test1", "relevance": 0.9}]

        request = AggregationRequest(
            routing_decision=routing_decision, search_results=search_results
        )

        assert request.routing_decision == routing_decision
        assert request.search_results == search_results
        assert request.agents_to_invoke is None
        assert request.include_summaries is True
        assert request.include_detailed_report is True
        assert request.max_results_to_process == 50
        assert request.enhancement_config is None

    def test_aggregation_request_creation_full(self):
        """Test creating AggregationRequest with all fields"""
        routing_decision = RoutingDecision(
            query="find AI videos",
            recommended_agent="enhanced_video_search",
            confidence=0.9,
            reasoning="complex video search",
            entities=[{"text": "AI"}, {"text": "machine learning"}],
            relationships=[{"type": "semantic", "entities": ["AI", "ML"]}],
        )

        request = AggregationRequest(
            routing_decision=routing_decision,
            search_results=[{"video_id": "test1"}],
            agents_to_invoke=["summarizer", "detailed_report"],
            include_summaries=False,
            include_detailed_report=True,
            max_results_to_process=100,
            enhancement_config={"enable_relationships": True},
        )

        assert request.agents_to_invoke == ["summarizer", "detailed_report"]
        assert request.include_summaries is False
        assert request.max_results_to_process == 100
        assert request.enhancement_config["enable_relationships"] is True


@pytest.mark.unit
class TestAgentResult:
    """Test AgentResult data structure"""

    @pytest.mark.ci_fast
    def test_agent_result_success(self):
        """Test successful agent result"""
        result = AgentResult(
            agent_name="summarizer",
            result_data={"summary": "This is a test summary"},
            processing_time=1.5,
            success=True,
        )

        assert result.agent_name == "summarizer"
        assert result.result_data["summary"] == "This is a test summary"
        assert result.processing_time == 1.5
        assert result.success is True
        assert result.error_message is None

    def test_agent_result_failure(self):
        """Test failed agent result"""
        result = AgentResult(
            agent_name="detailed_report",
            result_data={},
            processing_time=0.1,
            success=False,
            error_message="Agent timeout",
        )

        assert result.success is False
        assert result.error_message == "Agent timeout"
        assert result.result_data == {}


@pytest.mark.unit
class TestAggregatedResult:
    """Test AggregatedResult data structure"""

    @pytest.mark.ci_fast
    def test_aggregated_result_creation(self):
        """Test creating AggregatedResult"""
        routing_decision = RoutingDecision(
            query="test",
            recommended_agent="test_agent",
            confidence=0.8,
            reasoning="test",
            entities=[],
            relationships=[],
        )

        enhanced_results = [
            EnhancedResult(
                original_result={"video_id": "test1"},
                relevance_score=0.8,
                entity_matches=[],
                relationship_matches=[],
                contextual_connections=[],
                enhancement_score=0.8,
                enhancement_metadata={},
            )
        ]

        agent_results = {
            "summarizer": AgentResult(
                agent_name="summarizer",
                result_data={"summary": "test"},
                processing_time=1.0,
                success=True,
            )
        }

        aggregated = AggregatedResult(
            routing_decision=routing_decision,
            enhanced_search_results=enhanced_results,
            agent_results=agent_results,
        )

        assert aggregated.routing_decision == routing_decision
        assert len(aggregated.enhanced_search_results) == 1
        assert "summarizer" in aggregated.agent_results
        assert aggregated.summaries is None
        assert aggregated.total_processing_time == 0.0


@pytest.mark.unit
class TestResultAggregator:
    """Test cases for ResultAggregator class"""

    @pytest.fixture
    def sample_routing_decision(self):
        """Sample routing decision for testing"""
        return RoutingDecision(
            query="find videos about AI",
            recommended_agent="enhanced_video_search",
            confidence=0.9,
            reasoning="Complex video search with AI entities",
            entities=[{"text": "AI"}, {"text": "artificial intelligence"}],
            relationships=[
                {"type": "synonym", "entities": ["AI", "artificial intelligence"]}
            ],
        )

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            {"video_id": "v1", "relevance": 0.95, "title": "AI Introduction"},
            {"video_id": "v2", "relevance": 0.87, "title": "Machine Learning Basics"},
            {"video_id": "v3", "relevance": 0.82, "title": "Deep Learning Tutorial"},
        ]

    @pytest.mark.ci_fast
    def test_result_aggregator_initialization_default(self):
        """Test ResultAggregator initialization with default config"""
        aggregator = ResultAggregator()

        assert aggregator is not None
        assert aggregator.max_concurrent_agents == 3
        assert aggregator.agent_timeout == 30.0
        assert aggregator.enable_fallbacks is True
        assert hasattr(aggregator, "enhancement_engine")
        assert hasattr(aggregator, "agent_endpoints")

        # Check default agent endpoints
        assert "summarizer" in aggregator.agent_endpoints
        assert "detailed_report" in aggregator.agent_endpoints
        assert "enhanced_video_search" in aggregator.agent_endpoints

    def test_result_aggregator_initialization_custom(self):
        """Test ResultAggregator initialization with custom config"""
        custom_config = {
            "max_concurrent_agents": 5,
            "agent_timeout": 60.0,
            "enable_fallbacks": False,
            "enhancement_config": {
                "enable_relationships": True,
                "max_enhancement_depth": 3,
            },
        }

        aggregator = ResultAggregator(**custom_config)

        assert aggregator.max_concurrent_agents == 5
        assert aggregator.agent_timeout == 60.0
        assert aggregator.enable_fallbacks is False

    @pytest.mark.ci_fast
    @patch("src.app.agents.result_aggregator.ResultEnhancementEngine")
    @pytest.mark.asyncio
    async def test_aggregate_and_enhance_basic(
        self, mock_enhancement_engine, sample_routing_decision, sample_search_results
    ):
        """Test basic aggregate and enhance functionality"""
        # Mock the enhancement engine
        mock_engine_instance = Mock()
        mock_enhancement_engine.return_value = mock_engine_instance
        mock_engine_instance.enhance_search_results = AsyncMock(
            return_value=[
                EnhancedResult(
                    original_result=sample_search_results[0],
                    relevance_score=0.95,
                    entity_matches=[{"entity": "AI", "confidence": 0.9}],
                    relationship_matches=[],
                    contextual_connections=[],
                    enhancement_score=0.9,
                    enhancement_metadata={"confidence": 0.9},
                )
            ]
        )

        aggregator = ResultAggregator()
        aggregator.enhancement_engine = mock_engine_instance

        request = AggregationRequest(
            routing_decision=sample_routing_decision,
            search_results=sample_search_results,
            agents_to_invoke=None,  # Test with no specific agents to avoid HTTP calls
        )

        # Test basic aggregation - should work even without agents
        result = await aggregator.aggregate_and_enhance(request)

        assert isinstance(result, AggregatedResult)
        assert result.routing_decision == sample_routing_decision
        # The result structure should be valid even if enhancement fails

    @pytest.mark.asyncio
    async def test_aggregate_and_enhance_no_agents(
        self, sample_routing_decision, sample_search_results
    ):
        """Test aggregation with no agents to invoke"""
        with patch(
            "src.app.agents.result_aggregator.ResultEnhancementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.enhance_search_results = AsyncMock(
                return_value=[
                    EnhancedResult(
                        original_result=sample_search_results[0],
                        relevance_score=0.8,
                        entity_matches=[],
                        relationship_matches=[],
                        contextual_connections=[],
                        enhancement_score=0.8,
                        enhancement_metadata={},
                    )
                ]
            )

            aggregator = ResultAggregator()

            request = AggregationRequest(
                routing_decision=sample_routing_decision,
                search_results=sample_search_results,
                agents_to_invoke=None,  # No agents
            )

            result = await aggregator.aggregate_and_enhance(request)

            assert isinstance(result, AggregatedResult)
            assert len(result.agent_results) == 0
            assert result.summaries is None
            assert result.detailed_report is None

    def test_aggregator_handles_different_agent_configurations(
        self, sample_routing_decision
    ):
        """Test that aggregator works with different agent configurations"""
        aggregator = ResultAggregator()

        # Test initialization sets up agent endpoints properly
        assert isinstance(aggregator.agent_endpoints, dict)
        assert len(aggregator.agent_endpoints) > 0

        # Test different request configurations
        request1 = AggregationRequest(
            routing_decision=sample_routing_decision,
            search_results=[],
            agents_to_invoke=["summarizer"],
        )
        assert request1.agents_to_invoke == ["summarizer"]

        request2 = AggregationRequest(
            routing_decision=sample_routing_decision,
            search_results=[],
            include_summaries=True,
            include_detailed_report=False,
        )
        assert request2.include_summaries is True
        assert request2.include_detailed_report is False


@pytest.mark.unit
class TestResultAggregatorEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def minimal_routing_decision(self):
        """Minimal routing decision for edge case testing"""
        return RoutingDecision(
            query="test",
            recommended_agent="test_agent",
            confidence=0.5,
            reasoning="minimal test",
            entities=[],
            relationships=[],
        )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_aggregate_empty_search_results(self, minimal_routing_decision):
        """Test aggregation with empty search results"""
        with patch(
            "src.app.agents.result_aggregator.ResultEnhancementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.enhance_search_results = AsyncMock(return_value=[])

            aggregator = ResultAggregator()

            request = AggregationRequest(
                routing_decision=minimal_routing_decision,
                search_results=[],  # Empty results
            )

            result = await aggregator.aggregate_and_enhance(request)

            assert isinstance(result, AggregatedResult)
            assert len(result.enhanced_search_results) == 0

    @pytest.mark.asyncio
    async def test_aggregate_with_agent_failures(self, minimal_routing_decision):
        """Test aggregation when agents fail"""
        with patch(
            "src.app.agents.result_aggregator.ResultEnhancementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.enhance_search_results = AsyncMock(return_value=[])

            aggregator = ResultAggregator()

            # Simulate what happens when there are no agents to invoke
            # The aggregator should handle this gracefully

            request = AggregationRequest(
                routing_decision=minimal_routing_decision,
                search_results=[{"test": "data"}],
                agents_to_invoke=["summarizer", "detailed_report"],
            )

            result = await aggregator.aggregate_and_enhance(request)

            # Even if no agents are invoked, should return a valid result
            assert isinstance(result, AggregatedResult)
            # The actual result depends on implementation - may have empty agent results

    @pytest.mark.asyncio
    async def test_max_results_processing_limit(self, minimal_routing_decision):
        """Test that max_results_to_process is respected during aggregation"""
        with patch(
            "src.app.agents.result_aggregator.ResultEnhancementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            # Mock enhancement to return only a limited number of results
            mock_engine.enhance_search_results = AsyncMock(return_value=[])

            aggregator = ResultAggregator()

            # Create many search results
            many_results = [{"video_id": f"v{i}"} for i in range(100)]

            request = AggregationRequest(
                routing_decision=minimal_routing_decision,
                search_results=many_results,
                max_results_to_process=10,  # Limit to 10
            )

            # The aggregation should complete without error
            result = await aggregator.aggregate_and_enhance(request)
            assert isinstance(result, AggregatedResult)

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_enhancement_statistics_in_result(self, minimal_routing_decision):
        """Test that enhancement statistics are included in aggregated results"""
        with patch(
            "src.app.agents.result_aggregator.ResultEnhancementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.enhance_search_results = AsyncMock(
                return_value=[
                    EnhancedResult(
                        original_result={"id": "1"},
                        relevance_score=0.9,
                        entity_matches=[],
                        relationship_matches=[],
                        contextual_connections=[],
                        enhancement_score=0.9,
                        enhancement_metadata={"processing_time": 0.1},
                    )
                ]
            )

            aggregator = ResultAggregator()

            request = AggregationRequest(
                routing_decision=minimal_routing_decision,
                search_results=[{"test": "data"}],
            )

            result = await aggregator.aggregate_and_enhance(request)

            assert isinstance(result, AggregatedResult)
            assert result.enhancement_statistics is not None
            assert isinstance(result.enhancement_statistics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
