"""
Unit tests for DSPy routing components.

Tests the routing-specific DSPy components including relationship extraction,
query enhancement, adaptive threshold learning, and advanced optimization.
"""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cogniverse_agents.routing.adaptive_threshold_learner import (
    AdaptiveThresholdLearner,
)
from cogniverse_agents.routing.advanced_optimizer import (
    AdvancedOptimizerConfig,
    AdvancedRoutingOptimizer,
)
from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
    create_composable_query_analysis_module,
)
from cogniverse_agents.routing.dspy_routing_signatures import (
    AdvancedRoutingSignature,
    BasicQueryAnalysisSignature,
)
from cogniverse_agents.routing.query_enhancement_engine import QueryEnhancementPipeline

# DSPy routing components
from cogniverse_agents.routing.relationship_extraction_tools import (
    RelationshipExtractorTool,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


def _make_mock_telemetry_provider():
    """Create a mock TelemetryProvider with in-memory stores."""
    provider = MagicMock()
    datasets: dict[str, pd.DataFrame] = {}

    async def create_dataset(name, data, metadata=None):
        datasets[name] = data
        return f"ds-{name}"

    async def get_dataset(name):
        if name not in datasets:
            raise KeyError(f"Dataset {name} not found")
        return datasets[name]

    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(side_effect=create_dataset)
    provider.datasets.get_dataset = AsyncMock(side_effect=get_dataset)
    provider.experiments = MagicMock()
    provider.experiments.create_experiment = AsyncMock(return_value="exp-test")
    provider.experiments.log_run = AsyncMock(return_value="run-test")
    return provider


class TestDSPyRoutingSignatures:
    """Test DSPy routing signature definitions"""

    def test_basic_query_analysis_signature(self):
        """Test basic query analysis signature"""
        signature = BasicQueryAnalysisSignature

        # Check that it's a proper DSPy signature
        assert hasattr(signature, "__annotations__")
        assert hasattr(signature, "__doc__")

        # Basic structure check
        assert signature is not None

    def test_advanced_routing_signature(self):
        """Test advanced routing signature"""
        signature = AdvancedRoutingSignature

        # Check that it's a proper DSPy signature
        assert hasattr(signature, "__annotations__")
        assert hasattr(signature, "__doc__")

        # Basic structure check
        assert signature is not None


class TestRelationshipExtractionTool:
    """Test relationship extraction tool"""

    def test_tool_initialization(self):
        """Test relationship extraction tool can be initialized"""
        tool = RelationshipExtractorTool()
        assert tool is not None
        # Tool should be usable even without spaCy models
        assert hasattr(tool, "extract_comprehensive_relationships")

    @pytest.mark.asyncio
    async def test_extract_relationships_basic(self):
        """Test basic relationship extraction"""
        tool = RelationshipExtractorTool()

        # Test with actual extraction (no mocking - it should handle missing models gracefully)
        result = await tool.extract_comprehensive_relationships(
            "A person performs an action"
        )
        assert isinstance(result, dict)
        assert "entities" in result
        assert "relationships" in result


class TestQueryEnhancementPipeline:
    """Test query enhancement pipeline"""

    def test_pipeline_initialization(self):
        """Test query enhancement pipeline initialization"""
        pipeline = QueryEnhancementPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "enhance_query_with_relationships")

    @pytest.mark.asyncio
    async def test_enhance_query_basic(self):
        """Test basic query enhancement"""
        pipeline = QueryEnhancementPipeline()

        # Test with actual enhancement (should handle gracefully)
        result = await pipeline.enhance_query_with_relationships("original query")
        assert isinstance(result, dict)
        assert "enhanced_query" in result


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner"""

    def test_learner_initialization(self):
        """Test adaptive threshold learner initialization"""
        learner = AdaptiveThresholdLearner(tenant_id="test-tenant")
        assert learner is not None
        assert hasattr(learner, "config")
        assert hasattr(learner, "threshold_states")

    def test_learning_status(self):
        """Test getting learning status"""
        learner = AdaptiveThresholdLearner(tenant_id="test-tenant")
        status = learner.get_learning_status()

        assert isinstance(status, dict)
        assert "threshold_status" in status

    @pytest.mark.asyncio
    async def test_record_performance_sample(self):
        """Test recording performance sample"""
        learner = AdaptiveThresholdLearner(tenant_id="test-tenant")

        # Test recording performance sample
        try:
            await learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.7,
                response_time=1.0,
                user_satisfaction=0.9,
            )
            # Should not raise an error
            assert True
        except Exception as e:
            pytest.fail(f"record_performance_sample failed: {e}")


class TestAdvancedRoutingOptimizer:
    """Test advanced routing optimizer"""

    def test_config_creation(self):
        """Test optimizer configuration"""
        config = AdvancedOptimizerConfig()
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.optimizer_strategy is not None

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        config = AdvancedOptimizerConfig(min_experiences_for_training=100)
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test-tenant",
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
            telemetry_provider=_make_mock_telemetry_provider(),
            config=config,
        )

        assert optimizer.config == config
        assert len(optimizer.experiences) == 0
        assert optimizer.training_step == 0

    @pytest.mark.asyncio
    async def test_record_experience(self):
        """Test recording routing experience"""
        config = AdvancedOptimizerConfig(min_experiences_for_training=100)
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test-tenant",
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
            telemetry_provider=_make_mock_telemetry_provider(),
            config=config,
        )

        reward = await optimizer.record_routing_experience(
            query="test query",
            entities=[],
            relationships=[],
            enhanced_query="enhanced test query",
            chosen_agent="test_agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True,
        )

        assert isinstance(reward, float)
        assert 0 <= reward <= 1
        assert len(optimizer.experiences) == 1

    def test_get_optimization_status(self):
        """Test getting optimization status"""
        config = AdvancedOptimizerConfig()
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test-tenant",
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
            telemetry_provider=_make_mock_telemetry_provider(),
            config=config,
        )

        status = optimizer.get_optimization_status()
        assert isinstance(status, dict)
        assert "optimizer_ready" in status
        assert "total_experiences" in status


class TestDSPyRouterModules:
    """Test DSPy router modules"""

    def test_composable_query_analysis_module(self):
        """Test ComposableQueryAnalysisModule creation and structure."""
        module = create_composable_query_analysis_module()
        assert module is not None
        assert isinstance(module, ComposableQueryAnalysisModule)
        assert hasattr(module, "forward")
        assert hasattr(module, "reformulator")
        assert hasattr(module, "unified_extractor")


class TestDSPyRoutingIntegration:
    """Test integration between DSPy routing components"""

    @pytest.mark.asyncio
    async def test_component_interaction(self, tmp_path):
        """Test basic interaction between components"""
        # Initialize components with temporary storage
        extractor = RelationshipExtractorTool()
        pipeline = QueryEnhancementPipeline()
        AdaptiveThresholdLearner(
            tenant_id="test-tenant"
        )  # Create but don't assign to unused variable
        optimizer = AdvancedRoutingOptimizer(
            tenant_id="test-tenant",
            llm_config=LLMEndpointConfig(model="ollama/test-model"),
            telemetry_provider=_make_mock_telemetry_provider(),
            config=AdvancedOptimizerConfig(min_experiences_for_training=10),
        )

        # Test real interactions (components handle missing models gracefully)
        # Extract relationships
        extract_result = await extractor.extract_comprehensive_relationships(
            "test query"
        )

        # Enhance query
        enhanced = await pipeline.enhance_query_with_relationships("test query")

        assert isinstance(extract_result, dict)
        assert isinstance(enhanced, dict)

        # Test optimizer recording
        await optimizer.record_routing_experience(
            query="test query",
            entities=[],
            relationships=[],
            enhanced_query=enhanced.get("enhanced_query", "test"),
            chosen_agent="test_agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True,
        )

        assert len(optimizer.experiences) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
