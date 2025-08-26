"""
Unit tests for the Advanced Routing Optimizer and related components.

Tests the advanced routing optimization functionality including experience recording,
reward computation, and multi-stage DSPy optimization.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

# Import components to test
from src.app.routing.advanced_optimizer import (
    AdvancedOptimizerConfig,
    AdvancedRoutingOptimizer,
    RoutingExperience,
)


class TestAdvancedRoutingOptimizerCore:
    """Test core advanced routing optimization functionality."""

    def test_config_creation(self):
        """Test that configuration can be created with default values."""
        config = AdvancedOptimizerConfig()

        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.optimizer_strategy == "adaptive"
        assert config.bootstrap_threshold == 20
        assert config.simba_threshold == 50
        assert config.mipro_threshold == 100
        assert config.gepa_threshold == 200

    def test_config_customization(self):
        """Test that configuration can be customized."""
        config = AdvancedOptimizerConfig(
            learning_rate=0.01, optimizer_strategy="gepa", force_optimizer="simba"
        )

        assert config.learning_rate == 0.01
        assert config.optimizer_strategy == "gepa"
        assert config.force_optimizer == "simba"

    def test_routing_experience_creation(self):
        """Test creating routing experience objects."""
        experience = RoutingExperience(
            query="Find AI videos",
            entities=[{"type": "CONCEPT", "text": "AI"}],
            relationships=[],
            enhanced_query="Find artificial intelligence videos",
            chosen_agent="video_search_agent",
            routing_confidence=0.8,
            search_quality=0.75,
            agent_success=True,
            reward=0.7,
        )

        assert experience.query == "Find AI videos"
        assert experience.chosen_agent == "video_search_agent"
        assert experience.routing_confidence == 0.8
        assert experience.search_quality == 0.75
        assert experience.agent_success is True
        assert experience.reward == 0.7
        assert isinstance(experience.timestamp, datetime)

    def test_optimizer_initialization_basic(self):
        """Test basic optimizer initialization without complex dependencies."""
        config = AdvancedOptimizerConfig(min_experiences_for_training=5)

        # Advanced optimizer doesn't use SentenceTransformer directly, so no mocking needed
        optimizer = AdvancedRoutingOptimizer(config)

        assert optimizer.config == config
        assert len(optimizer.experiences) == 0
        assert len(optimizer.experience_replay) == 0
        assert optimizer.training_step == 0
        assert optimizer.current_epsilon == config.exploration_epsilon

    @pytest.mark.asyncio
    async def test_record_experience_basic(self):
        """Test basic experience recording functionality."""
        config = AdvancedOptimizerConfig(
            min_experiences_for_training=100
        )  # High threshold

        optimizer = AdvancedRoutingOptimizer(config)

        # Record a basic experience
        reward = await optimizer.record_routing_experience(
            query="Test query",
            entities=[],
            relationships=[],
            enhanced_query="Test enhanced query",
            chosen_agent="test_agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True,
        )

        assert isinstance(reward, float)
        assert 0 <= reward <= 1
        assert len(optimizer.experiences) == 1
        assert len(optimizer.experience_replay) == 1

        experience = optimizer.experiences[0]
        assert experience.query == "Test query"
        assert experience.chosen_agent == "test_agent"

    def test_reward_computation(self):
        """Test reward computation logic."""
        config = AdvancedOptimizerConfig()

        optimizer = AdvancedRoutingOptimizer(config)

        # Test successful case
        reward_success = optimizer._compute_reward(
            search_quality=0.9,
            agent_success=True,
            processing_time=1.0,
            user_satisfaction=0.8,
        )

        # Test failure case
        reward_failure = optimizer._compute_reward(
            search_quality=0.3,
            agent_success=False,
            processing_time=5.0,
            user_satisfaction=0.2,
        )

        assert isinstance(reward_success, float)
        assert isinstance(reward_failure, float)
        assert 0 <= reward_success <= 1
        assert 0 <= reward_failure <= 1
        assert reward_success > reward_failure

    @pytest.mark.asyncio
    async def test_get_baseline_recommendations(self):
        """Test baseline recommendation functionality."""
        config = AdvancedOptimizerConfig()

        optimizer = AdvancedRoutingOptimizer(config)

        # Test video-related query
        recommendations = optimizer._get_baseline_recommendations(
            query="Find video about machine learning", entities=[], relationships=[]
        )

        assert isinstance(recommendations, dict)
        assert "recommended_agent" in recommendations
        assert "confidence" in recommendations
        assert "reasoning" in recommendations
        assert "optimization_ready" in recommendations

        assert recommendations["recommended_agent"] == "video_search_agent"
        assert recommendations["optimization_ready"] is False

        # Test summary-related query
        recommendations = optimizer._get_baseline_recommendations(
            query="Give me a summary of the document", entities=[], relationships=[]
        )

        assert recommendations["recommended_agent"] == "summarizer_agent"

    def test_optimization_status(self):
        """Test getting optimization status."""
        config = AdvancedOptimizerConfig()

        optimizer = AdvancedRoutingOptimizer(config)

        status = optimizer.get_optimization_status()

        assert isinstance(status, dict)
        assert "optimizer_ready" in status
        assert "total_experiences" in status
        assert "training_step" in status
        assert "metrics" in status
        assert "config" in status

        assert status["optimizer_ready"] is False  # No experiences yet
        assert status["total_experiences"] == 0
        assert status["training_step"] == 0


class TestSIMBAIntegration:
    """Test SIMBA integration functionality."""

    def test_simba_config_creation(self):
        """Test SIMBA configuration creation."""
        from src.app.routing.simba_query_enhancer import SIMBAConfig

        config = SIMBAConfig()
        assert config.similarity_threshold >= 0
        assert config.max_memory_size > 0

    def test_simba_enhancer_creation(self):
        """Test SIMBA enhancer can be created."""
        from src.app.routing.simba_query_enhancer import SIMBAConfig, SIMBAQueryEnhancer

        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)
            assert enhancer.config == config
            assert len(enhancer.enhancement_patterns) == 0


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner functionality."""

    def test_threshold_learner_creation(self):
        """Test threshold learner can be created."""
        from src.app.routing.adaptive_threshold_learner import AdaptiveThresholdLearner

        learner = AdaptiveThresholdLearner()

        assert learner.config is not None
        assert hasattr(learner.config, "global_learning_rate")
        assert hasattr(learner.config, "performance_window_size")
        assert hasattr(learner, "threshold_states")

    def test_threshold_learner_status(self):
        """Test getting threshold learner status."""
        from src.app.routing.adaptive_threshold_learner import AdaptiveThresholdLearner

        learner = AdaptiveThresholdLearner()
        status = learner.get_learning_status()

        assert isinstance(status, dict)
        assert "threshold_status" in status  # This is what the actual API returns


class TestMLflowIntegration:
    """Test MLflow integration functionality."""

    def test_mlflow_integration_creation(self):
        """Test MLflow integration can be created."""
        from src.app.routing.mlflow_integration import (
            ExperimentConfig,
            MLflowIntegration,
        )

        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.create_experiment"),
        ):

            config = ExperimentConfig(
                experiment_name="test_experiment", tracking_uri="file:///tmp/test"
            )

            integration = MLflowIntegration(config)

            # Just verify it was created without errors
            assert integration is not None
            assert integration.config == config


class TestAdvancedRoutingOptimizerIntegration:
    """Test integration between advanced routing optimization components."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self):
        """Test a simulated end-to-end workflow."""
        # Create basic optimizer with low threshold for testing
        config = AdvancedOptimizerConfig(min_experiences_for_training=3)

        optimizer = AdvancedRoutingOptimizer(config)

        # Simulate multiple experiences
        queries = [
            "Find machine learning videos",
            "Search for deep learning tutorials",
            "Locate AI research papers",
            "Get summary of neural networks",
            "Find detailed analysis of transformers",
        ]

        total_reward = 0
        for i, query in enumerate(queries):
            reward = await optimizer.record_routing_experience(
                query=query,
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
                enhanced_query=f"Enhanced: {query}",
                chosen_agent="video_search_agent",
                routing_confidence=0.7 + i * 0.05,
                search_quality=0.6 + i * 0.05,
                agent_success=True,
            )
            total_reward += reward

        # Verify experiences were recorded
        assert len(optimizer.experiences) == len(queries)
        assert len(optimizer.experience_replay) == len(queries)

        # Verify metrics were updated
        assert optimizer.metrics.total_experiences == len(queries)
        assert optimizer.metrics.successful_routes == len(queries)
        assert optimizer.metrics.avg_reward > 0

        # Get final status
        status = optimizer.get_optimization_status()
        assert status["total_experiences"] == len(queries)
        # Optimizer readiness depends on advanced_optimizer being created
        # which happens after min_experiences_for_training threshold
        assert status["total_experiences"] >= config.min_experiences_for_training

    def test_components_import_successfully(self):
        """Test that all Phase 6 components can be imported."""
        try:
            from src.app.routing.adaptive_threshold_learner import (
                AdaptiveThresholdLearner,
            )
            from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
            from src.app.routing.mlflow_integration import MLflowIntegration
            from src.app.routing.simba_query_enhancer import SIMBAQueryEnhancer

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Phase 6 components: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
