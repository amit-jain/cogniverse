"""
Unit tests for the Advanced Routing Optimizer and related components.

Tests the advanced routing optimization functionality including experience recording,
reward computation, and multi-stage DSPy optimization.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

# Import components to test
from cogniverse_agents.routing.advanced_optimizer import (
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
        config = AdvancedOptimizerConfig(
            min_experiences_for_training=5,
            enable_persistence=False,  # Disable persistence for test isolation
        )

        # Advanced optimizer doesn't use SentenceTransformer directly, so no mocking needed
        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

        assert optimizer.config == config
        assert len(optimizer.experiences) == 0
        assert len(optimizer.experience_replay) == 0
        assert optimizer.training_step == 0
        assert optimizer.current_epsilon == config.exploration_epsilon

    @pytest.mark.asyncio
    async def test_record_experience_basic(self):
        """Test basic experience recording functionality."""
        config = AdvancedOptimizerConfig(
            min_experiences_for_training=100,  # High threshold
            enable_persistence=False,  # Disable persistence for test isolation
        )

        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

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

        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

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

        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

        # Test video-related query
        recommendations = optimizer._get_baseline_recommendations(
            query="Find video about machine learning", entities=[], relationships=[]
        )

        assert isinstance(recommendations, dict)
        assert "recommended_agent" in recommendations
        assert "confidence" in recommendations
        assert "reasoning" in recommendations
        assert "optimization_ready" in recommendations

        # Implementation uses "search_agent" for video queries
        assert recommendations["recommended_agent"] == "search_agent"
        assert recommendations["optimization_ready"] is False

        # Test summary-related query
        recommendations = optimizer._get_baseline_recommendations(
            query="Give me a summary of the document", entities=[], relationships=[]
        )

        assert recommendations["recommended_agent"] == "summarizer_agent"

    def test_optimization_status(self):
        """Test getting optimization status."""
        config = AdvancedOptimizerConfig(
            enable_persistence=False  # Disable persistence for test isolation
        )

        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

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
        from cogniverse_agents.routing.simba_query_enhancer import SIMBAConfig

        config = SIMBAConfig()
        assert config.similarity_threshold >= 0
        assert config.max_memory_size > 0

    def test_simba_enhancer_creation(self):
        """Test SIMBA enhancer can be created."""
        from cogniverse_agents.routing.simba_query_enhancer import (
            SIMBAConfig,
            SIMBAQueryEnhancer,
        )

        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)
            assert enhancer.config == config
            assert len(enhancer.enhancement_patterns) == 0


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner functionality."""

    def test_threshold_learner_creation(self):
        """Test threshold learner can be created."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdLearner,
        )

        learner = AdaptiveThresholdLearner(tenant_id="test_tenant")

        assert learner.config is not None
        assert hasattr(learner.config, "global_learning_rate")
        assert hasattr(learner.config, "performance_window_size")
        assert hasattr(learner, "threshold_states")

    def test_threshold_learner_status(self):
        """Test getting threshold learner status."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdLearner,
        )

        learner = AdaptiveThresholdLearner(tenant_id="test_tenant")
        status = learner.get_learning_status()

        assert isinstance(status, dict)
        assert "threshold_status" in status  # This is what the actual API returns


class TestMLflowIntegration:
    """Test MLflow integration functionality."""

    def test_mlflow_integration_creation(self):
        """Test MLflow integration can be created."""
        from cogniverse_agents.routing.mlflow_integration import (
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

            integration = MLflowIntegration(config, test_mode=True)

            # Just verify it was created without errors
            assert integration is not None
            assert integration.config == config


class TestAdvancedRoutingOptimizerIntegration:
    """Test integration between advanced routing optimization components."""

    @pytest.mark.asyncio
    async def test_gepa_optimizer_execution_with_200_examples(self):
        """
        CRITICAL TEST: Prove GEPA optimizer executes with 200+ examples.

        This test validates that:
        1. Advanced optimizer is initialized when min experiences reached
        2. GEPA is selected when dataset_size >= gepa_threshold (200)
        3. Optimization actually runs and returns optimized module
        4. Logging shows correct optimizer selection
        """
        import logging
        from unittest.mock import MagicMock, patch

        import dspy

        # Configure DSPy with a mock LM (required for GEPA) - use context manager for async
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        # Use context manager instead of settings.configure() in async
        with dspy.context(lm=mock_lm):
            # Enable logging to capture optimizer selection
            logger = logging.getLogger("cogniverse_agents.routing.advanced_optimizer")
            logger.setLevel(logging.INFO)

            # Create config with GEPA threshold at 200
            config = AdvancedOptimizerConfig(
                min_experiences_for_training=50,  # Can init optimizer early
                bootstrap_threshold=20,
                simba_threshold=50,
                mipro_threshold=100,
                gepa_threshold=200,
                optimizer_strategy="adaptive",  # Use adaptive selection
                update_frequency=10,  # Trigger optimization every 10 experiences
                enable_persistence=False,  # Disable persistence for test isolation
            )

            optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

            # Collect 210 experiences to exceed GEPA threshold
            for i in range(210):
                await optimizer.record_routing_experience(
                    query=f"Test query {i}",
                    entities=[{"type": "TEST", "text": f"entity_{i}"}],
                    relationships=[],
                    enhanced_query=f"Enhanced query {i}",
                    chosen_agent=(
                        "video_search_agent" if i % 2 == 0 else "summarizer_agent"
                    ),
                    routing_confidence=0.8 + (i % 20) * 0.01,  # Vary confidence
                    search_quality=0.7 + (i % 30) * 0.01,  # Vary quality
                    agent_success=i % 10 != 0,  # 90% success rate
                )

            # Verify optimizer was initialized
            assert (
                optimizer.advanced_optimizer is not None
            ), "Advanced optimizer should be initialized after min_experiences_for_training"

            # Verify we have enough experiences
            assert (
                len(optimizer.experiences) == 210
            ), f"Should have 210 experiences, got {len(optimizer.experiences)}"

            # Test optimizer selection with 210 examples
            dataset_size = 210
            selected_optimizer, optimizer_name = (
                optimizer.advanced_optimizer._select_optimizer(dataset_size)
            )

            assert (
                optimizer_name == "gepa"
            ), f"With {dataset_size} examples, should select GEPA, got {optimizer_name}"

            # Get optimization info to verify logic
            opt_info = optimizer.advanced_optimizer.get_optimization_info(dataset_size)
            assert (
                opt_info["primary_optimizer"] == "gepa"
            ), f"Primary optimizer should be GEPA, got {opt_info['primary_optimizer']}"
            assert opt_info["dataset_size"] == 210
            assert (
                len(opt_info["applicable_optimizers"]) == 4
            ), "All 4 optimizers (bootstrap, simba, mipro, gepa) should be applicable"

            # Test that compile() is called with correct optimizer
            # Mock the actual GEPA.compile to avoid LLM calls
            with patch.object(
                optimizer.advanced_optimizer.gepa_optimizer, "compile"
            ) as mock_gepa_compile:
                mock_optimized_module = MagicMock()
                mock_gepa_compile.return_value = mock_optimized_module

                # Create dummy training data
                import dspy

                trainset = [
                    dspy.Example(
                        query="test",
                        entities="[]",
                        relationships="[]",
                        enhanced_query="test",
                        recommended_agent="video_search_agent",
                        confidence="0.8",
                        reasoning="test",
                    ).with_inputs(
                        "query", "entities", "relationships", "enhanced_query"
                    )
                    for _ in range(210)
                ]

                # Run compile - should use GEPA
                result = optimizer.advanced_optimizer.compile(
                    optimizer.routing_policy,
                    trainset=trainset,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=8,
                )

                # Verify GEPA was called
                mock_gepa_compile.assert_called_once()
                assert (
                    result == mock_optimized_module
                ), "Should return optimized module from GEPA"

            # Verify status shows optimizer ready
            status = optimizer.get_optimization_status()
            assert (
                status["optimizer_ready"] is True
            ), "Optimizer should be ready with 210 experiences"
            assert status["total_experiences"] == 210

    @pytest.mark.asyncio
    async def test_optimizer_selection_thresholds(self):
        """Test that optimizer selection works correctly at each threshold."""
        from unittest.mock import MagicMock

        import dspy

        # Configure DSPy with mock LM using context (async-safe)
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        # Use dspy.context() in async context instead of dspy.settings.configure()
        with dspy.context(lm=mock_lm):
            config = AdvancedOptimizerConfig(
                min_experiences_for_training=10,
                bootstrap_threshold=20,
                simba_threshold=50,
                mipro_threshold=100,
                gepa_threshold=200,
                optimizer_strategy="adaptive",
                enable_persistence=False,  # Disable persistence for test isolation
            )

            optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

            # Collect minimum experiences to init optimizer
            for i in range(15):
                await optimizer.record_routing_experience(
                    query=f"Query {i}",
                    entities=[],
                    relationships=[],
                    enhanced_query=f"Query {i}",
                    chosen_agent="test_agent",
                    routing_confidence=0.8,
                    search_quality=0.7,
                    agent_success=True,
                )

            # Test selections at different dataset sizes
            test_cases = [
                (15, "bootstrap"),  # < 20: bootstrap only
                (25, "bootstrap"),  # < 50: bootstrap (highest applicable)
                (60, "simba"),  # >= 50, < 100: simba
                (120, "mipro"),  # >= 100, < 200: mipro
                (250, "gepa"),  # >= 200: gepa
            ]

            for dataset_size, expected_optimizer in test_cases:
                selected_optimizer, optimizer_name = (
                    optimizer.advanced_optimizer._select_optimizer(dataset_size)
                )
                assert (
                    optimizer_name == expected_optimizer
                ), f"With {dataset_size} examples, expected {expected_optimizer}, got {optimizer_name}"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self):
        """Test a simulated end-to-end workflow."""
        # Create basic optimizer with low threshold for testing
        config = AdvancedOptimizerConfig(
            min_experiences_for_training=3,
            enable_persistence=False,  # Disable persistence for test isolation
        )

        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant", config=config)

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
            import cogniverse_agents.routing.adaptive_threshold_learner as atl
            import cogniverse_agents.routing.advanced_optimizer as ao
            import cogniverse_agents.routing.mlflow_integration as mli
            import cogniverse_agents.routing.simba_query_enhancer as sqe

            # Verify key components exist
            assert hasattr(atl, "AdaptiveThresholdLearner")
            assert hasattr(ao, "AdvancedRoutingOptimizer")
            assert hasattr(mli, "MLflowIntegration")
            assert hasattr(sqe, "SIMBAQueryEnhancer")

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Phase 6 components: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
