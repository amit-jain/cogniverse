"""
Unit tests for Adaptive Threshold Learner component.

Tests the adaptive threshold learning functionality including threshold optimization,
performance monitoring, and automatic adaptation strategies.
"""

import tempfile

import pytest

from src.app.routing.adaptive_threshold_learner import (
    AdaptationStrategy,
    AdaptiveThresholdConfig,
    AdaptiveThresholdLearner,
    PerformanceMetrics,
    ThresholdConfig,
    ThresholdParameter,
    ThresholdState,
)


class TestThresholdParameter:
    """Test threshold parameter enumeration."""

    @pytest.mark.ci_fast
    def test_threshold_parameter_values(self):
        """Test that threshold parameter enum has expected values."""
        assert ThresholdParameter.ROUTING_CONFIDENCE.value == "routing_confidence"
        assert ThresholdParameter.SIMILARITY_THRESHOLD.value == "similarity_threshold"
        assert ThresholdParameter.PATTERN_CONFIDENCE.value == "pattern_confidence"
        assert ThresholdParameter.ENTITY_CONFIDENCE.value == "entity_confidence"
        assert (
            ThresholdParameter.RELATIONSHIP_CONFIDENCE.value
            == "relationship_confidence"
        )


class TestAdaptationStrategy:
    """Test adaptation strategy enumeration."""

    def test_adaptation_strategy_values(self):
        """Test that adaptation strategy enum has expected values."""
        assert AdaptationStrategy.GRADIENT_BASED.value == "gradient_based"
        assert AdaptationStrategy.EVOLUTIONARY.value == "evolutionary"
        assert AdaptationStrategy.BANDIT.value == "bandit"
        assert AdaptationStrategy.BAYESIAN.value == "bayesian"
        assert AdaptationStrategy.STATISTICAL.value == "statistical"


class TestPerformanceMetrics:
    """Test performance metrics functionality."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            success_rate=0.85,
            average_confidence=0.82,
            response_time=1.2,
            user_satisfaction=0.9,
        )

        assert metrics.success_rate == 0.85
        assert metrics.average_confidence == 0.82
        assert metrics.response_time == 1.2
        assert metrics.user_satisfaction == 0.9
        assert metrics.sample_count == 0  # Default value


class TestAdaptiveThresholdConfig:
    """Test adaptive threshold configuration."""

    def test_config_creation_with_defaults(self):
        """Test creating configuration with default values."""
        config = AdaptiveThresholdConfig()

        assert config.global_learning_rate > 0
        assert config.performance_window_size > 0
        assert config.update_frequency > 0
        assert config.enable_automatic_rollback in [True, False]
        assert isinstance(config.threshold_configs, dict)

    def test_config_customization(self):
        """Test customizing configuration."""
        config = AdaptiveThresholdConfig(
            global_learning_rate=0.05,
            performance_window_size=200,
            update_frequency=25,
            enable_automatic_rollback=False,
        )

        assert config.global_learning_rate == 0.05
        assert config.performance_window_size == 200
        assert config.update_frequency == 25
        assert config.enable_automatic_rollback is False


class TestThresholdConfig:
    """Test individual threshold configuration."""

    def test_threshold_config_creation(self):
        """Test creating threshold configuration."""
        config = ThresholdConfig(
            parameter=ThresholdParameter.ROUTING_CONFIDENCE,
            initial_value=0.7,
            min_value=0.1,
            max_value=0.9,
            learning_rate=0.01,
        )

        assert config.parameter == ThresholdParameter.ROUTING_CONFIDENCE
        assert config.initial_value == 0.7
        assert config.min_value == 0.1
        assert config.max_value == 0.9
        assert config.learning_rate == 0.01
        assert (
            config.adaptation_strategy == AdaptationStrategy.GRADIENT_BASED
        )  # Default


class TestThresholdState:
    """Test threshold state functionality."""

    def test_threshold_state_creation(self):
        """Test creating threshold state."""
        state = ThresholdState(
            parameter=ThresholdParameter.ROUTING_CONFIDENCE,
            current_value=0.75,
            best_value=0.75,
            best_performance=0.0,
        )

        assert state.parameter == ThresholdParameter.ROUTING_CONFIDENCE
        assert state.current_value == 0.75
        assert state.best_value == 0.75
        assert state.best_performance == 0.0
        assert state.total_updates == 0
        assert state.rollbacks == 0


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner functionality."""

    def test_learner_initialization_with_defaults(self):
        """Test threshold learner initialization with defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            assert learner.config is not None
            assert isinstance(learner.config, AdaptiveThresholdConfig)
            assert hasattr(learner.config, "global_learning_rate")
            assert hasattr(learner.config, "performance_window_size")
            assert hasattr(learner, "threshold_states")
            assert len(learner.threshold_states) > 0

    def test_learner_with_custom_config(self):
        """Test learner with custom configuration."""
        config = AdaptiveThresholdConfig(global_learning_rate=0.02)

        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(config, storage_dir=temp_dir)

            assert learner.config.global_learning_rate == 0.02

    @pytest.mark.asyncio
    async def test_record_performance_sample(self):
        """Test recording performance samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            initial_sample_count = learner.sample_count

            # Record a performance sample
            await learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.75,
                response_time=1.2,
                user_satisfaction=0.85,
            )

            # Should have incremented sample count
            assert learner.sample_count == initial_sample_count + 1

    def test_get_threshold_value(self):
        """Test getting current threshold values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            threshold = learner.get_threshold_value(
                ThresholdParameter.ROUTING_CONFIDENCE
            )

            assert isinstance(threshold, float)
            assert 0 <= threshold <= 1

    def test_get_current_thresholds(self):
        """Test getting all current threshold values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            thresholds = learner.get_current_thresholds()

            assert isinstance(thresholds, dict)
            assert len(thresholds) > 0
            for param, value in thresholds.items():
                assert isinstance(param, ThresholdParameter)
                assert isinstance(value, float)

    def test_get_learning_status(self):
        """Test getting learning status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            status = learner.get_learning_status()

            assert isinstance(status, dict)
            assert "adaptive_learning_enabled" in status
            assert "threshold_status" in status
            assert "total_samples" in status
            assert "current_performance" in status

    @pytest.mark.asyncio
    async def test_learning_state_reset(self):
        """Test resetting learning state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            # Record some samples first
            await learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.75,
                response_time=1.2,
                user_satisfaction=0.85,
            )

            initial_sample_count = learner.sample_count
            assert initial_sample_count > 0

            # Reset learning state
            await learner.reset_learning_state()

            # Should have reset the sample count
            assert learner.sample_count == 0

    @pytest.mark.asyncio
    async def test_adaptive_learning_cycle(self):
        """Test complete adaptive learning cycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            # Simulate learning cycle with multiple samples
            for i in range(100):
                await learner.record_performance_sample(
                    routing_success=i % 4 != 0,  # 75% success rate
                    routing_confidence=0.7,
                    search_quality=0.6 + (i % 10) * 0.02,
                    response_time=0.8 + (i % 5) * 0.02,
                    user_satisfaction=0.8 + (i % 5) * 0.02,
                )

            # Get learning status
            status = learner.get_learning_status()

            assert status["total_samples"] == 100
            # Should be approximately 75% success rate (small variation due to rounding)
            assert 0.74 <= status["current_performance"]["success_rate"] <= 0.79

            # Should have meaningful performance data
            assert status["current_performance"]["average_confidence"] > 0
            assert status["current_performance"]["search_quality"] > 0

    def test_threshold_states_access(self):
        """Test accessing threshold states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            # Access threshold states directly
            assert isinstance(learner.threshold_states, dict)
            assert len(learner.threshold_states) > 0

            # Each state should be a ThresholdState
            for param, state in learner.threshold_states.items():
                assert isinstance(param, ThresholdParameter)
                assert isinstance(state, ThresholdState)
                assert state.current_value >= 0


class TestAdaptiveThresholdLearnerIntegration:
    """Test adaptive threshold learner integration functionality."""

    def test_component_imports_successfully(self):
        """Test that adaptive threshold learner components can be imported."""
        try:
            import src.app.routing.adaptive_threshold_learner as atl

            # Verify key components exist
            assert hasattr(atl, "AdaptationStrategy")
            assert hasattr(atl, "AdaptiveThresholdConfig")
            assert hasattr(atl, "AdaptiveThresholdLearner")
            assert hasattr(atl, "PerformanceMetrics")
            assert hasattr(atl, "ThresholdConfig")
            assert hasattr(atl, "ThresholdParameter")
            assert hasattr(atl, "ThresholdState")

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import adaptive threshold learner components: {e}")

    @pytest.mark.asyncio
    async def test_multi_parameter_learning(self):
        """Test learning across multiple threshold parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = AdaptiveThresholdLearner(storage_dir=temp_dir)

            # Record diverse performance samples
            samples = [
                {
                    "routing_success": True,
                    "routing_confidence": 0.9,
                    "search_quality": 0.95,
                },
                {
                    "routing_success": False,
                    "routing_confidence": 0.7,
                    "search_quality": 0.6,
                },
                {
                    "routing_success": True,
                    "routing_confidence": 0.8,
                    "search_quality": 0.85,
                },
                {
                    "routing_success": False,
                    "routing_confidence": 0.6,
                    "search_quality": 0.5,
                },
            ]

            for sample in samples:
                await learner.record_performance_sample(
                    routing_success=sample["routing_success"],
                    routing_confidence=sample["routing_confidence"],
                    search_quality=sample["search_quality"],
                    response_time=1.0,
                    user_satisfaction=0.8,
                )

            # Get status for multiple parameters
            status = learner.get_learning_status()
            threshold_status = status["threshold_status"]

            # Should have data for multiple threshold types
            assert "routing_confidence" in threshold_status
            assert "similarity_threshold" in threshold_status
            assert len(threshold_status) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
