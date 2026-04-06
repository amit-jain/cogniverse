"""
Unit tests for Adaptive Threshold Learner component.

Tests the adaptive threshold learning functionality including threshold optimization,
performance monitoring, and automatic adaptation strategies.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.routing.adaptive_threshold_learner import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdLearner,
    ThresholdParameter,
    ThresholdState,
)


def _make_mock_telemetry_provider():
    provider = MagicMock()
    datasets = {}

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


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner functionality."""

    def test_learner_initialization_with_defaults(self):
        """Test threshold learner initialization with defaults."""
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

        assert learner.config is not None
        assert isinstance(learner.config, AdaptiveThresholdConfig)
        assert hasattr(learner.config, "global_learning_rate")
        assert hasattr(learner.config, "performance_window_size")
        assert hasattr(learner, "threshold_states")
        assert len(learner.threshold_states) > 0

    def test_learner_with_custom_config(self):
        """Test learner with custom configuration."""
        config = AdaptiveThresholdConfig(global_learning_rate=0.02)

        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test_tenant",
            config=config,
        )

        assert learner.config.global_learning_rate == 0.02

    @pytest.mark.asyncio
    async def test_record_performance_sample(self):
        """Test recording performance samples."""
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

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
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

        threshold = learner.get_threshold_value(ThresholdParameter.ROUTING_CONFIDENCE)

        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1

    def test_get_current_thresholds(self):
        """Test getting all current threshold values."""
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

        thresholds = learner.get_current_thresholds()

        assert isinstance(thresholds, dict)
        assert len(thresholds) > 0
        for param, value in thresholds.items():
            assert isinstance(param, ThresholdParameter)
            assert isinstance(value, float)

    def test_get_learning_status(self):
        """Test getting learning status."""
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

        status = learner.get_learning_status()

        assert isinstance(status, dict)
        assert "adaptive_learning_enabled" in status
        assert "threshold_status" in status
        assert "total_samples" in status
        assert "current_performance" in status

    @pytest.mark.asyncio
    async def test_learning_state_reset(self):
        """Test resetting learning state."""
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

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
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

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
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

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
            import cogniverse_agents.routing.adaptive_threshold_learner as atl

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
        learner = AdaptiveThresholdLearner(
            telemetry_provider=_make_mock_telemetry_provider(), tenant_id="test_tenant"
        )

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
