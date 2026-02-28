"""
Core MLflow Integration Tests

Tests essential MLflow functionality without excessive mocking or implementation detail testing.
Focuses on the components we actually need working.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_agents.routing.mlflow_integration import (
    ExperimentConfig,
    MLflowIntegration,
)


def _make_mock_telemetry_provider():
    """Create a mock TelemetryProvider with in-memory stores."""
    provider = MagicMock()
    datasets: dict = {}

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


@pytest.mark.unit
class TestMLflowCoreIntegration:
    """Test core MLflow integration functionality"""

    def test_experiment_config_basic(self):
        """Test basic experiment configuration"""
        config = ExperimentConfig(experiment_name="test_experiment")
        assert config.experiment_name == "test_experiment"
        assert config.tracking_uri == "http://localhost:5000"  # default
        assert isinstance(config.tags, dict)

    def test_mlflow_integration_initialization(self):
        """Test MLflow integration can be initialized"""
        config = ExperimentConfig(experiment_name="test_init")

        # Should not raise exception
        integration = MLflowIntegration(
            config,
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test_tenant",
            test_mode=True,
        )
        assert integration.config == config

    def test_context_manager_pattern(self):
        """Test that MLflow integration supports context manager pattern"""
        config = ExperimentConfig(experiment_name="test_context")

        integration = MLflowIntegration(
            config,
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test_tenant",
            test_mode=True,
        )

        # Test context manager exists and is callable
        assert hasattr(integration, "start_run")
        context_manager = integration.start_run(run_name="test_run")
        assert hasattr(context_manager, "__enter__")
        assert hasattr(context_manager, "__exit__")

    def test_essential_methods_exist(self):
        """Test that essential methods exist on MLflow integration"""
        config = ExperimentConfig(experiment_name="test_methods")

        integration = MLflowIntegration(
            config,
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test_tenant",
            test_mode=True,
        )

        # Check essential methods exist
        assert hasattr(integration, "start_run")
        assert hasattr(integration, "log_routing_performance")
        assert hasattr(integration, "log_optimization_metrics")
        assert hasattr(integration, "save_dspy_model")
        assert callable(integration.start_run)
        assert callable(integration.log_routing_performance)
        assert callable(integration.log_optimization_metrics)
        assert callable(integration.save_dspy_model)

    def test_component_imports_successfully(self):
        """Test that all MLflow components can be imported"""
        # These imports should not raise exceptions
        from cogniverse_agents.routing.mlflow_integration import (
            ABTestConfig,
            ExperimentConfig,
            ModelVersionInfo,
        )

        # Basic instantiation should work
        config = ExperimentConfig(experiment_name="import_test")
        assert config is not None

        # Other configs should instantiate
        ab_config = ABTestConfig(test_name="test")
        assert ab_config.test_name == "test"

        version_info = ModelVersionInfo(
            name="test_model",
            version="1.0",
            stage="None",
            model_uri="test_uri",
            run_id="test_run",
            creation_time=None,
            last_updated=None,
        )
        assert version_info.name == "test_model"


@pytest.mark.unit
class TestMLflowIntegrationReadiness:
    """Test that MLflow integration is ready for use in the DSPy system"""

    @patch("cogniverse_agents.routing.mlflow_integration.mlflow")
    def test_integration_with_dspy_system(self, mock_mlflow):
        """Test MLflow integration can work with DSPy routing system"""
        config = ExperimentConfig(experiment_name="dspy_integration_test")

        # Mock MLflow to avoid actual MLflow server dependency
        mock_mlflow.set_tracking_uri = lambda x: None
        mock_mlflow.create_experiment = (
            lambda x, artifact_location=None, tags=None: "test_exp_id"
        )

        integration = MLflowIntegration(
            config,
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test_tenant",
            test_mode=True,
        )

        # Should be ready for use
        assert integration is not None
        assert integration.config.experiment_name == "dspy_integration_test"

    def test_factory_function_exists(self):
        """Test factory function for creating MLflow integration"""
        from cogniverse_agents.routing.mlflow_integration import (
            create_mlflow_integration,
        )

        assert callable(create_mlflow_integration)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
