"""
Core MLflow Integration Tests

Tests essential MLflow functionality without excessive mocking or implementation detail testing.
Focuses on the components we actually need working.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.app.routing.mlflow_integration import ExperimentConfig, MLflowIntegration


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
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="test_init", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            # Should not raise exception
            integration = MLflowIntegration(config, storage_dir=temp_dir)
            assert integration.config == config
            assert integration.storage_dir == Path(temp_dir)

    def test_context_manager_pattern(self):
        """Test that MLflow integration supports context manager pattern"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="test_context", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Test context manager exists and is callable
            assert hasattr(integration, "start_run")
            context_manager = integration.start_run(run_name="test_run")
            assert hasattr(context_manager, "__enter__")
            assert hasattr(context_manager, "__exit__")

    def test_essential_methods_exist(self):
        """Test that essential methods exist on MLflow integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="test_methods", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

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
        from src.app.routing.mlflow_integration import (
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

    @patch("src.app.routing.mlflow_integration.mlflow")
    def test_integration_with_dspy_system(self, mock_mlflow):
        """Test MLflow integration can work with DSPy routing system"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="dspy_integration_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            # Mock MLflow to avoid actual MLflow server dependency
            mock_mlflow.set_tracking_uri = lambda x: None
            mock_mlflow.create_experiment = (
                lambda x, artifact_location=None, tags=None: "test_exp_id"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Should be ready for use
            assert integration is not None
            assert integration.config.experiment_name == "dspy_integration_test"

    def test_factory_function_exists(self):
        """Test factory function for creating MLflow integration"""
        from src.app.routing.mlflow_integration import create_mlflow_integration

        assert callable(create_mlflow_integration)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
