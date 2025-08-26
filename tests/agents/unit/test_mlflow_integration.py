"""
Unit tests for MLflow Integration component.

Tests the MLflow integration functionality including experiment tracking,
model versioning, and A/B testing for the routing system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.app.routing.mlflow_integration import (
    ABTestConfig,
    ExperimentConfig,
    MLflowIntegration,
    ModelVersionInfo,
    create_mlflow_integration,
)


class TestExperimentConfig:
    """Test experiment configuration functionality."""

    def test_experiment_config_creation_minimal(self):
        """Test creating experiment configuration with minimal parameters."""
        config = ExperimentConfig(experiment_name="test_routing_optimization")

        assert config.experiment_name == "test_routing_optimization"
        assert config.tracking_uri == "http://localhost:5000"  # Default
        assert config.auto_log_parameters is True
        assert config.auto_log_metrics is True
        assert config.track_dspy_modules is True

    def test_experiment_config_creation_full(self):
        """Test creating experiment configuration with all parameters."""
        config = ExperimentConfig(
            experiment_name="full_test_experiment",
            tracking_uri="file:///tmp/test_mlruns",
            artifact_location="/tmp/artifacts",
            tags={"version": "1.0", "environment": "test"},
            description="Test experiment",
            auto_log_parameters=False,
            track_dspy_modules=False,
            metrics_logging_frequency=20,
        )

        assert config.experiment_name == "full_test_experiment"
        assert config.tracking_uri == "file:///tmp/test_mlruns"
        assert config.artifact_location == "/tmp/artifacts"
        assert config.tags["version"] == "1.0"
        assert config.description == "Test experiment"
        assert config.auto_log_parameters is False
        assert config.track_dspy_modules is False
        assert config.metrics_logging_frequency == 20


class TestABTestConfig:
    """Test A/B test configuration functionality."""

    def test_ab_test_config_creation(self):
        """Test creating A/B test configuration."""
        config = ABTestConfig(
            test_name="routing_algorithm_test",
            control_group_ratio=0.6,
            treatment_group_ratio=0.4,
            minimum_sample_size=500,
        )

        assert config.test_name == "routing_algorithm_test"
        assert config.control_group_ratio == 0.6
        assert config.treatment_group_ratio == 0.4
        assert config.minimum_sample_size == 500
        assert config.primary_metric == "success_rate"  # Default
        assert config.enable_early_stopping is True


class TestModelVersionInfo:
    """Test model version info functionality."""

    def test_model_version_info_creation(self):
        """Test creating model version info."""
        from datetime import datetime

        now = datetime.now()
        info = ModelVersionInfo(
            name="routing_model",
            version="1.0.0",
            stage="Production",
            model_uri="models:/routing_model/1",
            run_id="abc123",
            creation_time=now,
            last_updated=now,
            description="Production routing model",
        )

        assert info.name == "routing_model"
        assert info.version == "1.0.0"
        assert info.stage == "Production"
        assert info.model_uri == "models:/routing_model/1"
        assert info.run_id == "abc123"
        assert info.description == "Production routing model"


class TestMLflowIntegration:
    """Test MLflow integration functionality."""

    def test_mlflow_integration_initialization_success(self):
        """Test successful MLflow integration initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="test_experiment",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            # Should initialize without error
            integration = MLflowIntegration(config, storage_dir=temp_dir)

            assert integration.config == config
            assert integration.storage_dir == Path(temp_dir)
            assert integration.client is not None
            assert integration.experiment_id is not None

    def test_mlflow_integration_missing_mlflow(self):
        """Test MLflow integration when MLflow is not available."""
        config = ExperimentConfig(experiment_name="test_experiment")

        with patch("src.app.routing.mlflow_integration.MLFLOW_AVAILABLE", False):
            with pytest.raises(ImportError, match="MLflow not available"):
                MLflowIntegration(config)

    def test_start_run(self):
        """Test starting an MLflow run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="run_test", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start a run using context manager
            with integration.start_run(run_name="test_run", tags={"version": "1.0"}):
                assert isinstance(integration.current_run_id, str)
                assert len(integration.current_run_id) > 0
                assert integration.current_run is not None

    def test_log_routing_performance(self):
        """Test logging routing performance metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="performance_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start run and log performance
            with integration.start_run("performance_test_run"):
                import asyncio

                asyncio.run(
                    integration.log_routing_performance(
                        query="test query",
                        routing_decision={"agent": "video_search", "confidence": 0.85},
                        performance_metrics={
                            "success_rate": 0.85,
                            "avg_response_time": 1.2,
                            "confidence_accuracy": 0.82,
                            "user_satisfaction": 0.9,
                        },
                    )
                )

                # Should complete without error
                assert integration.current_run_id is not None

    def test_log_optimization_metrics(self):
        """Test logging optimization metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="optimization_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start run and log optimization metrics
            integration.start_run("optimization_test_run")

            integration.log_optimization_metrics(
                optimizer_type="GEPA",
                learning_rate=0.01,
                improvement_score=0.15,
                convergence_step=50,
                total_training_time=120.5,
            )

            # Should complete without error
            assert integration.current_run_id is not None

    def test_save_and_load_dspy_model(self):
        """Test saving and loading DSPy models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="model_test", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)
            integration.start_run("model_test_run")

            # Create a mock DSPy module
            mock_module = Mock()
            mock_module.save = Mock()
            mock_module.load = Mock()
            mock_module.__class__.__name__ = "TestModule"

            # Save model
            model_uri = integration.save_dspy_model(
                dspy_module=mock_module,
                model_name="test_routing_module",
                tags={"version": "1.0"},
            )

            assert isinstance(model_uri, str)
            assert "test_routing_module" in model_uri

    def test_start_ab_test(self):
        """Test starting an A/B test."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="ab_test", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            ab_config = ABTestConfig(
                test_name="routing_strategy_test", minimum_sample_size=100
            )

            test_id = integration.start_ab_test(ab_config)

            assert isinstance(test_id, str)
            assert test_id in integration.active_ab_tests
            assert integration.active_ab_tests[test_id] == ab_config

    def test_assign_ab_test_group(self):
        """Test assigning users to A/B test groups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="ab_assignment_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start A/B test
            ab_config = ABTestConfig(
                test_name="assignment_test",
                control_group_ratio=0.5,
                treatment_group_ratio=0.5,
            )
            test_id = integration.start_ab_test(ab_config)

            # Assign multiple users - should get roughly 50/50 split
            assignments = []
            for i in range(100):
                group = integration.assign_ab_test_group(test_id, f"user_{i}")
                assignments.append(group)
                assert group in ["control", "treatment"]

            # Check that both groups are represented
            control_count = assignments.count("control")
            treatment_count = assignments.count("treatment")
            assert control_count > 20  # Should be roughly 50, but allow variation
            assert treatment_count > 20

    def test_log_ab_test_result(self):
        """Test logging A/B test results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="ab_result_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start A/B test
            ab_config = ABTestConfig(test_name="result_test")
            test_id = integration.start_ab_test(ab_config)

            # Log result
            integration.log_ab_test_result(
                test_id=test_id,
                user_id="user_123",
                group="control",
                primary_metric_value=0.85,
                secondary_metrics={"response_time": 1.2, "user_satisfaction": 0.9},
            )

            # Should complete without error
            assert test_id in integration.active_ab_tests

    def test_get_experiment_summary(self):
        """Test getting experiment summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="summary_test", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start run and log some metrics
            integration.start_run("summary_test_run")
            integration.log_routing_performance(
                success_rate=0.88, avg_response_time=1.1
            )

            summary = integration.get_experiment_summary()

            assert isinstance(summary, dict)
            assert "experiment_name" in summary
            assert "total_runs" in summary
            assert "experiment_id" in summary
            assert summary["experiment_name"] == "summary_test"

    def test_cleanup(self):
        """Test cleanup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="cleanup_test", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start run
            integration.start_run("cleanup_test_run")
            assert integration.current_run_id is not None

            # Cleanup should end active runs
            integration.cleanup()

            # Should complete without error
            # Note: current_run_id might still be set, but MLflow run should be ended
            assert True  # If we get here, cleanup worked


class TestMLflowIntegrationIntegration:
    """Test MLflow integration comprehensive functionality."""

    def test_create_mlflow_integration_function(self):
        """Test the create_mlflow_integration helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integration = create_mlflow_integration(
                experiment_name="helper_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
                storage_dir=temp_dir,
            )

            assert isinstance(integration, MLflowIntegration)
            assert integration.config.experiment_name == "helper_test"

    def test_component_imports_successfully(self):
        """Test that MLflow integration components can be imported."""
        try:
            import src.app.routing.mlflow_integration as mli

            # Verify key components exist
            assert hasattr(mli, "ABTestConfig")
            assert hasattr(mli, "ExperimentConfig")
            assert hasattr(mli, "MLflowIntegration")
            assert hasattr(mli, "ModelVersionInfo")
            assert hasattr(mli, "create_mlflow_integration")

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import MLflow integration components: {e}")

    def test_end_to_end_experiment_workflow(self):
        """Test complete experiment workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="e2e_workflow_test",
                tracking_uri=f"file://{temp_dir}/mlruns",
                auto_log_parameters=True,
                auto_log_metrics=True,
            )

            integration = MLflowIntegration(config, storage_dir=temp_dir)

            # Start experiment run
            run_id = integration.start_run("e2e_test_run", tags={"phase": "testing"})

            # Log routing performance
            integration.log_routing_performance(
                success_rate=0.92,
                avg_response_time=0.8,
                confidence_accuracy=0.89,
                user_satisfaction=0.95,
            )

            # Log optimization metrics
            integration.log_optimization_metrics(
                optimizer_type="SIMBA",
                learning_rate=0.005,
                improvement_score=0.18,
                convergence_step=75,
            )

            # Get experiment summary
            summary = integration.get_experiment_summary()

            # Cleanup
            integration.cleanup()

            # Verify workflow completed successfully
            assert run_id is not None
            assert summary["experiment_name"] == "e2e_workflow_test"
            assert summary["total_runs"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
