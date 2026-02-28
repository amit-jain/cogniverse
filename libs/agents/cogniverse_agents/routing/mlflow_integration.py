"""
MLflow Integration for DSPy 3.0 Multi-Agent Routing System

This module provides comprehensive MLflow integration for experiment tracking,
model versioning, performance monitoring, and A/B testing of the enhanced
routing system with GRPO, SIMBA, and adaptive threshold learning.

Key Features:
- Experiment tracking for routing performance and optimization
- Model versioning for DSPy modules and learned parameters
- Performance metrics logging and visualization
- A/B testing infrastructure for system improvements
- Integration with GRPO, SIMBA, and adaptive threshold learning
- Automated model registry and deployment management
"""

import json
import logging
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

# MLflow imports
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    ViewType = None
    MLFLOW_AVAILABLE = False

# DSPy integration
import dspy

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment"""

    experiment_name: str
    tracking_uri: str = "http://localhost:5000"
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    # Run configuration
    auto_log_parameters: bool = True
    auto_log_metrics: bool = True
    auto_log_artifacts: bool = True
    log_system_metrics: bool = True

    # Model tracking
    track_dspy_modules: bool = True
    track_optimization_state: bool = True
    track_threshold_parameters: bool = True

    # Performance tracking
    metrics_logging_frequency: int = 10  # Log every N samples
    batch_metrics_size: int = 100
    enable_real_time_logging: bool = True


@dataclass
class ModelVersionInfo:
    """Information about a model version"""

    name: str
    version: str
    stage: str  # None, Staging, Production, Archived
    model_uri: str
    run_id: str
    creation_time: datetime
    last_updated: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""

    test_name: str
    control_group_ratio: float = 0.5
    treatment_group_ratio: float = 0.5
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    power: float = 0.8
    effect_size_threshold: float = 0.05
    max_duration_hours: int = 72

    # Metrics to track
    primary_metric: str = "success_rate"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["response_time", "user_satisfaction"]
    )

    # Early stopping
    enable_early_stopping: bool = True
    early_stop_check_frequency: int = 100  # Check every N samples
    futility_threshold: float = 0.1


class MLflowIntegration:
    """
    MLflow integration for comprehensive experiment tracking and model management

    This class provides full MLflow integration for the DSPy 3.0 multi-agent
    routing system, including experiment tracking, model versioning, performance
    monitoring, and A/B testing capabilities.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        telemetry_provider: TelemetryProvider,
        tenant_id: str,
        test_mode: bool = False,
    ):
        """Initialize MLflow integration

        Args:
            config: Experiment configuration
            telemetry_provider: Telemetry provider for artifact persistence.
            tenant_id: Tenant identifier for multi-tenant isolation.
            test_mode: If True, use mock MLflow (for tests/CI)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        if not tenant_id:
            raise ValueError("tenant_id is required for MLflowIntegration")
        self._test_mode = test_mode

        self.config = config
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        # MLflow client
        self.client = None
        self.experiment_id = None
        self.current_run = None
        self.current_run_id = None

        # Model registry
        self.registered_models = {}
        self.active_model_versions = {}

        # A/B testing
        self.active_ab_tests = {}
        self.ab_test_assignments = {}

        # Metrics batching
        self.metrics_batch = []
        self.parameters_batch = {}
        self.tags_batch = {}

        # Performance tracking
        self.experiment_start_time = datetime.now()
        self.total_samples_logged = 0

        # Mock mode tracking
        self._is_mock = False

        # Initialize MLflow
        self._initialize_mlflow()

        logger.info(
            f"MLflow integration initialized for experiment: {config.experiment_name}"
        )

    def _initialize_mlflow(self):
        """Initialize MLflow tracking and experiment"""
        if self._test_mode:
            logger.warning("Test environment detected, using mock MLflow integration")
            self._initialize_mock_mlflow()
            return

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Initialize client with connection timeout
            self.client = MlflowClient(tracking_uri=self.config.tracking_uri)

            # Test connection with a quick operation
            try:
                # This will fail quickly if server is not available
                self.client.search_experiments(max_results=1)
            except Exception as conn_error:
                logger.warning(f"MLflow server not available: {conn_error}")
                self._initialize_mock_mlflow()
                return

            # Create or get experiment
            try:
                experiment = self.client.get_experiment_by_name(
                    self.config.experiment_name
                )
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    raise Exception("Experiment not found")
            except Exception:
                # Create new experiment
                self.experiment_id = self.client.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags,
                )

            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)

            # Configure auto-logging
            if self.config.auto_log_parameters:
                mlflow.autolog(log_models=False, exclusive=False)

            logger.info(
                f"MLflow experiment set up: {self.config.experiment_name} (ID: {self.experiment_id})"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow, falling back to mock: {e}")
            self._initialize_mock_mlflow()

    def _initialize_mock_mlflow(self):
        """Initialize mock MLflow integration for testing/development"""
        logger.info("Using mock MLflow integration")
        self.client = None
        self.experiment_id = "mock_experiment_id"
        self._is_mock = True

    @contextmanager
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Start MLflow run context manager"""
        if self._is_mock:
            # Mock run context
            class MockRun:
                def __init__(self):
                    self.info = type(
                        "info", (), {"run_id": f"mock_run_{uuid.uuid4().hex[:8]}"}
                    )()

            mock_run = MockRun()
            self.current_run = mock_run
            self.current_run_id = mock_run.info.run_id

            try:
                yield mock_run
            finally:
                self.current_run = None
                self.current_run_id = None
            return

        run_tags = {
            "system": "dspy_multi_agent_routing",
            "version": "1.0.0",
            "start_time": datetime.now().isoformat(),
            **self.config.tags,
            **(tags or {}),
        }

        with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
            self.current_run = run
            self.current_run_id = run.info.run_id

            # Log configuration
            if self.config.auto_log_parameters:
                self._log_system_configuration()

            try:
                yield run
            finally:
                # Flush any remaining batched metrics
                self._flush_metrics_batch()
                self.current_run = None
                self.current_run_id = None

    def _log_system_configuration(self):
        """Log system configuration parameters"""
        if self._is_mock:
            logger.debug("Mock mode: System configuration logging skipped")
            return

        try:
            # Log experiment config
            config_dict = asdict(self.config)
            for key, value in config_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"config.{key}", value)

            # Log system info
            mlflow.log_param(
                "python_version",
                f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            )
            mlflow.log_param("dspy_version", getattr(dspy, "__version__", "unknown"))

            logger.debug("System configuration logged to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log system configuration: {e}")

    async def log_routing_performance(
        self,
        query: str,
        routing_decision: Dict[str, Any],
        performance_metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Log routing performance metrics

        Args:
            query: User query
            routing_decision: Routing decision details
            performance_metrics: Performance metrics dict
            step: Optional step number
            timestamp: Optional timestamp
        """
        if not self.current_run:
            logger.warning("No active MLflow run for logging routing performance")
            return

        try:
            timestamp = timestamp or datetime.now()
            step = step or self.total_samples_logged

            # Log metrics
            for metric_name, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    if self.config.enable_real_time_logging:
                        mlflow.log_metric(f"routing.{metric_name}", value, step=step)
                    else:
                        self._batch_metric(f"routing.{metric_name}", value, step)

            # Log routing decision details
            if self.config.auto_log_parameters:
                mlflow.log_param(
                    f"routing.agent_{step}",
                    routing_decision.get("recommended_agent", "unknown"),
                )
                mlflow.log_metric(
                    "routing.confidence",
                    routing_decision.get("confidence", 0.0),
                    step=step,
                )

                if "enhanced_query" in routing_decision:
                    mlflow.log_param(
                        f"query.enhanced_{step}",
                        routing_decision["enhanced_query"] != query,
                    )

            # Log query analysis results
            if "entities" in routing_decision:
                mlflow.log_metric(
                    "analysis.entities_count",
                    len(routing_decision["entities"]),
                    step=step,
                )

            if "relationships" in routing_decision:
                mlflow.log_metric(
                    "analysis.relationships_count",
                    len(routing_decision["relationships"]),
                    step=step,
                )

            # Log GRPO/SIMBA metadata if available
            if routing_decision.get("grpo_applied"):
                mlflow.log_metric("optimization.grpo_applied", 1.0, step=step)
                if "training_step" in routing_decision:
                    mlflow.log_metric(
                        "optimization.grpo_training_step",
                        routing_decision["training_step"],
                        step=step,
                    )

            if routing_decision.get("simba_patterns_used", 0) > 0:
                mlflow.log_metric(
                    "enhancement.simba_patterns_used",
                    routing_decision["simba_patterns_used"],
                    step=step,
                )

            self.total_samples_logged += 1

            # Batch flush check
            if len(self.metrics_batch) >= self.config.batch_metrics_size:
                self._flush_metrics_batch()

        except Exception as e:
            logger.error(f"Failed to log routing performance: {e}")

    async def log_optimization_metrics(
        self,
        grpo_metrics: Optional[Dict[str, Any]] = None,
        simba_metrics: Optional[Dict[str, Any]] = None,
        threshold_metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ):
        """
        Log optimization system metrics

        Args:
            grpo_metrics: GRPO optimization metrics
            simba_metrics: SIMBA enhancement metrics
            threshold_metrics: Adaptive threshold metrics
            step: Optional step number
        """
        if not self.current_run:
            logger.warning("No active MLflow run for logging optimization metrics")
            return

        try:
            step = step or self.total_samples_logged

            # Log GRPO metrics
            if grpo_metrics:
                for key, value in grpo_metrics.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"grpo.{key}", value, step=step)

                # Log GRPO configuration
                grpo_config = grpo_metrics.get("config", {})
                for key, value in grpo_config.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"grpo.config.{key}", value)

            # Log SIMBA metrics
            if simba_metrics:
                for key, value in simba_metrics.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"simba.{key}", value, step=step)

                # Log SIMBA configuration
                simba_config = simba_metrics.get("config", {})
                for key, value in simba_config.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"simba.config.{key}", value)

            # Log threshold metrics
            if threshold_metrics:
                current_performance = threshold_metrics.get("current_performance", {})
                for key, value in current_performance.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(
                            f"thresholds.performance.{key}", value, step=step
                        )

                # Log individual threshold values
                threshold_status = threshold_metrics.get("threshold_status", {})
                for threshold_name, status in threshold_status.items():
                    if isinstance(status, dict):
                        current_value = status.get("current_value")
                        if current_value is not None:
                            mlflow.log_metric(
                                f"thresholds.{threshold_name}.value",
                                current_value,
                                step=step,
                            )

                        best_performance = status.get("best_performance")
                        if best_performance is not None:
                            mlflow.log_metric(
                                f"thresholds.{threshold_name}.best_performance",
                                best_performance,
                                step=step,
                            )

        except Exception as e:
            logger.error(f"Failed to log optimization metrics: {e}")

    def save_dspy_model(
        self,
        model: dspy.Module,
        model_name: str,
        version: Optional[str] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Save DSPy model to MLflow model registry

        Args:
            model: DSPy module to save
            model_name: Name for the model
            version: Optional version (auto-generated if None)
            description: Model description
            tags: Optional tags

        Returns:
            Model URI if successful, None otherwise
        """
        if not self.current_run:
            logger.warning("No active MLflow run for saving DSPy model")
            return None

        try:
            # Save DSPy module to temp file, then read JSON content
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "dspy_module.json"
                model.save(str(model_path))
                model_json = model_path.read_text()

            # Save to telemetry via ArtifactManager
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self._artifact_manager.save_blob("model", model_name, model_json)
                )

                # Create model info and log it
                model_info = {
                    "name": model_name,
                    "type": "dspy_module",
                    "class": model.__class__.__name__,
                    "version": version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "description": description,
                    "creation_time": datetime.now().isoformat(),
                    "run_id": self.current_run_id,
                    "tags": tags or {},
                }

                loop.run_until_complete(
                    self._artifact_manager.log_optimization_run(
                        "mlflow_model", model_info
                    )
                )
            finally:
                loop.close()

            # Register model
            try:
                model_uri = f"runs:/{self.current_run_id}/models/{model_name}"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    description=description,
                    tags=tags,
                )

                # Store model version info
                version_info = ModelVersionInfo(
                    name=model_name,
                    version=registered_model.version,
                    stage=registered_model.current_stage,
                    model_uri=model_uri,
                    run_id=self.current_run_id,
                    creation_time=datetime.now(),
                    last_updated=datetime.now(),
                    tags=tags or {},
                    description=description,
                )

                self.registered_models[model_name] = version_info

                logger.info(
                    f"DSPy model saved and registered: {model_name} v{registered_model.version}"
                )
                return model_uri

            except Exception as e:
                logger.warning(f"Failed to register model, but saved as artifact: {e}")
                return f"runs:/{self.current_run_id}/models/{model_name}"

        except Exception as e:
            logger.error(f"Failed to save DSPy model: {e}")
            return None

    def load_dspy_model(
        self, model_name: str, model_class: type = None
    ) -> Optional[dspy.Module]:
        """
        Load DSPy model from telemetry store.

        Args:
            model_name: Name of the model
            model_class: DSPy Module class to instantiate (required for .load())

        Returns:
            DSPy module if successful, None otherwise
        """
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                model_json = loop.run_until_complete(
                    self._artifact_manager.load_blob("model", model_name)
                )
            finally:
                loop.close()
            if not model_json:
                logger.warning(f"No model found in telemetry for {model_name}")
                return None

            # Write to temp file and load via DSPy
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "dspy_module.json"
                model_path.write_text(model_json)

                if model_class is None:
                    model_class = dspy.Module
                model = model_class()
                model.load(str(model_path))

            logger.info(f"DSPy model loaded from telemetry: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load DSPy model {model_name}: {e}")
            return None

    def start_ab_test(self, config: ABTestConfig) -> str:
        """
        Start A/B test experiment

        Args:
            config: A/B test configuration

        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())

        test_info = {
            "test_id": test_id,
            "config": config,
            "start_time": datetime.now(),
            "status": "running",
            "control_group": [],
            "treatment_group": [],
            "results": {
                "control": {
                    metric: []
                    for metric in [config.primary_metric] + config.secondary_metrics
                },
                "treatment": {
                    metric: []
                    for metric in [config.primary_metric] + config.secondary_metrics
                },
            },
        }

        self.active_ab_tests[test_id] = test_info

        # Log A/B test start
        if self.current_run:
            mlflow.log_param(
                f"ab_test.{config.test_name}.start", datetime.now().isoformat()
            )
            mlflow.log_param(
                f"ab_test.{config.test_name}.control_ratio", config.control_group_ratio
            )
            mlflow.log_param(
                f"ab_test.{config.test_name}.treatment_ratio",
                config.treatment_group_ratio,
            )

        logger.info(f"A/B test started: {config.test_name} (ID: {test_id})")
        return test_id

    def assign_ab_test_group(self, test_id: str, user_id: str) -> str:
        """
        Assign user to A/B test group

        Args:
            test_id: Test ID
            user_id: User ID

        Returns:
            Group assignment ("control" or "treatment")
        """
        if test_id not in self.active_ab_tests:
            return "control"  # Default fallback

        # Check if user already assigned
        if user_id in self.ab_test_assignments.get(test_id, {}):
            return self.ab_test_assignments[test_id][user_id]

        # Assign based on hash for consistency
        test_info = self.active_ab_tests[test_id]
        config = test_info["config"]

        # Use hash of user_id + test_id for deterministic assignment
        hash_value = hash(f"{user_id}_{test_id}") % 100

        if hash_value < config.control_group_ratio * 100:
            assignment = "control"
        else:
            assignment = "treatment"

        # Store assignment
        if test_id not in self.ab_test_assignments:
            self.ab_test_assignments[test_id] = {}
        self.ab_test_assignments[test_id][user_id] = assignment

        # Add to group
        test_info[f"{assignment}_group"].append(user_id)

        return assignment

    def log_ab_test_result(self, test_id: str, user_id: str, metrics: Dict[str, float]):
        """
        Log A/B test result

        Args:
            test_id: Test ID
            user_id: User ID
            metrics: Metrics dict
        """
        if test_id not in self.active_ab_tests:
            return

        # Get user's group assignment
        group = self.assign_ab_test_group(test_id, user_id)

        # Record metrics
        test_info = self.active_ab_tests[test_id]
        for metric_name, value in metrics.items():
            if metric_name in test_info["results"][group]:
                test_info["results"][group][metric_name].append(value)

        # Check for early stopping
        config = test_info["config"]
        if config.enable_early_stopping:
            self._check_ab_test_early_stopping(test_id)

    def _check_ab_test_early_stopping(self, test_id: str):
        """Check if A/B test should be stopped early"""
        test_info = self.active_ab_tests[test_id]
        config = test_info["config"]

        # Check sample size
        control_size = len(test_info["control_group"])
        treatment_size = len(test_info["treatment_group"])

        if (
            control_size < config.minimum_sample_size // 2
            or treatment_size < config.minimum_sample_size // 2
        ):
            return  # Not enough samples yet

        # Check statistical significance
        try:
            primary_metric = config.primary_metric
            control_values = test_info["results"]["control"][primary_metric]
            treatment_values = test_info["results"]["treatment"][primary_metric]

            if len(control_values) >= 30 and len(treatment_values) >= 30:
                from scipy import stats

                # Perform t-test
                statistic, p_value = stats.ttest_ind(treatment_values, control_values)

                # Check significance and effect size
                if p_value < config.significance_level:
                    effect_size = abs(
                        np.mean(treatment_values) - np.mean(control_values)
                    ) / np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)

                    if effect_size >= config.effect_size_threshold:
                        # Significant result - can stop early
                        self._stop_ab_test(
                            test_id,
                            reason="significant_result",
                            p_value=p_value,
                            effect_size=effect_size,
                        )
                        return

                # Check futility (no chance of significance)
                if (
                    p_value > config.futility_threshold
                    and control_size + treatment_size > config.minimum_sample_size
                ):
                    self._stop_ab_test(test_id, reason="futility", p_value=p_value)

        except Exception as e:
            logger.warning(f"Failed to check A/B test early stopping: {e}")

        # Check duration limit
        elapsed_time = datetime.now() - test_info["start_time"]
        if elapsed_time.total_seconds() > config.max_duration_hours * 3600:
            self._stop_ab_test(test_id, reason="duration_limit")

    def _stop_ab_test(self, test_id: str, reason: str, **kwargs):
        """Stop A/B test and log results"""
        if test_id not in self.active_ab_tests:
            return

        test_info = self.active_ab_tests[test_id]
        test_info["status"] = "stopped"
        test_info["stop_time"] = datetime.now()
        test_info["stop_reason"] = reason

        # Calculate final results
        config = test_info["config"]
        primary_metric = config.primary_metric

        control_values = test_info["results"]["control"][primary_metric]
        treatment_values = test_info["results"]["treatment"][primary_metric]

        final_results = {
            "control_mean": np.mean(control_values) if control_values else 0.0,
            "treatment_mean": np.mean(treatment_values) if treatment_values else 0.0,
            "control_samples": len(control_values),
            "treatment_samples": len(treatment_values),
            "stop_reason": reason,
            **kwargs,
        }

        # Log to MLflow
        if self.current_run:
            test_name = config.test_name
            for key, value in final_results.items():
                mlflow.log_metric(f"ab_test.{test_name}.{key}", value)

        logger.info(f"A/B test stopped: {config.test_name} ({reason})")

        # Move to completed tests
        if not hasattr(self, "completed_ab_tests"):
            self.completed_ab_tests = {}
        self.completed_ab_tests[test_id] = test_info

    def _batch_metric(self, key: str, value: float, step: int):
        """Add metric to batch for later logging"""
        self.metrics_batch.append(
            {"key": key, "value": value, "step": step, "timestamp": datetime.now()}
        )

    def _flush_metrics_batch(self):
        """Flush batched metrics to MLflow"""
        if not self.metrics_batch or not self.current_run:
            return

        if self._is_mock:
            logger.debug(
                f"Mock mode: Flushed {len(self.metrics_batch)} batched metrics (mock)"
            )
            self.metrics_batch.clear()
            return

        try:
            for metric in self.metrics_batch:
                mlflow.log_metric(metric["key"], metric["value"], step=metric["step"])

            logger.debug(f"Flushed {len(self.metrics_batch)} batched metrics")
            self.metrics_batch.clear()

        except Exception as e:
            logger.error(f"Failed to flush metrics batch: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary and statistics"""
        if self._is_mock:
            return {
                "experiment_name": self.config.experiment_name,
                "experiment_id": self.experiment_id,
                "total_runs": 0,
                "active_runs": 0,
                "completed_runs": 0,
                "recent_metrics": {},
                "ab_test_summary": {
                    "active_tests": 0,
                    "completed_tests": 0,
                    "total_test_users": 0,
                },
                "model_summary": {"registered_models": 0, "active_versions": 0},
                "is_mock": True,
            }

        try:
            # Get experiment info
            experiment = self.client.get_experiment(self.experiment_id)

            # Get runs for this experiment
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time desc"],
                max_results=100,
            )

            # Calculate statistics
            total_runs = len(runs)
            active_runs = len([r for r in runs if r.info.status == "RUNNING"])
            completed_runs = len([r for r in runs if r.info.status == "FINISHED"])

            # Get recent performance metrics
            recent_metrics = {}
            if runs:
                latest_run = runs[0]
                recent_metrics = {k: v for k, v in latest_run.data.metrics.items()}

            # A/B test summary
            ab_test_summary = {
                "active_tests": len(self.active_ab_tests),
                "completed_tests": len(getattr(self, "completed_ab_tests", {})),
                "total_test_users": sum(
                    len(test["control_group"]) + len(test["treatment_group"])
                    for test in self.active_ab_tests.values()
                ),
            }

            # Model registry summary
            model_summary = {
                "registered_models": len(self.registered_models),
                "active_versions": len(self.active_model_versions),
            }

            return {
                "experiment": {
                    "name": experiment.name,
                    "experiment_id": self.experiment_id,
                    "creation_time": datetime.fromtimestamp(
                        experiment.creation_time / 1000
                    ).isoformat(),
                    "artifact_location": experiment.artifact_location,
                },
                "runs": {
                    "total": total_runs,
                    "active": active_runs,
                    "completed": completed_runs,
                    "success_rate": completed_runs / max(1, total_runs),
                },
                "performance": {
                    "total_samples_logged": self.total_samples_logged,
                    "experiment_duration_hours": (
                        datetime.now() - self.experiment_start_time
                    ).total_seconds()
                    / 3600,
                    "recent_metrics": recent_metrics,
                },
                "ab_testing": ab_test_summary,
                "model_registry": model_summary,
                "config": {
                    "tracking_uri": self.config.tracking_uri,
                    "auto_log_enabled": self.config.auto_log_parameters,
                    "real_time_logging": self.config.enable_real_time_logging,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup MLflow integration"""
        try:
            # Flush remaining metrics
            self._flush_metrics_batch()

            # End current run if active
            if self.current_run:
                mlflow.end_run()

            # Stop active A/B tests
            for test_id in list(self.active_ab_tests.keys()):
                self._stop_ab_test(test_id, reason="cleanup")

            logger.info("MLflow integration cleanup complete")

        except Exception as e:
            logger.error(f"MLflow cleanup failed: {e}")


# Factory functions
def create_mlflow_integration(
    experiment_name: str,
    telemetry_provider: TelemetryProvider,
    tenant_id: str,
    tracking_uri: str = "http://localhost:5000",
    **kwargs,
) -> MLflowIntegration:
    """Create MLflow integration instance"""
    config = ExperimentConfig(
        experiment_name=experiment_name, tracking_uri=tracking_uri, **kwargs
    )
    return MLflowIntegration(
        config,
        telemetry_provider=telemetry_provider,
        tenant_id=tenant_id,
    )


def create_ab_test_config(
    test_name: str, primary_metric: str = "success_rate", **kwargs
) -> ABTestConfig:
    """Create A/B test configuration"""
    return ABTestConfig(test_name=test_name, primary_metric=primary_metric, **kwargs)
