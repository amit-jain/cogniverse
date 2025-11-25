"""
Phoenix Evaluation Provider

Implements the generic EvaluationProvider interface for Phoenix backend.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_evaluation.providers.base import EvaluationProvider
from phoenix.experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


class PhoenixEvaluationProvider(EvaluationProvider):
    """
    Phoenix implementation of the generic EvaluationProvider interface.

    Provides Phoenix-specific implementations for experiments, datasets,
    and evaluation result formatting.
    """

    def __init__(self):
        """Initialize Phoenix evaluation provider."""
        super().__init__()
        self.tenant_id: Optional[str] = None
        self.http_endpoint: str = "http://localhost:6006"
        self.phoenix_client: Optional[Any] = None
        self._telemetry_provider: Optional[Any] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: Configuration dictionary with:
                - tenant_id: Tenant identifier
                - http_endpoint: Phoenix server endpoint (default: http://localhost:6006)
                - grpc_endpoint: gRPC endpoint (default: http://localhost:4317)
                - project_name: Project name for telemetry (default: "evaluation")
                - Additional Phoenix-specific settings
        """
        self.tenant_id = config.get("tenant_id", "default")
        self.http_endpoint = config.get("http_endpoint", "http://localhost:6006")
        grpc_endpoint = config.get("grpc_endpoint", "http://localhost:4317")
        project_name = config.get("project_name", "evaluation")

        # Get telemetry provider for this tenant
        try:
            from cogniverse_foundation.telemetry.registry import TelemetryRegistry

            # Get telemetry provider from registry
            registry = TelemetryRegistry()
            self._telemetry_provider = registry.get_telemetry_provider(
                name="phoenix",
                tenant_id=self.tenant_id,
                config={
                    "project_name": project_name,
                    "http_endpoint": self.http_endpoint,
                    "grpc_endpoint": grpc_endpoint
                }
            )

            self._initialized = True
            logger.info(
                f"Initialized Phoenix evaluation provider for tenant: {self.tenant_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix evaluation provider: {e}")
            self._initialized = False

    @property
    def telemetry(self) -> Any:
        """
        Get telemetry provider for traces/datasets/experiments.

        Returns:
            Phoenix telemetry provider instance
        """
        return self._telemetry_provider

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new Phoenix experiment.

        Args:
            name: Experiment name
            description: Experiment description
            metadata: Additional metadata

        Returns:
            Phoenix experiment object
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        try:
            # Phoenix experiment creation logic
            logger.info(f"Creating Phoenix experiment: {name}")
            # Implementation depends on Phoenix SDK
            # For now, return a mock experiment object
            return {"id": name, "description": description, "metadata": metadata}
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise

    def create_dataset(
        self,
        name: str,
        data: List[Dict[str, Any]],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new Phoenix dataset.

        Args:
            name: Dataset name
            data: List of dataset examples
            description: Dataset description
            metadata: Additional metadata

        Returns:
            Phoenix dataset object
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        try:
            logger.info(f"Creating Phoenix dataset: {name} with {len(data)} examples")

            # Use Phoenix client to create dataset
            if self.phoenix_client:
                # Convert data to Phoenix format
                import pandas as pd

                df = pd.DataFrame(data)

                # Create dataset via Phoenix API
                # Note: This may need adjustment based on actual Phoenix SDK
                from phoenix.experiments import create_dataset

                dataset = create_dataset(
                    name=name,
                    data=df,
                    description=description or f"Dataset: {name}",
                )
                return dataset
            else:
                # Fallback if client not available
                return {"id": name, "data": data, "description": description}
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    def log_evaluation(
        self,
        experiment_id: str,
        evaluation_name: str,
        score: float,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an evaluation result to Phoenix.

        Args:
            experiment_id: Experiment identifier
            evaluation_name: Name of the evaluation
            score: Evaluation score
            label: Optional label
            explanation: Optional explanation
            metadata: Additional metadata
        """
        if not self._initialized:
            logger.warning("Provider not initialized, skipping evaluation logging")
            return

        try:
            logger.debug(
                f"Logging evaluation for experiment {experiment_id}: {evaluation_name} = {score}"
            )
            # Phoenix evaluation logging logic
            # Implementation depends on Phoenix SDK
        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")

    def create_evaluation_result(
        self,
        score: float,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a Phoenix EvaluationResult object.

        This is the key method that allows evaluators to return Phoenix-specific types
        while remaining generic in the evaluation package.

        Args:
            score: Evaluation score (typically 0-1)
            label: Optional categorical label
            explanation: Optional explanation text
            metadata: Additional metadata dict

        Returns:
            Phoenix EvaluationResult object
        """
        return EvaluationResult(
            score=score,
            label=label,
            explanation=explanation,
            metadata=metadata or {},
        )

    def get_experiment_url(self, experiment_id: str) -> str:
        """
        Get the URL for viewing an experiment in Phoenix UI.

        Args:
            experiment_id: Experiment identifier

        Returns:
            URL string for viewing the experiment
        """
        return f"{self.http_endpoint}/projects/{experiment_id}"

    def get_dataset_url(self, dataset_id: str) -> str:
        """
        Get the URL for viewing a dataset in Phoenix UI.

        Args:
            dataset_id: Dataset identifier

        Returns:
            URL string for viewing the dataset
        """
        return f"{self.http_endpoint}/datasets/{dataset_id}"

    def log_experiment_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log a generic experiment event to Phoenix.

        Args:
            event_type: Type of event (e.g., "experiment_start", "experiment_complete")
            data: Event data
        """
        if not self._initialized:
            logger.warning("Provider not initialized, skipping event logging")
            return

        try:
            logger.debug(f"Logging experiment event: {event_type}")
            # Phoenix event logging logic
            # This could use Phoenix's monitoring/RetrievalMonitor under the hood
            from cogniverse_telemetry_phoenix.evaluation.monitoring import (
                RetrievalMonitor,
            )

            monitor = RetrievalMonitor()
            monitor.log_retrieval_event({**data, "event_type": event_type})
        except Exception as e:
            logger.error(f"Failed to log experiment event: {e}")

    def log_session_evaluation(
        self,
        session_id: str,
        evaluation_name: str,
        session_score: float,
        session_outcome: str,
        turn_scores: Optional[List[float]] = None,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log session-level (multi-turn) evaluation result.

        This logs an evaluation for an entire conversation session, enabling
        trajectory-level analysis and fine-tuning data collection.

        Args:
            session_id: Session identifier (from span attributes)
            evaluation_name: Name of evaluation (e.g., "conversation_quality")
            session_score: Overall session score (0-1)
            session_outcome: Session outcome ("success", "partial", "failure")
            turn_scores: Optional per-turn scores
            explanation: Optional explanation
            metadata: Additional metadata
        """
        if not self._initialized:
            logger.warning("Provider not initialized, skipping session evaluation")
            return

        try:
            # Build annotation data
            annotation_data = {
                "evaluation_name": evaluation_name,
                "session_score": session_score,
                "session_outcome": session_outcome,
                "evaluated_at": datetime.now().isoformat(),
            }

            if turn_scores:
                annotation_data["turn_scores"] = turn_scores
                annotation_data["num_turns"] = len(turn_scores)
                annotation_data["avg_turn_score"] = sum(turn_scores) / len(turn_scores)

            if explanation:
                annotation_data["explanation"] = explanation

            if metadata:
                annotation_data.update(metadata)

            # Store as annotation using the telemetry provider's annotation store
            if self._telemetry_provider is not None:
                import asyncio

                # Get annotation store from provider
                annotation_store = self._telemetry_provider.annotations

                # Create task to add annotation
                async def add_annotation():
                    await annotation_store.add_annotation(
                        span_id=session_id,
                        name="session_evaluation",
                        label=session_outcome,
                        score=session_score,
                        metadata=annotation_data,
                    )

                # Run async operation
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(add_annotation())
                    else:
                        loop.run_until_complete(add_annotation())
                except RuntimeError:
                    asyncio.run(add_annotation())

            logger.info(
                f"Logged session evaluation for {session_id}: "
                f"{evaluation_name}={session_score:.2f} ({session_outcome})"
            )
        except Exception as e:
            logger.error(f"Failed to log session evaluation: {e}")
