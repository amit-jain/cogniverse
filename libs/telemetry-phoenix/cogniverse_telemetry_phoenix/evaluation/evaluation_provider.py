"""
Phoenix Evaluation Provider

Implements the generic EvaluationProvider interface for Phoenix backend.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cogniverse_evaluation.providers.base import EvaluationProvider
from cogniverse_telemetry_phoenix.evaluation.framework import (
    EvaluationResult,
    PhoenixEvaluatorFramework,
)

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
        self._telemetry_provider: Optional[Any] = None
        self._project_name: str = "evaluation"
        self._framework = PhoenixEvaluatorFramework()
        # Strong references to fire-and-forget annotation tasks so CPython
        # does not GC the coroutine before it runs.
        self._background_tasks: set[Any] = set()

    def _spawn_background(self, coro) -> Any:
        """Schedule a fire-and-forget coroutine while keeping a strong
        reference, so CPython does not GC the task before it runs."""
        import asyncio

        task = asyncio.get_running_loop().create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def framework(self) -> PhoenixEvaluatorFramework:
        """Return the Phoenix evaluator framework."""
        return self._framework

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: Configuration dictionary with:
                - tenant_id: Tenant identifier
                - http_endpoint: Phoenix server endpoint (resolved from TelemetryManager if not provided)
                - grpc_endpoint: gRPC endpoint (resolved from TelemetryManager if not provided)
                - project_name: Project name for telemetry (default: "evaluation")
                - Additional Phoenix-specific settings
        """
        tenant_id = config.get("tenant_id")
        if not tenant_id:
            raise ValueError(
                "tenant_id is required in PhoenixEvaluationProvider config"
            )
        self.tenant_id = tenant_id
        project_name = config.get("project_name", "evaluation")
        self._project_name = project_name

        # Resolve endpoints from TelemetryManager config (shared singleton)
        # This ensures evaluation providers use the same endpoints as telemetry providers
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        telemetry_manager = get_telemetry_manager()
        manager_config = telemetry_manager.config.provider_config

        self.http_endpoint = config.get(
            "http_endpoint",
            manager_config.get("http_endpoint", "http://localhost:6006"),
        )
        grpc_endpoint = config.get(
            "grpc_endpoint",
            manager_config.get("grpc_endpoint", "http://localhost:4317"),
        )

        # Get telemetry provider for this tenant. A failure here RAISES:
        # swallowing it left a provider that looked constructed but had
        # telemetry=None, so every later call died with an AttributeError
        # far from the root cause.
        try:
            from cogniverse_foundation.telemetry.registry import get_telemetry_registry

            # Get telemetry provider from singleton registry (shared cache)
            registry = get_telemetry_registry()
            self._telemetry_provider = registry.get(
                name="phoenix",
                tenant_id=self.tenant_id,
                config={
                    "project_name": project_name,
                    "http_endpoint": self.http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                },
            )
        except Exception as e:
            self._initialized = False
            logger.error(f"Failed to initialize Phoenix evaluation provider: {e}")
            raise

        self._initialized = True
        logger.info(
            f"Initialized Phoenix evaluation provider for tenant: {self.tenant_id}"
        )

    @property
    def telemetry(self) -> Any:
        """
        Get telemetry provider for traces/datasets/experiments.

        Returns:
            Phoenix telemetry provider instance
        """
        return self._telemetry_provider

    @staticmethod
    def _run_sync(coro):
        """Drive an async dataset-store call from these sync methods."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "create_experiment/log_evaluation are sync facades and cannot be "
            "driven from a running event loop; use telemetry.datasets directly"
        )

    @staticmethod
    def _experiment_dataset_name(experiment_id: str) -> str:
        return (
            experiment_id
            if experiment_id.startswith("experiment-")
            else f"experiment-{experiment_id}"
        )

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register an experiment as a durable dataset record.

        The registry dataset (``experiment-{name}``) holds the creation
        record; ``log_evaluation`` appends evaluation rows to it, so the
        experiment's history is readable back from the telemetry backend.

        Returns:
            Dict with ``id`` (the dataset name), ``name``, ``description``,
            ``metadata`` and ``created_at``.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        import json as _json

        import pandas as pd

        dataset_name = self._experiment_dataset_name(name)
        created_at = datetime.now(timezone.utc).isoformat()
        df = pd.DataFrame(
            [
                {
                    "event": "experiment_created",
                    "experiment": name,
                    "description": description or "",
                    "created_at": created_at,
                    "metadata": _json.dumps(metadata or {}, default=str),
                }
            ]
        )
        self._run_sync(
            self._telemetry_provider.datasets.create_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "description": description or f"Experiment {name}",
                    "input_keys": ["event", "experiment"],
                    "output_keys": ["description", "created_at", "metadata"],
                },
            )
        )
        logger.info(f"Registered experiment '{name}' as dataset {dataset_name}")
        return {
            "id": dataset_name,
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "created_at": created_at,
        }

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

            import pandas as pd
            from phoenix.client import Client

            df = pd.DataFrame(data)

            sync_client = Client(base_url=self.http_endpoint)
            dataset = sync_client.datasets.create_dataset(
                name=name,
                dataframe=df,
                dataset_description=description or f"Dataset: {name}",
            )

            logger.info(f"Created Phoenix dataset '{name}' with {len(data)} examples")
            return dataset
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
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        import json as _json

        import pandas as pd

        dataset_name = self._experiment_dataset_name(experiment_id)
        df = pd.DataFrame(
            [
                {
                    "event": "evaluation",
                    "experiment": experiment_id,
                    "evaluation_name": evaluation_name,
                    "score": float(score),
                    "label": label or "",
                    "explanation": explanation or "",
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": _json.dumps(metadata or {}, default=str),
                }
            ]
        )
        # append raises DatasetNotFoundError (a ValueError) when the
        # experiment was never created — logging into a void would silently
        # drop the evaluation.
        self._run_sync(
            self._telemetry_provider.datasets.append_to_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "input_keys": ["event", "experiment", "evaluation_name"],
                    "output_keys": ["score", "label", "explanation", "logged_at"],
                },
            )
        )
        logger.debug(
            f"Logged evaluation for {dataset_name}: {evaluation_name} = {score}"
        )

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
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        annotation_data = {
            "evaluation_name": evaluation_name,
            "session_score": session_score,
            "session_outcome": session_outcome,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        if turn_scores:
            annotation_data["turn_scores"] = turn_scores
            annotation_data["num_turns"] = len(turn_scores)
            annotation_data["avg_turn_score"] = sum(turn_scores) / len(turn_scores)

        if explanation:
            annotation_data["explanation"] = explanation

        if metadata:
            annotation_data.update(metadata)

        if self._telemetry_provider is not None:
            import asyncio

            annotation_store = self._telemetry_provider.annotations

            async def add_annotation():
                await annotation_store.add_annotation(
                    span_id=session_id,
                    name="session_evaluation",
                    label=session_outcome,
                    score=session_score,
                    metadata=annotation_data,
                    project=self._project_name,
                )
                logger.info(
                    f"Logged session evaluation for {session_id}: "
                    f"{evaluation_name}={session_score:.2f} ({session_outcome})"
                )

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — caller is sync. Run to completion so a
                # write failure raises to the caller (the dashboard shows it)
                # instead of reporting success for an unpersisted evaluation.
                asyncio.run(add_annotation())
            else:
                # Fire-and-forget on a live loop; failures must still be
                # visible, so log them from the task itself (the done-callback
                # only drops the strong reference, it never retrieves errors).
                async def add_annotation_logged():
                    try:
                        await add_annotation()
                    except Exception:
                        logger.error(
                            f"Failed to log session evaluation for {session_id}",
                            exc_info=True,
                        )

                self._spawn_background(add_annotation_logged())
