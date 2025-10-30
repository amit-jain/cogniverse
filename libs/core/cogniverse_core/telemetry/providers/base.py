"""
Generic telemetry provider interfaces.

Core package defines ONLY abstractions - zero knowledge of Phoenix, LangSmith, etc.
Provider packages (cogniverse-telemetry-phoenix) implement these interfaces.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class TraceStore(ABC):
    """
    Query traces/spans from telemetry backend.

    Returns standardized DataFrames with provider-agnostic column names.
    """

    @abstractmethod
    async def get_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Query spans from backend.

        Args:
            project: Project/namespace identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            filters: Optional provider-specific filters
            limit: Maximum number of spans to return

        Returns:
            DataFrame with standardized columns:
            - context.span_id: Unique span identifier
            - name: Span operation name
            - attributes.*: Span attributes (flattened)
            - start_time: Span start timestamp
            - end_time: Span end timestamp
        """
        pass

    @abstractmethod
    async def get_span_by_id(
        self, span_id: str, project: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get single span by ID.

        Args:
            span_id: Span identifier
            project: Project/namespace identifier

        Returns:
            Span data as dictionary, None if not found
        """
        pass


class AnnotationStore(ABC):
    """
    Manage human/LLM annotations on spans.

    Annotations are used for approval workflows, feedback, evaluations.
    """

    @abstractmethod
    async def add_annotation(
        self,
        span_id: str,
        name: str,
        label: str,
        score: float,
        metadata: Dict[str, Any],
        project: str,
    ) -> str:
        """
        Add annotation to a span.

        Args:
            span_id: Target span identifier
            name: Annotation name/type (e.g., "item_status_update", "human_approval")
            label: Annotation label (e.g., "approved", "rejected")
            score: Numeric score (0.0-1.0)
            metadata: Additional metadata dictionary
            project: Project/namespace identifier

        Returns:
            Annotation identifier (if supported by backend)
        """
        pass

    @abstractmethod
    async def get_annotations(
        self,
        spans_df: pd.DataFrame,
        project: str,
        annotation_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get annotations for spans.

        Args:
            spans_df: DataFrame of spans (from TraceStore)
            project: Project/namespace identifier
            annotation_names: Optional filter by annotation names

        Returns:
            DataFrame with columns:
            - span_id: Associated span (if available)
            - result.label: Annotation label
            - result.score: Annotation score
            - metadata.*: Annotation metadata (flattened)
            - created_at: Timestamp
        """
        pass

    @abstractmethod
    async def log_evaluations(
        self,
        eval_name: str,
        evaluations_df: pd.DataFrame,
        project: str,
    ) -> None:
        """
        Log bulk evaluation results as span annotations.

        This is a batch operation for uploading evaluation results for multiple spans.
        Evaluations are stored as annotations with the evaluator name.

        Args:
            eval_name: Name of evaluation/evaluator (e.g., "relevance", "diversity", "golden_dataset")
            evaluations_df: DataFrame with evaluation results, must contain columns:
                - span_id: Span identifier to annotate
                - score: Evaluation score (typically 0.0-1.0)
                - label: Evaluation label (e.g., "good", "poor", "pass", "fail")
                - explanation: Optional explanation text (if not present, will use label)
            project: Project/namespace identifier

        Raises:
            ValueError: If evaluations_df is missing required columns
        """
        pass


class DatasetStore(ABC):
    """
    Manage training datasets.

    Datasets collect approved/validated data for model training.
    """

    @abstractmethod
    async def create_dataset(
        self, name: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new dataset.

        Args:
            name: Dataset name
            data: DataFrame with dataset records
            metadata: Optional metadata

        Returns:
            Dataset identifier
        """
        pass

    @abstractmethod
    async def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Load dataset by name.

        Args:
            name: Dataset name

        Returns:
            DataFrame with dataset records
        """
        pass

    @abstractmethod
    async def append_to_dataset(self, name: str, data: pd.DataFrame) -> None:
        """
        Append records to existing dataset.

        Args:
            name: Dataset name
            data: DataFrame with new records
        """
        pass


class ExperimentStore(ABC):
    """
    Manage experiments and evaluation runs.

    Experiments track evaluation results over time.
    """

    @abstractmethod
    async def create_experiment(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new experiment.

        Args:
            name: Experiment name
            metadata: Optional metadata

        Returns:
            Experiment identifier
        """
        pass

    @abstractmethod
    async def log_run(
        self,
        experiment_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log experiment run.

        Args:
            experiment_id: Experiment identifier
            inputs: Run inputs
            outputs: Run outputs/results
            metadata: Optional metadata

        Returns:
            Run identifier
        """
        pass


class AnalyticsStore(ABC):
    """
    Query analytics and aggregated metrics.

    Provides time-series metrics, aggregations, monitoring data.
    """

    @abstractmethod
    async def get_metrics(
        self,
        project: str,
        metric_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get time-series metrics.

        Args:
            project: Project/namespace identifier
            metric_names: List of metric names to retrieve
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            DataFrame with columns:
            - timestamp: Metric timestamp
            - metric_name: Metric name
            - value: Metric value
        """
        pass


class TelemetryProvider(ABC):
    """
    Base telemetry provider interface.

    Providers implement all store interfaces and handle backend-specific initialization.
    Core has zero knowledge of provider specifics (Phoenix, LangSmith, etc.).
    """

    def __init__(self, name: str):
        """
        Initialize provider with name.

        Args:
            name: Provider name (e.g., "phoenix", "langsmith")
        """
        self.name = name
        self._trace_store: Optional[TraceStore] = None
        self._annotation_store: Optional[AnnotationStore] = None
        self._dataset_store: Optional[DatasetStore] = None
        self._experiment_store: Optional[ExperimentStore] = None
        self._analytics_store: Optional[AnalyticsStore] = None

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize provider with configuration.

        Provider extracts its own keys from generic config dict.
        Core passes config without interpreting it.

        Args:
            config: Generic configuration dictionary containing:
                - tenant_id: Tenant identifier (always present)
                - Provider-specific keys (interpreted by provider)

        Example for Phoenix:
            config = {
                "tenant_id": "customer-123",
                "http_endpoint": "http://localhost:6006",
                "grpc_endpoint": "http://localhost:4317"
            }

        Example for LangSmith:
            config = {
                "tenant_id": "customer-123",
                "api_key": "xxx",
                "project": "my-project"
            }

        Provider validates required keys and raises ValueError if missing.
        """
        pass

    @abstractmethod
    def configure_span_export(
        self,
        endpoint: str,
        project_name: str,
        use_batch_export: bool = True,
    ) -> Any:
        """
        Configure OTLP span export for a project.

        Creates and returns a TracerProvider for the specified project.
        Used by TelemetryManager to set up span export for each tenant/project.

        Args:
            endpoint: OTLP gRPC endpoint for span export (e.g., "localhost:4317")
            project_name: Full project name (e.g., "cogniverse-tenant-service")
            use_batch_export: Use batch processor (True) vs simple/sync processor (False)

        Returns:
            TracerProvider instance (OpenTelemetry type)

        Raises:
            RuntimeError: If span export configuration fails
            ValueError: If endpoint or project_name invalid
        """
        pass

    @property
    def traces(self) -> TraceStore:
        """Get trace store (query spans)"""
        if self._trace_store is None:
            raise RuntimeError(f"{self.name} provider not initialized")
        return self._trace_store

    @property
    def annotations(self) -> AnnotationStore:
        """Get annotation store (manage annotations)"""
        if self._annotation_store is None:
            raise RuntimeError(f"{self.name} provider not initialized")
        return self._annotation_store

    @property
    def datasets(self) -> DatasetStore:
        """Get dataset store (manage training datasets)"""
        if self._dataset_store is None:
            raise RuntimeError(f"{self.name} provider not initialized")
        return self._dataset_store

    @property
    def experiments(self) -> ExperimentStore:
        """Get experiment store (manage experiments/runs)"""
        if self._experiment_store is None:
            raise RuntimeError(f"{self.name} provider not initialized")
        return self._experiment_store

    @property
    def analytics(self) -> AnalyticsStore:
        """Get analytics store (query metrics/aggregations)"""
        if self._analytics_store is None:
            raise RuntimeError(f"{self.name} provider not initialized")
        return self._analytics_store
