"""
Base Evaluation Provider Interface

Defines the contract for evaluation providers (Phoenix, Langsmith, etc).
Each provider implements experiment tracking, dataset management, and evaluation result formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class EvaluatorFramework(ABC):
    """
    Abstract base class for evaluator frameworks.

    Each telemetry provider implements this to expose its evaluator base class,
    evaluation result type, and result factory method.
    """

    @abstractmethod
    def get_evaluator_base_class(self) -> type:
        """Return the provider's base evaluator class."""
        pass

    @abstractmethod
    def get_evaluation_result_type(self) -> type:
        """Return the provider's evaluation result type."""
        pass

    @abstractmethod
    def create_evaluation_result(
        self,
        score: float,
        label: str,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a provider-specific evaluation result."""
        pass


class EvaluationProvider(ABC):
    """
    Abstract base class for evaluation providers.

    Providers implement backend-specific logic for:
    - Experiment creation and tracking
    - Dataset management
    - Evaluation result formatting
    - Monitoring and logging

    Example implementations:
    - PhoenixEvaluationProvider: Phoenix experiments/datasets
    - LangsmithEvaluationProvider: Langsmith experiments
    """

    def __init__(self):
        """Initialize the provider"""
        self._initialized = False

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration
                - tenant_id: Tenant identifier
                - project_name: Project name (optional)
                - Additional provider-specific settings
        """
        pass

    @abstractmethod
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            description: Experiment description
            metadata: Additional metadata

        Returns:
            Provider-specific experiment object
        """
        pass

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        data: List[Dict[str, Any]],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            data: List of dataset examples
            description: Dataset description
            metadata: Additional metadata

        Returns:
            Provider-specific dataset object
        """
        pass

    @abstractmethod
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
        Log an evaluation result.

        Args:
            experiment_id: Experiment identifier
            evaluation_name: Name of the evaluation
            score: Evaluation score
            label: Optional label
            explanation: Optional explanation
            metadata: Additional metadata
        """
        pass

    @abstractmethod
    def create_evaluation_result(
        self,
        score: float,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create an evaluation result object.

        This is used by evaluators to return provider-specific result types.
        For Phoenix: returns phoenix.experiments.types.EvaluationResult
        For Langsmith: returns langsmith-specific result type

        Args:
            score: Evaluation score (typically 0-1)
            label: Optional categorical label
            explanation: Optional explanation text
            metadata: Additional metadata dict

        Returns:
            Provider-specific evaluation result object
        """
        pass

    @abstractmethod
    def get_experiment_url(self, experiment_id: str) -> str:
        """
        Get the URL for viewing an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            URL string for viewing the experiment in provider UI
        """
        pass

    @abstractmethod
    def get_dataset_url(self, dataset_id: str) -> str:
        """
        Get the URL for viewing a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            URL string for viewing the dataset in provider UI
        """
        pass

    def log_experiment_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log a generic experiment event (optional).

        Args:
            event_type: Type of event
            data: Event data
        """
        # Optional - not all providers may need this
        pass

    def is_initialized(self) -> bool:
        """Check if provider is initialized"""
        return self._initialized


# Data structures for providers


@dataclass
class TraceMetrics:
    """Metrics extracted from traces"""

    trace_id: str
    timestamp: datetime
    duration_ms: float
    operation: str
    status: str
    profile: Optional[str] = None
    strategy: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Represents a pattern associated with failures"""

    pattern_type: str  # 'operation', 'profile', 'strategy', 'time', 'parameter'
    pattern_value: Any
    failure_rate: float
    occurrence_count: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    correlation_strength: float = 0.0


@dataclass
class RootCauseHypothesis:
    """Hypothesis for root cause"""

    hypothesis: str
    confidence: float
    evidence: List[str]
    affected_traces: List[str]
    suggested_action: str
    category: str  # 'configuration', 'resource', 'timeout', 'data', 'model'
    patterns: List[FailurePattern] = field(default_factory=list)


# Additional provider interfaces


class AnalyticsProvider(ABC):
    """
    Provider for trace analytics and visualization.

    Implementations analyze telemetry traces and generate reports/visualizations.
    """

    @abstractmethod
    async def get_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation_filter: Optional[str] = None,
        limit: int = 10000,
    ) -> List[TraceMetrics]:
        """
        Fetch traces from telemetry backend with optional filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            operation_filter: Filter by operation name
            limit: Maximum traces to fetch

        Returns:
            List of TraceMetrics objects
        """
        pass

    @abstractmethod
    def calculate_statistics(
        self,
        traces: List[TraceMetrics],
    ) -> Dict[str, Any]:
        """
        Calculate statistical metrics from traces.

        Args:
            traces: List of trace metrics

        Returns:
            Dictionary with statistics (mean, p50, p95, p99, etc)
        """
        pass

    @abstractmethod
    def create_time_series_plot(
        self,
        traces: List[TraceMetrics],
        metric: str = "duration",
    ) -> Any:
        """
        Create time series visualization.

        Args:
            traces: List of trace metrics
            metric: Metric to plot (duration, throughput, error_rate)

        Returns:
            Provider-specific plot object (e.g., Plotly figure)
        """
        pass

    @abstractmethod
    def create_distribution_plot(
        self,
        traces: List[TraceMetrics],
        metric: str = "duration",
    ) -> Any:
        """
        Create distribution histogram.

        Args:
            traces: List of trace metrics
            metric: Metric to plot

        Returns:
            Provider-specific plot object
        """
        pass

    @abstractmethod
    def generate_report(
        self,
        traces: List[TraceMetrics],
        format: str = "markdown",
    ) -> str:
        """
        Generate analytics report.

        Args:
            traces: List of trace metrics
            format: Report format (markdown, html, json)

        Returns:
            Formatted report string
        """
        pass


class MonitoringProvider(ABC):
    """
    Provider for real-time performance monitoring.

    Implementations track metrics and trigger alerts.
    """

    @abstractmethod
    def start(self) -> None:
        """Start monitoring background process."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring background process."""
        pass

    @abstractmethod
    def log_retrieval_event(self, event: Dict[str, Any]) -> None:
        """
        Log a retrieval event for monitoring.

        Args:
            event: Event data (query, latency, results, etc)
        """
        pass

    @abstractmethod
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get currently active alerts.

        Returns:
            List of alert dictionaries
        """
        pass

    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get current metrics summary.

        Returns:
            Dictionary with current metric values
        """
        pass


class RootCauseProvider(ABC):
    """
    Provider for automated root cause analysis.

    Implementations analyze failure patterns and generate hypotheses.
    """

    @abstractmethod
    def analyze_failures(
        self,
        traces: List[TraceMetrics],
        time_window_hours: int = 24,
    ) -> List[RootCauseHypothesis]:
        """
        Analyze failure patterns and generate root cause hypotheses.

        Args:
            traces: List of trace metrics (including failures)
            time_window_hours: Time window for analysis

        Returns:
            List of root cause hypotheses, sorted by confidence
        """
        pass


class DatasetsProvider(ABC):
    """
    Provider for evaluation dataset management.

    Implementations manage evaluation datasets and run batch evaluations.
    """

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new evaluation dataset.

        Args:
            name: Dataset name
            data: DataFrame with dataset examples
            description: Dataset description
            metadata: Additional metadata

        Returns:
            Provider-specific dataset object
        """
        pass

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load an existing dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DataFrame with dataset examples
        """
        pass

    @abstractmethod
    def run_batch_evaluation(
        self,
        dataset_name: str,
        evaluators: List[Any],
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on a dataset.

        Args:
            dataset_name: Name of the dataset
            evaluators: List of evaluator functions
            experiment_name: Optional experiment name

        Returns:
            Dictionary with evaluation results and metrics
        """
        pass
