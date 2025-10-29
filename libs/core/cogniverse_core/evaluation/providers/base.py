"""
Base abstractions for evaluation providers.

This module defines provider-agnostic interfaces for evaluation functionality,
separating concerns from the main TelemetryProvider.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from cogniverse_core.telemetry.providers.base import TelemetryProvider


class EvaluatorFramework(ABC):
    """
    Evaluator base classes and result types.

    Different telemetry providers have different evaluator frameworks:
    - Phoenix: phoenix.experiments.evaluators.base.Evaluator
    - LangSmith: langsmith.evaluation.Evaluator

    This abstraction allows evaluation code to work with any provider.
    """

    @abstractmethod
    def get_evaluator_base_class(self) -> type:
        """
        Return provider's base evaluator class.

        Returns:
            Base class that evaluators should inherit from
        """
        pass

    @abstractmethod
    def get_evaluation_result_type(self) -> type:
        """
        Return provider's evaluation result type.

        Returns:
            Type used for evaluation results
        """
        pass

    @abstractmethod
    def create_evaluation_result(
        self,
        score: float,
        label: str,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create evaluation result in provider format.

        Args:
            score: Numeric score for the evaluation
            label: Classification label
            explanation: Human-readable explanation
            metadata: Additional metadata

        Returns:
            Evaluation result in provider-specific format
        """
        pass


class AnalyticsProvider(ABC):
    """
    Analytics and visualization operations for traces and spans.

    Provides methods for:
    - Statistical analysis of traces
    - Distribution analysis
    - Outlier detection
    - Performance metrics
    """

    @abstractmethod
    async def get_trace_statistics(
        self,
        project: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Get statistical summary of traces in a project.

        Args:
            project: Project name
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary with statistics (count, latency, error_rate, etc.)
        """
        pass

    @abstractmethod
    async def get_span_distribution(
        self,
        project: str,
        metric: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Get distribution of a metric across spans.

        Args:
            project: Project name
            metric: Metric to analyze (e.g., 'latency', 'token_count')
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with distribution data
        """
        pass

    @abstractmethod
    async def detect_outliers(
        self,
        spans_df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> List[str]:
        """
        Detect outlier spans based on metrics.

        Args:
            spans_df: DataFrame with span data
            method: Detection method ('iqr', 'zscore', etc.)
            threshold: Threshold for outlier detection

        Returns:
            List of span IDs identified as outliers
        """
        pass

    @abstractmethod
    async def get_metric_trends(
        self,
        project: str,
        metric: str,
        window_minutes: int = 60,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Get time-series trends for a metric.

        Args:
            project: Project name
            metric: Metric to track
            window_minutes: Aggregation window size
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with time-series data
        """
        pass


class MonitoringProvider(ABC):
    """
    Real-time monitoring and alerting for evaluation metrics.

    Provides methods for:
    - Creating alerts on metric thresholds
    - Monitoring recent performance
    - Real-time metric windows
    """

    @abstractmethod
    async def create_alert(
        self,
        name: str,
        condition: str,
        threshold: float,
        project: str
    ) -> str:
        """
        Create an alert for monitoring metrics.

        Args:
            name: Alert name
            condition: Condition expression (e.g., 'latency > threshold')
            threshold: Threshold value
            project: Project to monitor

        Returns:
            Alert ID
        """
        pass

    @abstractmethod
    async def get_metrics_window(
        self,
        project: str,
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Get metrics for recent time window.

        Args:
            project: Project name
            window_minutes: Size of time window

        Returns:
            Dictionary with recent metrics
        """
        pass

    @abstractmethod
    async def check_alert_status(self, alert_id: str) -> Dict[str, Any]:
        """
        Check status of an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            Alert status information
        """
        pass


class EvaluatorProvider(ABC):
    """
    Main evaluator provider combining all evaluation capabilities.

    This is the primary interface used by evaluation code. It combines:
    - Evaluator framework (base classes, result types)
    - Analytics (statistics, distributions, outliers)
    - Monitoring (alerts, real-time metrics)
    - Telemetry (traces, datasets, experiments) - reused from TelemetryProvider
    """

    @property
    @abstractmethod
    def framework(self) -> EvaluatorFramework:
        """
        Get evaluator framework for this provider.

        Returns:
            EvaluatorFramework instance
        """
        pass

    @property
    @abstractmethod
    def analytics(self) -> AnalyticsProvider:
        """
        Get analytics provider for this provider.

        Returns:
            AnalyticsProvider instance
        """
        pass

    @property
    @abstractmethod
    def monitoring(self) -> MonitoringProvider:
        """
        Get monitoring provider for this provider.

        Returns:
            MonitoringProvider instance
        """
        pass

    @property
    @abstractmethod
    def telemetry(self) -> "TelemetryProvider":
        """
        Get telemetry provider for traces/datasets/experiments.

        Returns:
            TelemetryProvider instance
        """
        pass
