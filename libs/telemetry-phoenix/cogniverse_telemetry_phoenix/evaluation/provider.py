"""
Phoenix evaluator provider implementation.

Main provider class combining all Phoenix evaluation capabilities.
"""

from typing import TYPE_CHECKING

from cogniverse_core.evaluation.providers.base import (
    AnalyticsProvider,
    EvaluatorFramework,
    EvaluatorProvider,
    MonitoringProvider,
)

from .analytics_provider import PhoenixAnalyticsProvider
from .framework import PhoenixEvaluatorFramework
from .monitoring_provider import PhoenixMonitoringProvider

if TYPE_CHECKING:
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider


class PhoenixEvaluatorProvider(EvaluatorProvider):
    """
    Phoenix implementation of evaluator provider.

    Combines Phoenix evaluator framework, analytics, and monitoring capabilities
    with telemetry provider for complete evaluation support.
    """

    def __init__(self, telemetry_provider: "PhoenixProvider"):
        """
        Initialize Phoenix evaluator provider.

        Args:
            telemetry_provider: PhoenixProvider instance for telemetry operations
        """
        self._telemetry = telemetry_provider

        # Initialize Phoenix-specific components
        self._framework = PhoenixEvaluatorFramework()
        self._analytics = PhoenixAnalyticsProvider(telemetry_provider.client)
        self._monitoring = PhoenixMonitoringProvider(telemetry_provider.client)

    @property
    def framework(self) -> EvaluatorFramework:
        """
        Get Phoenix evaluator framework.

        Returns:
            PhoenixEvaluatorFramework instance
        """
        return self._framework

    @property
    def analytics(self) -> AnalyticsProvider:
        """
        Get Phoenix analytics provider.

        Returns:
            PhoenixAnalyticsProvider instance
        """
        return self._analytics

    @property
    def monitoring(self) -> MonitoringProvider:
        """
        Get Phoenix monitoring provider.

        Returns:
            PhoenixMonitoringProvider instance
        """
        return self._monitoring

    @property
    def telemetry(self) -> "PhoenixProvider":
        """
        Get telemetry provider for traces/datasets/experiments.

        Returns:
            PhoenixProvider instance
        """
        return self._telemetry
