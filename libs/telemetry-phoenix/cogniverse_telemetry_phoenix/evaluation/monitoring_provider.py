"""
Phoenix monitoring provider implementation.

Wraps Phoenix-specific monitoring functionality with provider abstraction.
"""

import logging
from typing import Any, Dict

import pandas as pd
import phoenix as px

from cogniverse_evaluation.providers.base import MonitoringProvider

logger = logging.getLogger(__name__)


class PhoenixMonitoringProvider(MonitoringProvider):
    """
    Phoenix implementation of monitoring provider.

    Provides monitoring operations using Phoenix client.
    """

    def __init__(self, phoenix_client: px.Client):
        """
        Initialize Phoenix monitoring provider.

        Args:
            phoenix_client: Phoenix client instance
        """
        self.client = phoenix_client
        self._alerts = {}  # Store alerts in memory for now

    async def create_alert(
        self, name: str, condition: str, threshold: float, project: str
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
        alert_id = f"{project}_{name}_{len(self._alerts)}"
        self._alerts[alert_id] = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "project": project,
            "status": "active",
            "triggered_count": 0,
        }
        logger.info(f"Created alert {alert_id}: {condition} > {threshold}")
        return alert_id

    async def get_metrics_window(
        self, project: str, window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Get metrics for recent time window.

        Args:
            project: Project name
            window_minutes: Size of time window

        Returns:
            Dictionary with recent metrics
        """
        try:
            from datetime import datetime, timedelta

            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=window_minutes)

            # Get recent spans
            spans_df = self.client.get_spans_dataframe(
                project_name=project, start_time=start_time, end_time=end_time
            )

            if spans_df is None or spans_df.empty:
                return {
                    "window_minutes": window_minutes,
                    "span_count": 0,
                    "error_count": 0,
                    "error_rate": 0,
                    "mean_latency_ms": 0,
                }

            # Calculate metrics
            metrics = {
                "window_minutes": window_minutes,
                "span_count": len(spans_df),
                "error_count": 0,
                "error_rate": 0,
                "mean_latency_ms": 0,
            }

            # Error rate
            if "status_code" in spans_df.columns:
                error_count = len(spans_df[spans_df["status_code"] == "ERROR"])
                metrics["error_count"] = error_count
                metrics["error_rate"] = (
                    error_count / len(spans_df) if len(spans_df) > 0 else 0
                )

            # Latency
            latency_col = None
            for col in ["latency_ms", "latency", "duration_ms", "duration"]:
                if col in spans_df.columns:
                    latency_col = col
                    break

            if latency_col:
                mean_latency = spans_df[latency_col].mean()
                metrics["mean_latency_ms"] = (
                    float(mean_latency) if not pd.isna(mean_latency) else 0
                )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics window: {e}")
            return {
                "window_minutes": window_minutes,
                "span_count": 0,
                "error_count": 0,
                "error_rate": 0,
                "mean_latency_ms": 0,
                "error": str(e),
            }

    async def check_alert_status(self, alert_id: str) -> Dict[str, Any]:
        """
        Check status of an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            Alert status information
        """
        if alert_id not in self._alerts:
            return {
                "alert_id": alert_id,
                "status": "not_found",
                "error": f"Alert {alert_id} not found",
            }

        alert = self._alerts[alert_id]
        return {
            "alert_id": alert_id,
            "name": alert["name"],
            "status": alert["status"],
            "condition": alert["condition"],
            "threshold": alert["threshold"],
            "project": alert["project"],
            "triggered_count": alert["triggered_count"],
        }
