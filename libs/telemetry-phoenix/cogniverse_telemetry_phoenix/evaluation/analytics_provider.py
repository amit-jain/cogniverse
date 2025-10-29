"""
Phoenix analytics provider implementation.

Wraps Phoenix-specific analytics functionality with provider abstraction.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import phoenix as px
from cogniverse_core.evaluation.providers.base import AnalyticsProvider

logger = logging.getLogger(__name__)


class PhoenixAnalyticsProvider(AnalyticsProvider):
    """
    Phoenix implementation of analytics provider.

    Provides analytics operations using Phoenix client.
    """

    def __init__(self, phoenix_client: px.Client):
        """
        Initialize Phoenix analytics provider.

        Args:
            phoenix_client: Phoenix client instance
        """
        self.client = phoenix_client

    async def get_trace_statistics(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
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
        try:
            # Get spans dataframe from Phoenix
            spans_df = self.client.get_spans_dataframe(
                project_name=project,
                start_time=start_time,
                end_time=end_time
            )

            if spans_df is None or spans_df.empty:
                return {
                    "count": 0,
                    "mean_latency_ms": 0,
                    "p50_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "error_rate": 0,
                }

            # Calculate statistics
            latency_col = None
            for col in ["latency_ms", "latency", "duration_ms", "duration"]:
                if col in spans_df.columns:
                    latency_col = col
                    break

            stats = {
                "count": len(spans_df),
                "mean_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_rate": 0,
            }

            if latency_col:
                latencies = spans_df[latency_col].dropna()
                if not latencies.empty:
                    stats["mean_latency_ms"] = float(latencies.mean())
                    stats["p50_latency_ms"] = float(latencies.quantile(0.50))
                    stats["p95_latency_ms"] = float(latencies.quantile(0.95))
                    stats["p99_latency_ms"] = float(latencies.quantile(0.99))

            # Calculate error rate
            if "status_code" in spans_df.columns:
                error_count = len(spans_df[spans_df["status_code"] == "ERROR"])
                stats["error_rate"] = error_count / len(spans_df) if len(spans_df) > 0 else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get trace statistics: {e}")
            return {
                "count": 0,
                "mean_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_rate": 0,
                "error": str(e),
            }

    async def get_span_distribution(
        self,
        project: str,
        metric: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
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
        try:
            spans_df = self.client.get_spans_dataframe(
                project_name=project,
                start_time=start_time,
                end_time=end_time
            )

            if spans_df is None or spans_df.empty:
                return pd.DataFrame()

            # Check if metric exists in dataframe
            if metric not in spans_df.columns:
                logger.warning(f"Metric '{metric}' not found in spans")
                return pd.DataFrame()

            # Return distribution
            return spans_df[[metric]].dropna()

        except Exception as e:
            logger.error(f"Failed to get span distribution: {e}")
            return pd.DataFrame()

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
        if spans_df.empty:
            return []

        outlier_ids = []

        # Find latency column
        latency_col = None
        for col in ["latency_ms", "latency", "duration_ms", "duration"]:
            if col in spans_df.columns:
                latency_col = col
                break

        if not latency_col:
            logger.warning("No latency column found for outlier detection")
            return []

        latencies = spans_df[latency_col].dropna()
        if latencies.empty:
            return []

        if method == "iqr":
            # IQR method
            q1 = latencies.quantile(0.25)
            q3 = latencies.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = spans_df[
                (spans_df[latency_col] < lower_bound) |
                (spans_df[latency_col] > upper_bound)
            ]

        elif method == "zscore":
            # Z-score method
            mean = latencies.mean()
            std = latencies.std()
            if std == 0:
                return []

            z_scores = (spans_df[latency_col] - mean) / std
            outliers = spans_df[abs(z_scores) > threshold]

        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return []

        # Extract span IDs
        if "span_id" in outliers.columns:
            outlier_ids = outliers["span_id"].tolist()
        elif "context.span_id" in outliers.columns:
            outlier_ids = outliers["context.span_id"].tolist()

        return outlier_ids

    async def get_metric_trends(
        self,
        project: str,
        metric: str,
        window_minutes: int = 60,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
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
        try:
            spans_df = self.client.get_spans_dataframe(
                project_name=project,
                start_time=start_time,
                end_time=end_time
            )

            if spans_df is None or spans_df.empty:
                return pd.DataFrame()

            # Check if metric and timestamp exist
            if metric not in spans_df.columns:
                logger.warning(f"Metric '{metric}' not found in spans")
                return pd.DataFrame()

            # Find timestamp column
            timestamp_col = None
            for col in ["start_time", "timestamp", "end_time"]:
                if col in spans_df.columns:
                    timestamp_col = col
                    break

            if not timestamp_col:
                logger.warning("No timestamp column found")
                return pd.DataFrame()

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(spans_df[timestamp_col]):
                spans_df[timestamp_col] = pd.to_datetime(spans_df[timestamp_col])

            # Resample and aggregate
            spans_df = spans_df.set_index(timestamp_col)
            window_str = f"{window_minutes}T"  # T = minutes
            trends = spans_df[[metric]].resample(window_str).agg(["mean", "count", "std"])
            trends.columns = ["mean", "count", "std"]
            trends = trends.reset_index()

            return trends

        except Exception as e:
            logger.error(f"Failed to get metric trends: {e}")
            return pd.DataFrame()
