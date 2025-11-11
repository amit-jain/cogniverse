"""
Analytics and Visualization for telemetry traces/spans

This module provides analytics and visualization capabilities for traces/spans,
including request statistics, response time analysis, and outlier detection.

Uses telemetry provider abstraction for backend-agnostic trace querying.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class TraceMetrics:
    """Metrics extracted from traces"""

    trace_id: str
    timestamp: datetime
    duration_ms: float
    operation: str
    status: str
    profile: str | None = None
    strategy: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PhoenixAnalytics:
    """Analytics engine for telemetry traces and spans"""

    def __init__(
        self,
        provider: Optional[TelemetryProvider] = None,
        project: str = "cogniverse-default"
    ):
        """
        Initialize analytics engine.

        Args:
            provider: Telemetry provider (if None, uses TelemetryManager's provider)
            project: Project name for trace queries
        """
        # Get provider from TelemetryManager if not provided
        if provider is None:
            telemetry_manager = TelemetryManager()
            provider = telemetry_manager.provider

        self.provider = provider
        self.project = project
        self._cache = {}

    async def get_traces(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        operation_filter: str | None = None,
        limit: int = 10000,
    ) -> list[TraceMetrics]:
        """
        Fetch traces from telemetry backend with optional filters

        Args:
            start_time: Start of time range
            end_time: End of time range
            operation_filter: Filter by operation name
            limit: Maximum traces to fetch

        Returns:
            List of TraceMetrics objects
        """
        # Fetch spans using telemetry provider
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.project,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to fetch spans: {e}")
            return []

        if spans_df.empty:
            return []

        # Filter by operation if specified
        if operation_filter:
            import re

            pattern = re.compile(operation_filter)
            spans_df = spans_df[
                spans_df["name"].apply(
                    lambda x: bool(pattern.search(str(x))) if pd.notna(x) else False
                )
            ]

        # Group by trace to get trace-level metrics
        # Use parent_id to identify root spans (traces)
        root_spans = spans_df[spans_df["parent_id"].isna()].copy()

        # Convert to TraceMetrics
        metrics = []
        for _, span in root_spans.iterrows():
            try:
                # Calculate duration
                if pd.notna(span.get("end_time")) and pd.notna(span.get("start_time")):
                    duration_ms = (
                        span["end_time"] - span["start_time"]
                    ).total_seconds() * 1000
                else:
                    duration_ms = 0

                # Extract attributes/metadata
                attributes = span.get("attributes", {})
                if isinstance(attributes, str):
                    try:
                        import json

                        attributes = json.loads(attributes)
                    except Exception:
                        attributes = {}

                # Get status
                status_code = span.get("status_code", "OK")
                status = "success" if status_code in ["OK", "UNSET", None] else "error"

                metric = TraceMetrics(
                    trace_id=str(
                        span.get("trace_id", span.get("context.trace_id", ""))
                    ),
                    timestamp=span.get("start_time", datetime.now()),
                    duration_ms=duration_ms,
                    operation=str(span.get("name", "unknown")),
                    status=status,
                    profile=attributes.get(
                        "profile", attributes.get("metadata.profile")
                    ),
                    strategy=attributes.get(
                        "strategy",
                        attributes.get(
                            "ranking_strategy", attributes.get("metadata.strategy")
                        ),
                    ),
                    error=span.get("status_message") if status == "error" else None,
                    metadata=attributes,
                )
                metrics.append(metric)
            except Exception as e:
                logger.warning(f"Failed to parse trace: {e}")

        return metrics

    def calculate_statistics(
        self, traces: list[TraceMetrics], group_by: str | None = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive statistics from traces

        Args:
            traces: List of trace metrics
            group_by: Optional field to group by (e.g., 'operation', 'profile')

        Returns:
            Dictionary with statistics
        """
        if not traces:
            return {"error": "No traces provided"}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(
            [
                {
                    "trace_id": t.trace_id,
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                    "operation": t.operation,
                    "status": t.status,
                    "profile": t.profile or "unknown",
                    "strategy": t.strategy or "unknown",
                    "hour": t.timestamp.hour,
                    "day": t.timestamp.date(),
                }
                for t in traces
            ]
        )

        stats = {
            "total_requests": len(df),
            "time_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty else None,
                "end": df["timestamp"].max().isoformat() if not df.empty else None,
            },
        }

        # Overall response time statistics
        stats["response_time"] = {
            "mean": df["duration_ms"].mean(),
            "median": df["duration_ms"].median(),
            "min": df["duration_ms"].min(),
            "max": df["duration_ms"].max(),
            "std": df["duration_ms"].std(),
            "p50": df["duration_ms"].quantile(0.50),
            "p75": df["duration_ms"].quantile(0.75),
            "p90": df["duration_ms"].quantile(0.90),
            "p95": df["duration_ms"].quantile(0.95),
            "p99": df["duration_ms"].quantile(0.99),
        }

        # Success/failure rates
        if "status" in df.columns:
            status_counts = df["status"].value_counts().to_dict()
            total = len(df)
            stats["status"] = {
                "counts": status_counts,
                "success_rate": (
                    status_counts.get("success", 0) / total if total > 0 else 0
                ),
                "error_rate": status_counts.get("error", 0) / total if total > 0 else 0,
            }

        # Group by analysis
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by)

            group_stats = {}
            for name, group in grouped:
                group_stats[str(name)] = {
                    "count": len(group),
                    "mean_duration": group["duration_ms"].mean(),
                    "median_duration": group["duration_ms"].median(),
                    "p95_duration": group["duration_ms"].quantile(0.95),
                    "error_rate": (
                        (group["status"] == "error").mean()
                        if "status" in group.columns
                        else 0
                    ),
                }

            stats[f"by_{group_by}"] = group_stats

        # Temporal patterns
        stats["temporal"] = {
            "requests_by_hour": df.groupby("hour").size().to_dict(),
            "avg_duration_by_hour": df.groupby("hour")["duration_ms"].mean().to_dict(),
        }

        # Outlier detection
        outliers = self._detect_outliers(df["duration_ms"].values)
        stats["outliers"] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100 if len(df) > 0 else 0,
            "values": (
                outliers.tolist() if isinstance(outliers, np.ndarray) else outliers
            ),
        }

        return stats

    def _detect_outliers(self, data: np.ndarray, method: str = "iqr") -> np.ndarray:
        """
        Detect outliers in response times

        Args:
            data: Array of response times
            method: Detection method ('iqr' or 'zscore')

        Returns:
            Array of outlier values
        """
        if len(data) < 4:
            return np.array([])

        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)
            outliers = data[z_scores > 3]
        else:
            outliers = np.array([])

        return outliers

    def create_time_series_plot(
        self,
        traces: list[TraceMetrics],
        metric: str = "duration_ms",
        aggregation: str = "mean",
        time_window: str = "1min",
    ) -> go.Figure:
        """
        Create time series plot of metrics

        Args:
            traces: List of trace metrics
            metric: Metric to plot
            aggregation: Aggregation method (mean, median, max, min, count)
            time_window: Time window for aggregation (1min, 5min, 1h, etc.)

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                    "operation": t.operation,
                    "profile": t.profile or "unknown",
                    "strategy": t.strategy or "unknown",
                }
                for t in traces
            ]
        )

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Resample based on time window
        resampled = df.resample(time_window)

        if aggregation == "count":
            aggregated = resampled.size()
            y_title = "Request Count"
        elif aggregation == "mean":
            aggregated = resampled[metric].mean()
            y_title = f"Mean {metric}"
        elif aggregation == "median":
            aggregated = resampled[metric].median()
            y_title = f"Median {metric}"
        elif aggregation == "max":
            aggregated = resampled[metric].max()
            y_title = f"Max {metric}"
        elif aggregation == "min":
            aggregated = resampled[metric].min()
            y_title = f"Min {metric}"
        else:
            aggregated = resampled[metric].mean()
            y_title = f"Mean {metric}"

        # Create plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=aggregated.index,
                y=aggregated.values,
                mode="lines+markers",
                name=y_title,
                line={"color": "blue", "width": 2},
                marker={"size": 6},
            )
        )

        # Add percentile bands if aggregation is mean
        if aggregation == "mean" and metric == "duration_ms":
            p95 = resampled[metric].quantile(0.95)
            p50 = resampled[metric].quantile(0.50)

            fig.add_trace(
                go.Scatter(
                    x=p95.index,
                    y=p95.values,
                    mode="lines",
                    name="P95",
                    line={"color": "red", "width": 1, "dash": "dash"},
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=p50.index,
                    y=p50.values,
                    mode="lines",
                    name="P50 (Median)",
                    line={"color": "green", "width": 1, "dash": "dot"},
                )
            )

        fig.update_layout(
            title=f"{y_title} Over Time ({time_window} windows)",
            xaxis_title="Time",
            yaxis_title=y_title,
            hovermode="x unified",
            showlegend=True,
            height=500,
        )

        return fig

    def create_distribution_plot(
        self,
        traces: list[TraceMetrics],
        metric: str = "duration_ms",
        group_by: str | None = None,
    ) -> go.Figure:
        """
        Create distribution plot of metrics

        Args:
            traces: List of trace metrics
            metric: Metric to plot
            group_by: Optional grouping field

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "duration_ms": t.duration_ms,
                    "operation": t.operation,
                    "profile": t.profile or "unknown",
                    "strategy": t.strategy or "unknown",
                }
                for t in traces
            ]
        )

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Distribution", "Box Plot", "Violin Plot", "ECDF"),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "violin"}, {"type": "scatter"}],
            ],
        )

        # Histogram
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(
                    go.Histogram(x=group_data, name=str(group_name), opacity=0.7),
                    row=1,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Histogram(x=df[metric], name=metric, nbinsx=50), row=1, col=1
            )

        # Box plot
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(go.Box(y=group_data, name=str(group_name)), row=1, col=2)
        else:
            fig.add_trace(go.Box(y=df[metric], name=metric), row=1, col=2)

        # Violin plot
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(
                    go.Violin(
                        y=group_data,
                        name=str(group_name),
                        box_visible=True,
                        meanline_visible=True,
                        points="outliers",  # Show outlier points
                        pointpos=-1.8,  # Position points to the left
                        jitter=0.05,  # Add some jitter to see overlapping points
                    ),
                    row=2,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Violin(
                    y=df[metric],
                    name=metric,
                    box_visible=True,
                    meanline_visible=True,
                    points="outliers",  # Show outlier points
                    pointpos=-1.8,  # Position points to the left
                    jitter=0.05,  # Add some jitter to see overlapping points
                ),
                row=2,
                col=1,
            )

        # ECDF (Empirical Cumulative Distribution Function)
        sorted_data = np.sort(df[metric])
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        fig.add_trace(
            go.Scatter(x=sorted_data, y=ecdf, mode="lines", name="ECDF"), row=2, col=2
        )

        # Add percentile lines to ECDF
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(sorted_data, p)
            fig.add_vline(
                x=val,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"P{p}",
                row=2,
                col=2,
            )

        fig.update_layout(
            title=f"Distribution Analysis: {metric}", showlegend=True, height=800
        )

        return fig

    def create_heatmap(
        self,
        traces: list[TraceMetrics],
        x_field: str = "hour",
        y_field: str = "day",
        metric: str = "duration_ms",
        aggregation: str = "mean",
    ) -> go.Figure:
        """
        Create heatmap of metrics

        Args:
            traces: List of trace metrics
            x_field: Field for x-axis
            y_field: Field for y-axis
            metric: Metric to aggregate
            aggregation: Aggregation method

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                    "hour": t.timestamp.hour,
                    "day": t.timestamp.strftime("%Y-%m-%d"),
                    "weekday": t.timestamp.strftime("%A"),
                    "day_of_week": t.timestamp.strftime("%A"),  # alias for weekday
                    "profile": t.profile or "unknown",
                    "strategy": t.strategy or "unknown",
                    "operation": t.operation or "unknown",
                    "status": t.status or "unknown",
                }
                for t in traces
            ]
        )

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        # Pivot table for heatmap
        if aggregation == "count":
            pivot_table = df.pivot_table(
                index=y_field,
                columns=x_field,
                values=metric,
                aggfunc="count",
                fill_value=0,
            )
            z_label = "Request Count"
        else:
            pivot_table = df.pivot_table(
                index=y_field,
                columns=x_field,
                values=metric,
                aggfunc=aggregation,
                fill_value=0,
            )
            z_label = f"{aggregation.capitalize()} {metric}"

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale="Viridis",
                colorbar={"title": z_label},
            )
        )

        fig.update_layout(
            title=f"Heatmap: {z_label} by {x_field} and {y_field}",
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize(),
            height=600,
        )

        return fig

    def create_outlier_plot(
        self, traces: list[TraceMetrics], metric: str = "duration_ms"
    ) -> go.Figure:
        """
        Create plot highlighting outliers

        Args:
            traces: List of trace metrics
            metric: Metric to analyze

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                    "trace_id": t.trace_id,
                    "operation": t.operation,
                    "status": t.status,
                    "has_error": t.status == "error" or bool(t.error),
                }
                for t in traces
            ]
        )

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        # Handle error_rate metric
        if metric == "error_rate":
            # Calculate error rate by time window (e.g., hourly)
            df["hour"] = df["timestamp"].dt.floor("h")
            error_rates = (
                df.groupby("hour")
                .agg(
                    {
                        "has_error": "mean",
                        "timestamp": "first",
                        "operation": lambda x: "mixed",
                    }
                )
                .reset_index(drop=True)
            )
            error_rates["error_rate"] = error_rates["has_error"] * 100
            error_rates["trace_id"] = "aggregated"
            df = error_rates
            metric = "error_rate"

        # Detect outliers
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        outlier_mask = self._is_outlier(df[metric].values)
        normal_points = df[~outlier_mask]
        outlier_points = df[outlier_mask]

        fig = go.Figure()

        # Format hover template based on metric
        if metric == "error_rate":
            hover_format = (
                "<b>%{text}</b><br>Time: %{x}<br>Error Rate: %{y:.1f}%<extra></extra>"
            )
            outlier_hover_format = "<b>OUTLIER: %{text}</b><br>Time: %{x}<br>Error Rate: %{y:.1f}%<br>Trace: %{customdata}<extra></extra>"
        else:
            hover_format = (
                "<b>%{text}</b><br>Time: %{x}<br>Duration: %{y:.2f}ms<extra></extra>"
            )
            outlier_hover_format = "<b>OUTLIER: %{text}</b><br>Time: %{x}<br>Duration: %{y:.2f}ms<br>Trace: %{customdata}<extra></extra>"

        # Normal points
        fig.add_trace(
            go.Scatter(
                x=normal_points["timestamp"],
                y=normal_points[metric],
                mode="markers",
                name="Normal",
                marker={"color": "blue", "size": 6, "opacity": 0.6},
                text=normal_points["operation"],
                hovertemplate=hover_format,
            )
        )

        # Outlier points
        fig.add_trace(
            go.Scatter(
                x=outlier_points["timestamp"],
                y=outlier_points[metric],
                mode="markers",
                name="Outliers",
                marker={"color": "red", "size": 10, "symbol": "x"},
                text=outlier_points["operation"],
                hovertemplate=outlier_hover_format,
                customdata=outlier_points["trace_id"],
            )
        )

        # Add threshold lines
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Format annotation text based on metric
        if metric == "error_rate":
            threshold_text = f"Outlier Threshold ({upper_bound:.1f}%)"
            unit = "%"
        else:
            threshold_text = f"Outlier Threshold ({upper_bound:.2f}ms)"
            unit = "ms"

        fig.add_hline(
            y=upper_bound,
            line_dash="dash",
            line_color="orange",
            annotation_text=threshold_text,
        )

        # Add percentile lines
        for p in [50, 95, 99]:
            val = df[metric].quantile(p / 100)
            fig.add_hline(
                y=val,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"P{p} ({val:.2f}{unit})",
            )

        # Set appropriate title and y-axis label
        if metric == "error_rate":
            title = "Error Rate Outlier Detection"
            y_title = "Error Rate (%)"
        else:
            title = f"Response Time Outlier Detection ({metric})"
            y_title = f"{metric.replace('_', ' ').title()}"

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_title,
            hovermode="closest",
            showlegend=True,
            height=600,
        )

        return fig

    def _is_outlier(self, data: np.ndarray) -> np.ndarray:
        """Check if values are outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    def create_comparison_plot(
        self,
        traces: list[TraceMetrics],
        compare_field: str = "profile",
        metric: str = "duration_ms",
    ) -> go.Figure:
        """
        Create comparison plot between different groups

        Args:
            traces: List of trace metrics
            compare_field: Field to compare by
            metric: Metric to compare

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "duration_ms": t.duration_ms,
                    "profile": t.profile or "unknown",
                    "strategy": t.strategy or "unknown",
                    "operation": t.operation,
                }
                for t in traces
            ]
        )

        if df.empty or compare_field not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Mean {metric} by {compare_field}",
                f"Median {metric} by {compare_field}",
                f"P95 {metric} by {compare_field}",
                f"Request Count by {compare_field}",
            ),
        )

        # Group statistics
        grouped = df.groupby(compare_field)[metric]

        # Mean comparison
        means = grouped.mean().sort_values(ascending=False)
        fig.add_trace(go.Bar(x=means.index, y=means.values, name="Mean"), row=1, col=1)

        # Median comparison
        medians = grouped.median().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=medians.index, y=medians.values, name="Median"), row=1, col=2
        )

        # P95 comparison
        p95s = grouped.quantile(0.95).sort_values(ascending=False)
        fig.add_trace(go.Bar(x=p95s.index, y=p95s.values, name="P95"), row=2, col=1)

        # Count comparison
        counts = grouped.count().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, name="Count"), row=2, col=2
        )

        fig.update_layout(
            title=f"Comparison Analysis by {compare_field}",
            showlegend=False,
            height=800,
        )

        return fig

    def generate_report(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        output_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive analytics report

        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            output_file: Optional file to save report

        Returns:
            Report dictionary with all analytics
        """
        # Fetch traces
        traces = self.get_traces(start_time, end_time)

        if not traces:
            return {"error": "No traces found in the specified time range"}

        # Calculate statistics
        stats = self.calculate_statistics(traces)
        stats_by_profile = self.calculate_statistics(traces, group_by="profile")
        stats_by_operation = self.calculate_statistics(traces, group_by="operation")

        # Create visualizations
        time_series_fig = self.create_time_series_plot(traces)
        distribution_fig = self.create_distribution_plot(traces, group_by="profile")
        heatmap_fig = self.create_heatmap(traces)
        outlier_fig = self.create_outlier_plot(traces)
        comparison_fig = self.create_comparison_plot(traces)

        # Compile report
        report = {
            "summary": {
                "total_requests": stats["total_requests"],
                "time_range": stats["time_range"],
                "mean_response_time": stats["response_time"]["mean"],
                "p95_response_time": stats["response_time"]["p95"],
                "outlier_percentage": stats["outliers"]["percentage"],
            },
            "statistics": stats,
            "statistics_by_profile": stats_by_profile,
            "statistics_by_operation": stats_by_operation,
            "visualizations": {
                "time_series": time_series_fig.to_json(),
                "distribution": distribution_fig.to_json(),
                "heatmap": heatmap_fig.to_json(),
                "outliers": outlier_fig.to_json(),
                "comparison": comparison_fig.to_json(),
            },
        }

        # Save if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")

        return report
