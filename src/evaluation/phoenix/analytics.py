"""
Phoenix Analytics and Visualization

This module provides analytics and visualization capabilities for Phoenix traces/spans,
including request statistics, response time analysis, and outlier detection.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict

import phoenix as px
import plotly.graph_objects as go
import plotly.express as px_plotly
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
    profile: Optional[str] = None
    strategy: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhoenixAnalytics:
    """Analytics engine for Phoenix traces and spans"""
    
    def __init__(self, phoenix_url: str = "http://localhost:6006"):
        self.phoenix_url = phoenix_url
        self.client = px.Client()
        self._cache = {}
    
    def get_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation_filter: Optional[str] = None,
        limit: int = 10000
    ) -> List[TraceMetrics]:
        """
        Fetch traces from Phoenix with optional filters
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            operation_filter: Filter by operation name
            limit: Maximum traces to fetch
            
        Returns:
            List of TraceMetrics objects
        """
        # Build filter
        filter_dict = {}
        if start_time:
            filter_dict["timestamp"] = {"$gte": start_time.isoformat()}
        if end_time:
            if "timestamp" not in filter_dict:
                filter_dict["timestamp"] = {}
            filter_dict["timestamp"]["$lte"] = end_time.isoformat()
        if operation_filter:
            filter_dict["name"] = operation_filter
        
        # Fetch traces
        traces = self.client.get_traces(filter=filter_dict, limit=limit)
        
        # Convert to TraceMetrics
        metrics = []
        for trace in traces:
            try:
                metric = TraceMetrics(
                    trace_id=trace.get("trace_id", ""),
                    timestamp=datetime.fromisoformat(trace.get("timestamp", datetime.now().isoformat())),
                    duration_ms=trace.get("duration_ms", 0.0),
                    operation=trace.get("name", "unknown"),
                    status=trace.get("status", "success"),
                    profile=trace.get("metadata", {}).get("profile"),
                    strategy=trace.get("metadata", {}).get("strategy"),
                    error=trace.get("error", None),
                    metadata=trace.get("metadata", {})
                )
                metrics.append(metric)
            except Exception as e:
                logger.warning(f"Failed to parse trace: {e}")
        
        return metrics
    
    def calculate_statistics(
        self,
        traces: List[TraceMetrics],
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
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
        df = pd.DataFrame([
            {
                "trace_id": t.trace_id,
                "timestamp": t.timestamp,
                "duration_ms": t.duration_ms,
                "operation": t.operation,
                "status": t.status,
                "profile": t.profile,
                "strategy": t.strategy,
                "hour": t.timestamp.hour,
                "day": t.timestamp.date()
            }
            for t in traces
        ])
        
        stats = {
            "total_requests": len(df),
            "time_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty else None,
                "end": df["timestamp"].max().isoformat() if not df.empty else None
            }
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
            "p99": df["duration_ms"].quantile(0.99)
        }
        
        # Success/failure rates
        if "status" in df.columns:
            status_counts = df["status"].value_counts().to_dict()
            total = len(df)
            stats["status"] = {
                "counts": status_counts,
                "success_rate": status_counts.get("success", 0) / total if total > 0 else 0,
                "error_rate": status_counts.get("error", 0) / total if total > 0 else 0
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
                    "error_rate": (group["status"] == "error").mean() if "status" in group.columns else 0
                }
            
            stats[f"by_{group_by}"] = group_stats
        
        # Temporal patterns
        stats["temporal"] = {
            "requests_by_hour": df.groupby("hour").size().to_dict(),
            "avg_duration_by_hour": df.groupby("hour")["duration_ms"].mean().to_dict()
        }
        
        # Outlier detection
        outliers = self._detect_outliers(df["duration_ms"].values)
        stats["outliers"] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100 if len(df) > 0 else 0,
            "values": outliers.tolist() if isinstance(outliers, np.ndarray) else outliers
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
        traces: List[TraceMetrics],
        metric: str = "duration_ms",
        aggregation: str = "mean",
        time_window: str = "1min"
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
        df = pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "duration_ms": t.duration_ms,
                "operation": t.operation,
                "profile": t.profile or "unknown",
                "strategy": t.strategy or "unknown"
            }
            for t in traces
        ])
        
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
        
        fig.add_trace(go.Scatter(
            x=aggregated.index,
            y=aggregated.values,
            mode='lines+markers',
            name=y_title,
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add percentile bands if aggregation is mean
        if aggregation == "mean" and metric == "duration_ms":
            p95 = resampled[metric].quantile(0.95)
            p50 = resampled[metric].quantile(0.50)
            
            fig.add_trace(go.Scatter(
                x=p95.index,
                y=p95.values,
                mode='lines',
                name='P95',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=p50.index,
                y=p50.values,
                mode='lines',
                name='P50 (Median)',
                line=dict(color='green', width=1, dash='dot')
            ))
        
        fig.update_layout(
            title=f"{y_title} Over Time ({time_window} windows)",
            xaxis_title="Time",
            yaxis_title=y_title,
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_distribution_plot(
        self,
        traces: List[TraceMetrics],
        metric: str = "duration_ms",
        group_by: Optional[str] = None
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
        df = pd.DataFrame([
            {
                "duration_ms": t.duration_ms,
                "operation": t.operation,
                "profile": t.profile or "unknown",
                "strategy": t.strategy or "unknown"
            }
            for t in traces
        ])
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Distribution", "Box Plot", "Violin Plot", "ECDF"),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "violin"}, {"type": "scatter"}]]
        )
        
        # Histogram
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(
                    go.Histogram(x=group_data, name=str(group_name), opacity=0.7),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Histogram(x=df[metric], name=metric, nbinsx=50),
                row=1, col=1
            )
        
        # Box plot
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(
                    go.Box(y=group_data, name=str(group_name)),
                    row=1, col=2
                )
        else:
            fig.add_trace(
                go.Box(y=df[metric], name=metric),
                row=1, col=2
            )
        
        # Violin plot
        if group_by and group_by in df.columns:
            for group_name in df[group_by].unique():
                group_data = df[df[group_by] == group_name][metric]
                fig.add_trace(
                    go.Violin(y=group_data, name=str(group_name), box_visible=True),
                    row=2, col=1
                )
        else:
            fig.add_trace(
                go.Violin(y=df[metric], name=metric, box_visible=True),
                row=2, col=1
            )
        
        # ECDF (Empirical Cumulative Distribution Function)
        sorted_data = np.sort(df[metric])
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(
            go.Scatter(x=sorted_data, y=ecdf, mode='lines', name='ECDF'),
            row=2, col=2
        )
        
        # Add percentile lines to ECDF
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(sorted_data, p)
            fig.add_vline(x=val, line_dash="dash", line_color="gray", 
                         annotation_text=f"P{p}", row=2, col=2)
        
        fig.update_layout(
            title=f"Distribution Analysis: {metric}",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_heatmap(
        self,
        traces: List[TraceMetrics],
        x_field: str = "hour",
        y_field: str = "day",
        metric: str = "duration_ms",
        aggregation: str = "mean"
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
        df = pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "duration_ms": t.duration_ms,
                "hour": t.timestamp.hour,
                "day": t.timestamp.strftime("%Y-%m-%d"),
                "weekday": t.timestamp.strftime("%A"),
                "profile": t.profile or "unknown",
                "strategy": t.strategy or "unknown"
            }
            for t in traces
        ])
        
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
                aggfunc='count',
                fill_value=0
            )
            z_label = "Request Count"
        else:
            pivot_table = df.pivot_table(
                index=y_field,
                columns=x_field,
                values=metric,
                aggfunc=aggregation,
                fill_value=0
            )
            z_label = f"{aggregation.capitalize()} {metric}"
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            colorbar=dict(title=z_label)
        ))
        
        fig.update_layout(
            title=f"Heatmap: {z_label} by {x_field} and {y_field}",
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize(),
            height=600
        )
        
        return fig
    
    def create_outlier_plot(
        self,
        traces: List[TraceMetrics],
        metric: str = "duration_ms"
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
        df = pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "duration_ms": t.duration_ms,
                "trace_id": t.trace_id,
                "operation": t.operation
            }
            for t in traces
        ])
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig
        
        # Detect outliers
        outlier_mask = self._is_outlier(df[metric].values)
        normal_points = df[~outlier_mask]
        outlier_points = df[outlier_mask]
        
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=normal_points["timestamp"],
            y=normal_points[metric],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6, opacity=0.6),
            text=normal_points["operation"],
            hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Duration: %{y:.2f}ms<extra></extra>'
        ))
        
        # Outlier points
        fig.add_trace(go.Scatter(
            x=outlier_points["timestamp"],
            y=outlier_points[metric],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=10, symbol='x'),
            text=outlier_points["operation"],
            hovertemplate='<b>OUTLIER: %{text}</b><br>Time: %{x}<br>Duration: %{y:.2f}ms<br>Trace: %{customdata}<extra></extra>',
            customdata=outlier_points["trace_id"]
        ))
        
        # Add threshold lines
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange",
                     annotation_text=f"Outlier Threshold ({upper_bound:.2f}ms)")
        
        # Add percentile lines
        for p in [50, 95, 99]:
            val = df[metric].quantile(p/100)
            fig.add_hline(y=val, line_dash="dot", line_color="gray",
                         annotation_text=f"P{p} ({val:.2f}ms)")
        
        fig.update_layout(
            title=f"Outlier Detection: {metric}",
            xaxis_title="Time",
            yaxis_title=f"{metric} (ms)",
            hovermode='closest',
            showlegend=True,
            height=600
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
        traces: List[TraceMetrics],
        compare_field: str = "profile",
        metric: str = "duration_ms"
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
        df = pd.DataFrame([
            {
                "duration_ms": t.duration_ms,
                "profile": t.profile or "unknown",
                "strategy": t.strategy or "unknown",
                "operation": t.operation
            }
            for t in traces
        ])
        
        if df.empty or compare_field not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Mean {metric} by {compare_field}",
                f"Median {metric} by {compare_field}",
                f"P95 {metric} by {compare_field}",
                f"Request Count by {compare_field}"
            )
        )
        
        # Group statistics
        grouped = df.groupby(compare_field)[metric]
        
        # Mean comparison
        means = grouped.mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=means.index, y=means.values, name="Mean"),
            row=1, col=1
        )
        
        # Median comparison
        medians = grouped.median().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=medians.index, y=medians.values, name="Median"),
            row=1, col=2
        )
        
        # P95 comparison
        p95s = grouped.quantile(0.95).sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=p95s.index, y=p95s.values, name="P95"),
            row=2, col=1
        )
        
        # Count comparison
        counts = grouped.count().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, name="Count"),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Comparison Analysis by {compare_field}",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
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
                "outlier_percentage": stats["outliers"]["percentage"]
            },
            "statistics": stats,
            "statistics_by_profile": stats_by_profile,
            "statistics_by_operation": stats_by_operation,
            "visualizations": {
                "time_series": time_series_fig.to_json(),
                "distribution": distribution_fig.to_json(),
                "heatmap": heatmap_fig.to_json(),
                "outliers": outlier_fig.to_json(),
                "comparison": comparison_fig.to_json()
            }
        }
        
        # Save if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")
        
        return report