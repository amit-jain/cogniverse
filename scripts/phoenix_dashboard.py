#!/usr/bin/env python3
"""
Interactive Phoenix Analytics Dashboard using Streamlit

This dashboard provides real-time visualization and analysis of Phoenix traces
collected during Cogniverse evaluations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.phoenix.analytics import PhoenixAnalytics
from src.evaluation.phoenix.root_cause_analysis import RootCauseAnalyzer

# Page configuration
st.set_page_config(
    page_title="Phoenix Analytics Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .plot-container {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e3e6eb;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = PhoenixAnalytics()
if 'rca_analyzer' not in st.session_state:
    st.session_state.rca_analyzer = RootCauseAnalyzer()
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30
if 'enable_rca' not in st.session_state:
    st.session_state.enable_rca = False


def format_duration(ms):
    """Format duration in milliseconds to human readable"""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    if delta is not None:
        st.metric(label, value, delta, delta_color=delta_color)
    else:
        st.metric(label, value)


def main():
    # Header
    st.title("🔥 Phoenix Analytics Dashboard")
    st.markdown("Real-time visualization and analysis of Cogniverse evaluation traces")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Phoenix connection
        phoenix_url = st.text_input(
            "Phoenix URL",
            value="http://localhost:6006",
            help="URL of the Phoenix server"
        )
        
        if st.button("🔄 Test Connection"):
            try:
                test_analytics = PhoenixAnalytics(phoenix_url)
                test_traces = test_analytics.get_traces(limit=1)
                st.success("✅ Connected to Phoenix!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        
        st.divider()
        
        # Time range selection
        st.subheader("📅 Time Range")
        
        time_preset = st.selectbox(
            "Quick Select",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Custom"]
        )
        
        if time_preset == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=1))
                start_time = st.time_input("Start Time", datetime.min.time())
            with col2:
                end_date = st.date_input("End Date", datetime.now().date())
                end_time = st.time_input("End Time", datetime.now().time())
            
            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)
        else:
            end_datetime = datetime.now()
            if time_preset == "Last Hour":
                start_datetime = end_datetime - timedelta(hours=1)
            elif time_preset == "Last 6 Hours":
                start_datetime = end_datetime - timedelta(hours=6)
            elif time_preset == "Last 24 Hours":
                start_datetime = end_datetime - timedelta(hours=24)
            else:  # Last 7 Days
                start_datetime = end_datetime - timedelta(days=7)
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        
        operation_filter = st.text_input(
            "Operation Filter",
            placeholder="e.g., video_search",
            help="Filter traces by operation name"
        )
        
        profile_filter = st.multiselect(
            "Profile Filter",
            ["frame_based_colpali", "direct_video_global", "direct_video_global_large"],
            help="Filter by video processing profile"
        )
        
        strategy_filter = st.multiselect(
            "Strategy Filter",
            ["binary_binary", "float_float", "hybrid_binary_bm25", "phased"],
            help="Filter by ranking strategy"
        )
        
        st.divider()
        
        # Root Cause Analysis settings
        st.subheader("🔍 Root Cause Analysis")
        
        st.session_state.enable_rca = st.checkbox(
            "Enable RCA",
            value=st.session_state.enable_rca,
            help="Enable automated root cause analysis for failures and performance issues"
        )
        
        if st.session_state.enable_rca:
            rca_options = st.expander("RCA Options", expanded=False)
            with rca_options:
                include_performance = st.checkbox(
                    "Include Performance Analysis",
                    value=True,
                    help="Analyze slow traces in addition to failures"
                )
                perf_threshold = st.slider(
                    "Performance Threshold Percentile",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Traces above this percentile are considered slow"
                )
                st.session_state.rca_options = {
                    "include_performance": include_performance,
                    "performance_threshold_percentile": perf_threshold
                }
        
        st.divider()
        
        # Auto-refresh settings
        st.subheader("🔄 Auto Refresh")
        
        st.session_state.auto_refresh = st.checkbox(
            "Enable Auto Refresh",
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=300,
                value=st.session_state.refresh_interval,
                step=5
            )
            
            # Show countdown
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
            time_until_refresh = max(0, st.session_state.refresh_interval - time_since_refresh)
            st.progress(
                1 - (time_until_refresh / st.session_state.refresh_interval),
                text=f"Next refresh in {int(time_until_refresh)}s"
            )
        
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Main content area
    try:
        # Fetch traces
        with st.spinner("Loading traces..."):
            traces = st.session_state.analytics.get_traces(
                start_time=start_datetime,
                end_time=end_datetime,
                operation_filter=operation_filter if operation_filter else None,
                limit=10000
            )
        
        if not traces:
            st.warning("No traces found for the selected time range and filters.")
            return
        
        # Apply additional filters
        if profile_filter or strategy_filter:
            filtered_traces = []
            for trace in traces:
                if profile_filter and trace.profile not in profile_filter:
                    continue
                if strategy_filter and trace.strategy not in strategy_filter:
                    continue
                filtered_traces.append(trace)
            traces = filtered_traces
        
        if not traces:
            st.warning("No traces match the selected filters.")
            return
        
        # Calculate statistics
        stats = st.session_state.analytics.calculate_statistics(traces)
        stats_by_profile = st.session_state.analytics.calculate_statistics(traces, group_by="profile")
        stats_by_operation = st.session_state.analytics.calculate_statistics(traces, group_by="operation")
        
        # Display tabs - add RCA tab if enabled
        if st.session_state.enable_rca:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview",
                "📈 Time Series",
                "📉 Distributions",
                "🔥 Heatmaps",
                "⚠️ Outliers",
                "🔍 Trace Explorer",
                "🎯 Root Cause Analysis"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview",
                "📈 Time Series",
                "📉 Distributions",
                "🔥 Heatmaps",
                "⚠️ Outliers",
                "🔍 Trace Explorer"
            ])
        
        with tab1:
            st.header("Overview")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                create_metric_card(
                    "Total Requests",
                    f"{stats['total_requests']:,}",
                    None
                )
            
            with col2:
                mean_time = stats['response_time']['mean']
                create_metric_card(
                    "Mean Response Time",
                    format_duration(mean_time),
                    None
                )
            
            with col3:
                p95_time = stats['response_time']['p95']
                create_metric_card(
                    "P95 Response Time",
                    format_duration(p95_time),
                    None
                )
            
            with col4:
                success_rate = stats.get('status', {}).get('success_rate', 0)
                create_metric_card(
                    "Success Rate",
                    f"{success_rate:.1%}",
                    None
                )
            
            with col5:
                outlier_pct = stats['outliers']['percentage']
                create_metric_card(
                    "Outliers",
                    f"{outlier_pct:.1f}%",
                    None
                )
            
            st.divider()
            
            # Response time statistics table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Time Statistics")
                rt_stats = stats['response_time']
                rt_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 
                               'P50', 'P75', 'P90', 'P95', 'P99'],
                    'Value (ms)': [
                        rt_stats['mean'], rt_stats['median'], rt_stats['min'],
                        rt_stats['max'], rt_stats['std'], rt_stats['p50'],
                        rt_stats['p75'], rt_stats['p90'], rt_stats['p95'],
                        rt_stats['p99']
                    ]
                })
                rt_df['Value (ms)'] = rt_df['Value (ms)'].round(2)
                st.dataframe(rt_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Status Distribution")
                if 'status' in stats:
                    status_counts = stats['status']['counts']
                    fig = px.pie(
                        values=list(status_counts.values()),
                        names=list(status_counts.keys()),
                        title="Request Status Distribution",
                        color_discrete_map={'success': '#2ecc71', 'error': '#e74c3c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Comparison by profile
            if stats_by_profile.get('by_profile'):
                st.subheader("Performance by Profile")
                profile_data = []
                for profile, profile_stats in stats_by_profile['by_profile'].items():
                    profile_data.append({
                        'Profile': profile,
                        'Count': profile_stats['count'],
                        'Mean Duration (ms)': profile_stats['mean_duration'],
                        'P95 Duration (ms)': profile_stats['p95_duration'],
                        'Error Rate': profile_stats['error_rate']
                    })
                
                profile_df = pd.DataFrame(profile_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        profile_df,
                        x='Profile',
                        y='Mean Duration (ms)',
                        title='Mean Response Time by Profile',
                        color='Mean Duration (ms)',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        profile_df,
                        x='Profile',
                        y='Count',
                        title='Request Count by Profile',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Time Series Analysis")
            
            # Time window selection
            col1, col2, col3 = st.columns(3)
            with col1:
                time_window = st.selectbox(
                    "Aggregation Window",
                    ["1min", "5min", "15min", "30min", "1h"],
                    index=1
                )
            with col2:
                aggregation = st.selectbox(
                    "Aggregation Method",
                    ["mean", "median", "max", "min", "count"],
                    index=0
                )
            with col3:
                show_percentiles = st.checkbox("Show Percentile Bands", value=True)
            
            # Create time series plot
            fig = st.session_state.analytics.create_time_series_plot(
                traces,
                metric="duration_ms",
                aggregation=aggregation,
                time_window=time_window
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Request rate over time
            st.subheader("Request Rate Over Time")
            rate_fig = st.session_state.analytics.create_time_series_plot(
                traces,
                metric="duration_ms",
                aggregation="count",
                time_window=time_window
            )
            rate_fig.update_layout(
                title="Request Rate Over Time",
                yaxis_title="Requests per " + time_window,
                height=400
            )
            st.plotly_chart(rate_fig, use_container_width=True)
        
        with tab3:
            st.header("Distribution Analysis")
            
            # Distribution plot
            group_by = st.selectbox(
                "Group By",
                ["None", "profile", "strategy", "operation"],
                index=0
            )
            
            group_by_value = None if group_by == "None" else group_by
            
            dist_fig = st.session_state.analytics.create_distribution_plot(
                traces,
                metric="duration_ms",
                group_by=group_by_value
            )
            dist_fig.update_layout(height=800)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Comparison plot
            if group_by_value:
                st.subheader(f"Comparison by {group_by.capitalize()}")
                comp_fig = st.session_state.analytics.create_comparison_plot(
                    traces,
                    compare_field=group_by_value,
                    metric="duration_ms"
                )
                comp_fig.update_layout(height=800)
                st.plotly_chart(comp_fig, use_container_width=True)
        
        with tab4:
            st.header("Heatmap Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_field = st.selectbox(
                    "X-Axis",
                    ["hour", "weekday", "profile", "strategy"],
                    index=0
                )
            with col2:
                y_field = st.selectbox(
                    "Y-Axis",
                    ["day", "weekday", "profile", "strategy"],
                    index=0
                )
            
            heatmap_metric = st.selectbox(
                "Metric",
                ["duration_ms", "count"],
                index=0,
                format_func=lambda x: "Response Time" if x == "duration_ms" else "Request Count"
            )
            
            heatmap_agg = st.selectbox(
                "Aggregation",
                ["mean", "median", "max", "count"],
                index=0 if heatmap_metric == "duration_ms" else 3
            )
            
            heatmap_fig = st.session_state.analytics.create_heatmap(
                traces,
                x_field=x_field,
                y_field=y_field,
                metric=heatmap_metric if heatmap_metric == "duration_ms" else "duration_ms",
                aggregation=heatmap_agg
            )
            heatmap_fig.update_layout(height=600)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with tab5:
            st.header("Outlier Analysis")
            
            outlier_fig = st.session_state.analytics.create_outlier_plot(
                traces,
                metric="duration_ms"
            )
            outlier_fig.update_layout(height=600)
            st.plotly_chart(outlier_fig, use_container_width=True)
            
            # Outlier statistics
            col1, col2, col3 = st.columns(3)
            
            outlier_stats = stats['outliers']
            with col1:
                st.metric("Outlier Count", outlier_stats['count'])
            with col2:
                st.metric("Outlier Percentage", f"{outlier_stats['percentage']:.2f}%")
            with col3:
                if outlier_stats.get('values'):
                    max_outlier = max(outlier_stats['values'])
                    st.metric("Max Outlier", format_duration(max_outlier))
            
            # Outlier details table
            if outlier_stats.get('values') and len(outlier_stats['values']) > 0:
                st.subheader("Outlier Details")
                
                # Create DataFrame of outlier traces
                outlier_traces = []
                for trace in traces:
                    if trace.duration_ms in outlier_stats['values']:
                        outlier_traces.append({
                            'Timestamp': trace.timestamp,
                            'Operation': trace.operation,
                            'Duration (ms)': trace.duration_ms,
                            'Profile': trace.profile or 'N/A',
                            'Strategy': trace.strategy or 'N/A',
                            'Status': trace.status
                        })
                
                if outlier_traces:
                    outlier_df = pd.DataFrame(outlier_traces)
                    outlier_df = outlier_df.sort_values('Duration (ms)', ascending=False)
                    st.dataframe(
                        outlier_df.head(20),
                        use_container_width=True,
                        hide_index=True
                    )
        
        with tab6:
            st.header("Trace Explorer")
            
            # Search and filter
            col1, col2, col3 = st.columns(3)
            with col1:
                search_query = st.text_input("Search traces", placeholder="Enter trace ID or operation")
            with col2:
                status_filter = st.selectbox("Status", ["All", "success", "error"])
            with col3:
                sort_by = st.selectbox("Sort By", ["Timestamp", "Duration", "Operation"])
            
            # Convert traces to DataFrame
            trace_data = []
            for trace in traces:
                if search_query and search_query.lower() not in str(trace).lower():
                    continue
                if status_filter != "All" and trace.status != status_filter:
                    continue
                
                trace_data.append({
                    'Timestamp': trace.timestamp,
                    'Trace ID': trace.trace_id[:8] + "...",
                    'Operation': trace.operation,
                    'Duration (ms)': trace.duration_ms,
                    'Profile': trace.profile or 'N/A',
                    'Strategy': trace.strategy or 'N/A',
                    'Status': trace.status,
                    'Error': trace.error or ''
                })
            
            if trace_data:
                trace_df = pd.DataFrame(trace_data)
                
                # Sort
                if sort_by == "Timestamp":
                    trace_df = trace_df.sort_values('Timestamp', ascending=False)
                elif sort_by == "Duration":
                    trace_df = trace_df.sort_values('Duration (ms)', ascending=False)
                else:
                    trace_df = trace_df.sort_values('Operation')
                
                # Display with pagination
                rows_per_page = st.slider("Rows per page", 10, 100, 25)
                total_pages = len(trace_df) // rows_per_page + (1 if len(trace_df) % rows_per_page else 0)
                
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=1
                )
                
                start_idx = (page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(trace_df))
                
                st.info(f"Showing {start_idx + 1}-{end_idx} of {len(trace_df)} traces")
                
                # Style the dataframe
                styled_df = trace_df.iloc[start_idx:end_idx].style.applymap(
                    lambda x: 'color: red' if x == 'error' else '',
                    subset=['Status']
                ).format({
                    'Duration (ms)': '{:.2f}',
                    'Timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                st.dataframe(
                    trace_df.iloc[start_idx:end_idx],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No traces match the current filters")
        
        # Root Cause Analysis Tab (if enabled)
        if st.session_state.enable_rca:
            with tab7:
                st.header("🎯 Root Cause Analysis")
                
                # Run RCA button
                col1, col2, col3 = st.columns([2, 2, 6])
                with col1:
                    run_rca = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
                with col2:
                    if st.button("📥 Export RCA Report", use_container_width=True):
                        if 'rca_results' in st.session_state:
                            rca_report = st.session_state.rca_analyzer.generate_rca_report(
                                st.session_state.rca_results,
                                format="markdown"
                            )
                            st.download_button(
                                label="Download RCA Report",
                                data=rca_report,
                                file_name=f"rca_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                
                if run_rca or 'rca_results' not in st.session_state:
                    with st.spinner("Performing root cause analysis..."):
                        # Get RCA options
                        rca_options = st.session_state.get('rca_options', {
                            'include_performance': True,
                            'performance_threshold_percentile': 95
                        })
                        
                        # Run RCA
                        st.session_state.rca_results = st.session_state.rca_analyzer.analyze_failures(
                            traces,
                            include_performance=rca_options['include_performance'],
                            performance_threshold_percentile=rca_options['performance_threshold_percentile']
                        )
                
                # Display RCA results
                if 'rca_results' in st.session_state:
                    rca_results = st.session_state.rca_results
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Failed Traces",
                            rca_results['summary']['failed_traces'],
                            help="Number of traces with errors"
                        )
                    with col2:
                        st.metric(
                            "Failure Rate",
                            f"{rca_results['summary']['failure_rate']:.1%}",
                            help="Percentage of failed traces"
                        )
                    with col3:
                        st.metric(
                            "Performance Degraded",
                            rca_results['summary']['performance_degraded'],
                            help="Traces with slow performance"
                        )
                    with col4:
                        st.metric(
                            "Root Causes Found",
                            len(rca_results.get('root_causes', [])),
                            help="Number of identified root causes"
                        )
                    
                    st.divider()
                    
                    # Root Causes
                    if rca_results.get('root_causes'):
                        st.subheader("🎯 Identified Root Causes")
                        
                        for i, hypothesis in enumerate(rca_results['root_causes'][:5], 1):
                            with st.expander(
                                f"{i}. {hypothesis.hypothesis} "
                                f"(Confidence: {hypothesis.confidence:.0%})",
                                expanded=(i <= 2)  # Expand first 2
                            ):
                                # Evidence
                                st.markdown("**Evidence:**")
                                for evidence in hypothesis.evidence:
                                    st.markdown(f"- {evidence}")
                                
                                # Suggested Action
                                st.info(f"**Suggested Action:** {hypothesis.suggested_action}")
                                
                                # Category and affected traces
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Category:** `{hypothesis.category}`")
                                with col2:
                                    if hypothesis.affected_traces:
                                        st.markdown(f"**Sample Traces:** {len(hypothesis.affected_traces)} affected")
                                        if st.checkbox(f"Show trace IDs #{i}", key=f"show_traces_{i}"):
                                            for trace_id in hypothesis.affected_traces[:5]:
                                                st.code(trace_id)
                    
                    # Failure Analysis
                    if rca_results.get('failure_analysis'):
                        st.divider()
                        st.subheader("📊 Failure Pattern Analysis")
                        
                        failure_analysis = rca_results['failure_analysis']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Error types distribution
                            if failure_analysis.get('error_types'):
                                st.markdown("**Error Type Distribution:**")
                                error_df = pd.DataFrame(
                                    list(failure_analysis['error_types'].items()),
                                    columns=['Error Type', 'Count']
                                ).sort_values('Count', ascending=False)
                                
                                fig = px.bar(
                                    error_df,
                                    x='Count',
                                    y='Error Type',
                                    orientation='h',
                                    title='Error Types',
                                    color='Count',
                                    color_continuous_scale='Reds'
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Failed operations
                            if failure_analysis.get('failed_operations'):
                                st.markdown("**Failed Operations:**")
                                ops_df = pd.DataFrame(
                                    list(failure_analysis['failed_operations'].items()),
                                    columns=['Operation', 'Failures']
                                ).sort_values('Failures', ascending=False).head(10)
                                
                                fig = px.bar(
                                    ops_df,
                                    x='Failures',
                                    y='Operation',
                                    orientation='h',
                                    title='Operations with Failures',
                                    color='Failures',
                                    color_continuous_scale='OrRd'
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Temporal patterns
                        if failure_analysis.get('temporal_patterns'):
                            st.markdown("**Temporal Patterns:**")
                            for pattern in failure_analysis['temporal_patterns']:
                                if pattern['type'] == 'burst':
                                    st.warning(
                                        f"⚠️ **Failure Burst Detected:** "
                                        f"{pattern['failure_count']} failures in "
                                        f"{pattern['duration_minutes']:.1f} minutes "
                                        f"starting at {pattern['start_time']}"
                                    )
                                elif pattern['type'] == 'hourly':
                                    st.info(
                                        f"📊 **High failure rate at hour {pattern['hour']}:** "
                                        f"{pattern['failure_rate']:.1%} "
                                        f"({pattern['failed_requests']}/{pattern['total_requests']} requests)"
                                    )
                    
                    # Performance Analysis
                    if rca_results.get('performance_analysis') and rca_results['performance_analysis']:
                        st.divider()
                        st.subheader("⚡ Performance Degradation Analysis")
                        
                        perf_analysis = rca_results['performance_analysis']
                        
                        # Latency comparison
                        if perf_analysis.get('latency_distribution'):
                            latency = perf_analysis['latency_distribution']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Slow Traces Mean",
                                    f"{latency.get('slow_mean', 0):.0f}ms"
                                )
                            with col2:
                                st.metric(
                                    "Normal Traces Mean",
                                    f"{latency.get('normal_mean', 0):.0f}ms"
                                )
                            with col3:
                                slowdown = latency.get('slowdown_factor', 0)
                                st.metric(
                                    "Slowdown Factor",
                                    f"{slowdown:.1f}x",
                                    delta=f"{(slowdown - 1) * 100:.0f}% slower"
                                )
                        
                        # Slow operations
                        if perf_analysis.get('slow_operations'):
                            st.markdown("**Slow Operations:**")
                            slow_ops_df = pd.DataFrame(
                                list(perf_analysis['slow_operations'].items()),
                                columns=['Operation', 'Slow Count']
                            ).sort_values('Slow Count', ascending=False).head(5)
                            st.dataframe(slow_ops_df, use_container_width=True, hide_index=True)
                    
                    # Recommendations
                    if rca_results.get('recommendations'):
                        st.divider()
                        st.subheader("💡 Recommendations")
                        
                        for rec in rca_results['recommendations']:
                            priority_color = {
                                'high': '🔴',
                                'medium': '🟡',
                                'low': '🟢'
                            }.get(rec['priority'], '⚪')
                            
                            with st.expander(
                                f"{priority_color} {rec['recommendation']} ({rec['priority'].upper()} Priority)",
                                expanded=(rec['priority'] == 'high')
                            ):
                                st.markdown("**Actions:**")
                                for detail in rec['details']:
                                    st.markdown(f"- {detail}")
                                
                                if rec.get('affected_components'):
                                    st.markdown("**Affected Components:**")
                                    for component in rec['affected_components']:
                                        st.markdown(f"- {component}")
                    
                    # Statistical Analysis
                    if rca_results.get('statistical_analysis'):
                        st.divider()
                        st.subheader("📈 Statistical Analysis")
                        
                        stats_analysis = rca_results['statistical_analysis']
                        
                        # Profile comparison
                        if stats_analysis.get('profile_comparison'):
                            st.markdown("**Performance by Profile:**")
                            profile_df = pd.DataFrame.from_dict(
                                stats_analysis['profile_comparison'],
                                orient='index'
                            ).reset_index()
                            profile_df.columns = ['Profile', 'Total', 'Failed', 'Failure Rate', 'Mean Duration', 'P95 Duration']
                            profile_df['Failure Rate'] = (profile_df['Failure Rate'] * 100).round(1)
                            profile_df['Mean Duration'] = profile_df['Mean Duration'].round(0)
                            profile_df['P95 Duration'] = profile_df['P95 Duration'].round(0)
                            
                            st.dataframe(
                                profile_df.style.background_gradient(subset=['Failure Rate'], cmap='RdYlGn_r'),
                                use_container_width=True,
                                hide_index=True
                            )
                
                else:
                    st.info("Click 'Run Analysis' to perform root cause analysis on the current traces")
        
        # Export section in sidebar
        with st.sidebar:
            st.divider()
            st.subheader("📥 Export")
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "HTML Report"]
            )
            
            if st.button("Export Data", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "JSON":
                    report = st.session_state.analytics.generate_report(
                        start_time=start_datetime,
                        end_time=end_datetime
                    )
                    st.download_button(
                        label="Download JSON Report",
                        data=json.dumps(report, indent=2, default=str),
                        file_name=f"phoenix_report_{timestamp}.json",
                        mime="application/json"
                    )
                
                elif export_format == "CSV":
                    trace_df = pd.DataFrame([
                        {
                            'timestamp': t.timestamp,
                            'trace_id': t.trace_id,
                            'duration_ms': t.duration_ms,
                            'operation': t.operation,
                            'profile': t.profile,
                            'strategy': t.strategy,
                            'status': t.status
                        }
                        for t in traces
                    ])
                    csv = trace_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"phoenix_traces_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                else:  # HTML Report
                    # Generate comprehensive HTML report
                    html_content = generate_html_report(
                        traces, stats, start_datetime, end_datetime
                    )
                    st.download_button(
                        label="Download HTML Report",
                        data=html_content,
                        file_name=f"phoenix_report_{timestamp}.html",
                        mime="text/html"
                    )
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= st.session_state.refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()


def generate_html_report(traces, stats, start_time, end_time):
    """Generate a comprehensive HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phoenix Analytics Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-label {{ font-weight: bold; color: #666; }}
            .metric-value {{ font-size: 1.2em; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Phoenix Analytics Report</h1>
        <p>Generated: {datetime.now().isoformat()}</p>
        <p>Time Range: {start_time.isoformat()} to {end_time.isoformat()}</p>
        
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-label">Total Requests</div>
            <div class="metric-value">{stats['total_requests']:,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean Response Time</div>
            <div class="metric-value">{stats['response_time']['mean']:.2f} ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">P95 Response Time</div>
            <div class="metric-value">{stats['response_time']['p95']:.2f} ms</div>
        </div>
        
        <h2>Response Time Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value (ms)</th></tr>
            <tr><td>Mean</td><td>{stats['response_time']['mean']:.2f}</td></tr>
            <tr><td>Median</td><td>{stats['response_time']['median']:.2f}</td></tr>
            <tr><td>P95</td><td>{stats['response_time']['p95']:.2f}</td></tr>
            <tr><td>P99</td><td>{stats['response_time']['p99']:.2f}</td></tr>
        </table>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    main()