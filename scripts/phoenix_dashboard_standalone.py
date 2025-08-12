#!/usr/bin/env python3
"""
Interactive Phoenix Analytics Dashboard using Streamlit
Standalone version that avoids videoprism dependencies
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
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import analytics module directly without going through __init__.py
def import_module_directly(module_path):
    """Import a module directly from file path to avoid __init__.py execution"""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import only what we need
analytics_module = import_module_directly(project_root / "src/evaluation/phoenix/analytics.py")
rca_module = import_module_directly(project_root / "src/evaluation/phoenix/root_cause_analysis.py")

PhoenixAnalytics = analytics_module.PhoenixAnalytics
RootCauseAnalyzer = rca_module.RootCauseAnalyzer

# Page configuration
st.set_page_config(
    page_title="Phoenix Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def format_timestamp(ts_str):
    """Format timestamp string to be more readable"""
    try:
        if isinstance(ts_str, str):
            # Parse ISO format timestamp
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y %I:%M:%S %p")
        elif hasattr(ts_str, 'strftime'):
            return ts_str.strftime("%b %d, %Y %I:%M:%S %p")
        else:
            return str(ts_str)
    except:
        return str(ts_str)

def format_time_range(start_str, end_str):
    """Format a time range to be more readable"""
    try:
        start = format_timestamp(start_str)
        end = format_timestamp(end_str)
        
        # If same day, show more compact format
        if start[:12] == end[:12]:  # Same date
            return f"{start} - {end.split(' ', 2)[2]}"
        else:
            return f"{start} - {end}"
    except:
        return f"{start_str} to {end_str}"

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #333;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #1e1e1e;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    .stMetric [data-testid="metric-container"] label {
        color: #b0b0b0 !important;
        font-weight: 500;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #4ade80 !important;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"].negative {
        color: #f87171 !important;
    }
    .plot-container {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = PhoenixAnalytics()

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Sidebar configuration
with st.sidebar:
    st.title("🔥 Phoenix Dashboard")
    st.markdown("---")
    
    # Time range selection
    st.subheader("Time Range")
    time_range = st.selectbox(
        "Select time range",
        ["Last 15 minutes", "Last hour", "Last 6 hours", "Last 24 hours", 
         "Last 7 days", "Custom range"],
        index=2  # Default to "Last 6 hours"
    )
    
    if time_range == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.now() - timedelta(days=1))
            start_time = st.time_input("Start time", datetime.now().time())
        with col2:
            end_date = st.date_input("End date", datetime.now())
            end_time = st.time_input("End time", datetime.now().time())
        
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    else:
        end_datetime = datetime.now()
        if time_range == "Last 15 minutes":
            start_datetime = end_datetime - timedelta(minutes=15)
        elif time_range == "Last hour":
            start_datetime = end_datetime - timedelta(hours=1)
        elif time_range == "Last 6 hours":
            start_datetime = end_datetime - timedelta(hours=6)
        elif time_range == "Last 24 hours":
            start_datetime = end_datetime - timedelta(days=1)
        elif time_range == "Last 7 days":
            start_datetime = end_datetime - timedelta(days=7)
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    
    # Operation filter
    operation_filter = st.text_input(
        "Operation name (regex)",
        placeholder="e.g. search.*|evaluate.*"
    )
    
    # Profile filter
    profile_filter = st.multiselect(
        "Profiles",
        ["direct_video", "frame_based", "hierarchical", "all"],
        default=["all"]
    )
    
    # Strategy filter
    strategy_filter = st.multiselect(
        "Ranking strategies",
        ["rrf", "weighted", "max_score", "all"],
        default=["all"]
    )
    
    st.markdown("---")
    
    # Refresh settings
    st.subheader("Refresh Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5
        )
    
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("Advanced Options"):
        show_raw_data = st.checkbox("Show raw data tables", value=False)
        enable_rca = st.checkbox("Enable Root Cause Analysis", value=True)
        export_format = st.selectbox(
            "Export format",
            ["JSON", "CSV", "HTML Report"]
        )

# Main content area
st.title("Phoenix Dashboard")

# Last refresh time
st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Create main tabs
main_tabs = st.tabs(["📊 Analytics", "🧪 Evaluation", "🗺️ Embedding Atlas"])

# Analytics Tab
with main_tabs[0]:
    # Fetch traces
    with st.spinner("Fetching traces..."):
        traces = st.session_state.analytics.get_traces(
            start_time=start_datetime,
            end_time=end_datetime,
            operation_filter=operation_filter if operation_filter else None,
            limit=10000
        )

    if not traces:
        st.warning("No traces found for the selected time range and filters.")
        st.stop()

    # Convert to DataFrame for easier manipulation
    traces_df = pd.DataFrame([{
        'trace_id': t.trace_id,
        'timestamp': t.timestamp,
        'duration_ms': t.duration_ms,
        'operation': t.operation,
        'status': t.status,
        'profile': t.profile,
        'strategy': t.strategy,
        'error': t.error
    } for t in traces])

    # Apply profile and strategy filters
    if "all" not in profile_filter:
        traces_df = traces_df[traces_df['profile'].isin(profile_filter)]

    if "all" not in strategy_filter:
        traces_df = traces_df[traces_df['strategy'].isin(strategy_filter)]

    # Calculate statistics with operation grouping
    if not traces_df.empty:
        stats = st.session_state.analytics.calculate_statistics(
            [analytics_module.TraceMetrics(**row) for _, row in traces_df.iterrows()],
            group_by="operation"
        )
    else:
        # Default stats when no data
        stats = {
            'total_requests': 0,
            'status': {'success_rate': 0, 'error_rate': 0},
            'response_time': {
                'mean': 0, 'median': 0, 'p95': 0, 'p99': 0,
                'min': 0, 'max': 0, 'std': 0
            },
            'by_operation': {}
        }

    # Create sub-tabs for analytics
    tabs = st.tabs([
        "📊 Overview", 
        "📈 Time Series", 
        "📊 Distributions", 
        "🗺️ Heatmaps", 
        "🎯 Outliers", 
        "🔍 Trace Explorer"
    ] + (["🔬 Root Cause Analysis"] if enable_rca else []))

# Tab 1: Overview
with tabs[0]:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{stats['total_requests']:,}",
            delta=None
        )
    
    with col2:
        success_rate = stats.get('status', {}).get('success_rate', 0) * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{success_rate - 95:.1f}%" if success_rate < 95 else None
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            f"{stats.get('response_time', {}).get('mean', 0):.1f} ms",
            delta=None
        )
    
    with col4:
        st.metric(
            "P95 Response Time",
            f"{stats.get('response_time', {}).get('p95', 0):.1f} ms",
            delta=None
        )
    
    st.markdown("---")
    
    # Summary statistics
    st.markdown("### 📊 Detailed Analytics")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("⏱️ Response Time Statistics")
        rt = stats.get('response_time', {})
        
        # Create a box plot for response time distribution
        percentiles = [
            ('Min', rt.get('min', 0)),
            ('P25', rt.get('p25', rt.get('min', 0) * 1.5)),  # Estimate if not available
            ('P50', rt.get('median', 0)),
            ('P75', rt.get('p75', rt.get('p90', 0) * 0.8)),  # Estimate if not available
            ('P90', rt.get('p90', 0)),
            ('P95', rt.get('p95', 0)),
            ('P99', rt.get('p99', 0)),
            ('Max', rt.get('max', 0))
        ]
        
        # Create a simpler bar chart for percentiles
        percentile_data = [
            {'Metric': 'Min', 'Value': rt.get('min', 0), 'Color': '#3498db'},
            {'Metric': 'P50', 'Value': rt.get('median', 0), 'Color': '#2ecc71'},
            {'Metric': 'P90', 'Value': rt.get('p90', 0), 'Color': '#f39c12'},
            {'Metric': 'P95', 'Value': rt.get('p95', 0), 'Color': '#e67e22'},
            {'Metric': 'P99', 'Value': rt.get('p99', 0), 'Color': '#e74c3c'},
            {'Metric': 'Max', 'Value': rt.get('max', 0), 'Color': '#c0392b'}
        ]
        
        fig_percentiles = go.Figure()
        
        for item in percentile_data:
            fig_percentiles.add_trace(go.Bar(
                x=[item['Value']],
                y=[item['Metric']],
                orientation='h',
                marker_color=item['Color'],
                name=item['Metric'],
                text=f"{item['Value']:.1f} ms",
                textposition='outside',
                showlegend=False,
                hovertemplate=f"<b>{item['Metric']}</b><br>Value: {item['Value']:.1f} ms<extra></extra>"
            ))
        
        fig_percentiles.update_layout(
            title="Response Time Percentiles",
            xaxis_title="Time (ms)",
            yaxis_title="",
            height=250,
            showlegend=False,
            xaxis=dict(
                type='log' if rt.get('max', 0) > 10 * rt.get('median', 1) else 'linear',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                categoryorder='array',
                categoryarray=['Max', 'P99', 'P95', 'P90', 'P50', 'Min']
            ),
            margin=dict(l=60, r=20, t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.3
        )
        
        st.plotly_chart(fig_percentiles, use_container_width=True, key="response_time_chart")
        
        # Show key metrics in a more compact format
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Median (P50)", f"{rt.get('median', 0):.1f} ms")
            st.metric("P95", f"{rt.get('p95', 0):.1f} ms")
        with metrics_col2:
            st.metric("P90", f"{rt.get('p90', 0):.1f} ms")
            st.metric("P99", f"{rt.get('p99', 0):.1f} ms")
    
    with col2:
        st.subheader("📋 Request Distribution by Operation")
        
        # Debug: Check what's in stats
        # st.write("Debug - Stats keys:", list(stats.keys()))
        
        if 'by_operation' in stats and stats['by_operation']:
            op_df = pd.DataFrame([
                {'Operation': op, 'Count': data['count'], 'Percentage': (data['count'] / stats['total_requests']) * 100}
                for op, data in stats['by_operation'].items()
            ])
            # Sort by count descending
            op_df = op_df.sort_values('Count', ascending=False)
            
            # Create a donut chart instead of pie for better visibility
            fig_ops = px.pie(op_df, values='Count', names='Operation', 
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3)
            
            # Update layout for better appearance
            fig_ops.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig_ops.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Operations Breakdown", x=0.5, xanchor='center')
            )
            
            st.plotly_chart(fig_ops, use_container_width=True)
            
            # Add a small table below showing the exact counts
            st.markdown("##### Operation Details")
            for _, row in op_df.iterrows():
                col_op, col_count = st.columns([3, 1])
                with col_op:
                    st.text(row['Operation'])
                with col_count:
                    st.text(f"{row['Count']} ({row['Percentage']:.1f}%)")
        else:
            # Show operation counts from traces directly if by_operation is not available
            operation_counts = traces_df['operation'].value_counts()
            if not operation_counts.empty:
                op_df = pd.DataFrame({
                    'Operation': operation_counts.index,
                    'Count': operation_counts.values,
                    'Percentage': (operation_counts.values / len(traces_df)) * 100
                })
                
                # Create the donut chart
                fig_ops = px.pie(op_df, values='Count', names='Operation', 
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig_ops.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
                
                fig_ops.update_layout(
                    showlegend=True,
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(text="Operations Breakdown", x=0.5, xanchor='center')
                )
                
                st.plotly_chart(fig_ops, use_container_width=True)
                
                # Show details
                st.markdown("##### Operation Details")
                for _, row in op_df.iterrows():
                    col_op, col_count = st.columns([3, 1])
                    with col_op:
                        st.text(row['Operation'])
                    with col_count:
                        st.text(f"{row['Count']} ({row['Percentage']:.1f}%)")
            else:
                st.info("No operation data available")

# Tab 2: Time Series
with tabs[1]:
    st.subheader("Request Volume and Response Time Over Time")
    
    # Time window selection
    time_window = st.select_slider(
        "Aggregation window",
        options=["1min", "5min", "15min", "1h"],
        value="5min"
    )
    
    # Create time series plot
    fig_ts = st.session_state.analytics.create_time_series_plot(
        traces,
        metric="duration_ms",
        aggregation="mean",
        time_window=time_window
    )
    
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Request volume over time
    fig_volume = st.session_state.analytics.create_time_series_plot(
        traces,
        metric="count",
        aggregation="count",
        time_window=time_window
    )
    
    if fig_volume:
        st.plotly_chart(fig_volume, use_container_width=True)

# Tab 3: Distributions
with tabs[2]:
    st.subheader("Response Time Distributions")
    
    # Add explanation of plots
    with st.expander("📊 Understanding the Distribution Plots"):
        st.markdown("""
        ### Distribution Plot (Top Left)
        A histogram showing the frequency of different response times. Helps identify:
        - Most common response times (peaks)
        - Spread of response times
        - Multiple modes (if the system has different performance characteristics)
        
        ### Box Plot (Top Right)
        Shows the five-number summary: minimum, Q1 (25th percentile), median, Q3 (75th percentile), and maximum.
        - Box represents the interquartile range (IQR) where 50% of data lies
        - Whiskers extend to 1.5×IQR
        - Points beyond whiskers are outliers
        
        ### Violin Plot (Bottom Left)
        Shows the density distribution of response times:
        - **Width of violin**: Represents how many data points have that value
        - **Wider sections**: More common response times
        - **Narrower sections**: Less common response times
        - **Small horizontal line/box**: May appear near the widest part (this is Plotly's minimal box plot indicator for median/quartiles, often hard to see)
        - **Dashed line**: Mean value (if visible)
        - **Individual points**: Outliers beyond the typical range
        - **Overall shape**: 
          - Symmetric = balanced performance
          - Skewed right = some slow outliers
          - Multiple bulges = different performance modes
        - Useful for comparing shapes between groups (e.g., seeing if one profile has more consistent performance)
        
        ### ECDF - Empirical Cumulative Distribution Function (Bottom Right)
        Shows what percentage of requests are faster than a given response time:
        - X-axis: Response time
        - Y-axis: Percentage of requests (0-100%)
        - To read: "Y% of requests complete in X ms or less"
        - Vertical lines show key percentiles (P50, P90, P95, P99)
        - Steep curve = consistent performance
        - Gradual curve = high variability
        - Example: If P95 line is at 500ms, then 95% of requests complete in 500ms or less
        """)
    
    # Distribution plot
    group_by = st.selectbox(
        "Group by",
        ["None", "operation", "profile", "strategy", "status"]
    )
    
    fig_dist = st.session_state.analytics.create_distribution_plot(
        traces,
        metric="duration_ms",
        group_by=group_by if group_by != "None" else None
    )
    
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)

# Tab 4: Heatmaps
with tabs[3]:
    st.subheader("Performance Heatmaps")
    
    # Info about profile and strategy
    with st.expander("ℹ️ About Profile and Strategy fields"):
        st.markdown("""
        **Profile**: Represents the video processing profile (e.g., 'direct_video_global', 'frame_based_colpali'). 
        This field is only populated for search operations that have an associated profile.
        
        **Strategy**: Represents the ranking strategy used in search operations (e.g., 'default', 'bm25', 'hybrid').
        This field is only populated for search operations where a specific ranking strategy is used.
        
        If these fields appear empty in the heatmap, it means:
        - The traced operations don't have these attributes (e.g., non-search operations)
        - The attributes weren't captured during tracing
        - All values are 'unknown' or similar default values
        """)
    
    # Heatmap configuration
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(
            "X-axis", 
            ["hour", "day_of_week", "operation"],
            help="Select the primary dimension for the heatmap"
        )
    with col2:
        y_axis = st.selectbox(
            "Y-axis", 
            ["operation", "profile", "strategy", "status", "day"],
            help="Select the grouping dimension. Profile and strategy may be empty if not all operations have these attributes"
        )
    
    if x_axis != y_axis:
        fig_heatmap = st.session_state.analytics.create_heatmap(
            traces,
            x_field=x_axis,
            y_field=y_axis,
            metric="duration_ms"
        )
        
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Please select different values for X and Y axes.")

# Tab 5: Outliers
with tabs[4]:
    st.subheader("Response Time Outliers")
    
    # Explain outlier detection method
    with st.expander("ℹ️ How Outliers are Detected", expanded=False):
        st.write("""
        **IQR (Interquartile Range) Method - Tukey's Rule:**
        - Q1 = 25th percentile (25% of values are below this)
        - Q3 = 75th percentile (75% of values are below this)
        - IQR = Q3 - Q1 (the range containing the middle 50% of data)
        - **Upper Bound = Q3 + 1.5 × IQR**
        - **Lower Bound = Q1 - 1.5 × IQR**
        
        Any value outside these bounds is considered an outlier. The 1.5 multiplier is a 
        standard choice that typically identifies ~2-3% of normally distributed data as outliers.
        
        **Why IQR instead of standard deviation?**
        - More robust to extreme values
        - Works well with skewed distributions (common in response times)
        - Doesn't assume normal distribution
        """)
    
    # Outlier detection
    outlier_metric = st.selectbox(
        "Metric for outlier detection",
        ["duration_ms", "error_rate"]
    )
    
    fig_outliers = st.session_state.analytics.create_outlier_plot(
        traces,
        metric=outlier_metric
    )
    
    if fig_outliers:
        st.plotly_chart(fig_outliers, use_container_width=True)
        
        # Show outlier details
        if outlier_metric == "duration_ms":
            # Calculate IQR-based outlier threshold to match the plot
            Q1 = traces_df['duration_ms'].quantile(0.25)
            Q3 = traces_df['duration_ms'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            
            # Get outliers using the same method as the plot
            outliers_df = traces_df[(traces_df['duration_ms'] > upper_bound) | (traces_df['duration_ms'] < lower_bound)]
            
            if not outliers_df.empty:
                st.subheader(f"Outlier Details")
                st.caption(f"Showing traces outside bounds: [{lower_bound:.1f}, {upper_bound:.1f}] ms")
                st.dataframe(
                    outliers_df[['timestamp', 'operation', 'duration_ms', 'profile', 'strategy', 'error']]
                    .sort_values('duration_ms', ascending=False)
                    .head(20)
                )

# Tab 6: Trace Explorer
with tabs[5]:
    st.subheader("Individual Trace Explorer")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        trace_search = st.text_input(
            "Search by trace ID or operation",
            placeholder="Enter trace ID or operation name..."
        )
    with col2:
        search_type = st.selectbox("Search in", ["All", "Trace ID", "Operation"])
    
    # Filter traces based on search
    if trace_search:
        if search_type == "Trace ID":
            filtered_df = traces_df[traces_df['trace_id'].str.contains(trace_search, case=False)]
        elif search_type == "Operation":
            filtered_df = traces_df[traces_df['operation'].str.contains(trace_search, case=False)]
        else:
            filtered_df = traces_df[
                traces_df['trace_id'].str.contains(trace_search, case=False) |
                traces_df['operation'].str.contains(trace_search, case=False)
            ]
    else:
        filtered_df = traces_df
    
    # Sort options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["timestamp", "duration_ms", "operation", "status"],
            index=0
        )
    with col2:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(
        sort_by, 
        ascending=(sort_order == "Ascending")
    )
    
    # Display traces
    st.write(f"Showing {len(filtered_df)} traces")
    
    # Pagination
    traces_per_page = 20
    num_pages = max(1, (len(filtered_df) - 1) // traces_per_page + 1)
    page = st.selectbox("Page", range(1, num_pages + 1))
    
    start_idx = (page - 1) * traces_per_page
    end_idx = min(start_idx + traces_per_page, len(filtered_df))
    
    # Show traces
    for idx in range(start_idx, end_idx):
        row = filtered_df.iloc[idx]
        
        with st.expander(
            f"{row['operation']} - {row['timestamp'].strftime('%H:%M:%S')} "
            f"({row['duration_ms']:.1f} ms) - {row['status']}"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Trace ID:** `{row['trace_id']}`")
                st.write(f"**Operation:** {row['operation']}")
                st.write(f"**Status:** {row['status']}")
            
            with col2:
                st.write(f"**Duration:** {row['duration_ms']:.2f} ms")
                st.write(f"**Profile:** {row['profile']}")
                st.write(f"**Strategy:** {row['strategy']}")
            
            with col3:
                st.write(f"**Timestamp:** {row['timestamp']}")
                if row['error']:
                    st.write(f"**Error:** {row['error']}")

# Helper function to create Phoenix trace link
def create_phoenix_link(trace_id, text="View in Phoenix"):
    phoenix_base_url = "http://localhost:6006"
    # Phoenix uses project-based routing with base64 encoded project IDs
    # Default project is "Project:1" which encodes to "UHJvamVjdDox"
    import base64
    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
    # Phoenix uses client-side routing, so we'll link to the traces page
    # Include the trace ID query format
    return f"[{text}]({phoenix_base_url}/projects/{project_encoded}/traces)"


# Tab 7: Root Cause Analysis (if enabled)
if enable_rca and len(tabs) > 6:
    with tabs[6]:
        st.subheader("Root Cause Analysis")
        
        # Phoenix Query Reference
        with st.expander("📚 Phoenix Query Reference", expanded=False):
            st.write("**Common Phoenix Query Patterns:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Status Queries:**")
                st.code('status_code == "ERROR"', language="python")
                st.code('status_code == "OK"', language="python")
                
                st.write("**Latency Queries:**")
                st.code('latency_ms > 1000', language="python")
                st.code('latency_ms < 100', language="python")
            
            with col2:
                st.write("**Time Range Queries:**")
                st.code('timestamp >= "2025-01-01T00:00:00Z"', language="python")
                st.code('timestamp >= "2025-01-01" and timestamp <= "2025-01-02"', language="python")
                
                st.write("**Trace ID Queries:**")
                st.code('trace_id == "abc123"', language="python")
                st.code('trace_id == "id1" or trace_id == "id2"', language="python")
            
            st.caption("💡 You can combine queries with 'and' / 'or' operators")
        
        # Initialize RCA
        rca = RootCauseAnalyzer()
        
        # RCA configuration
        col1, col2 = st.columns(2)
        with col1:
            include_performance = st.checkbox(
                "Include performance degradations",
                value=True,
                help="Analyze slow requests in addition to failures"
            )
        with col2:
            performance_threshold = st.slider(
                "Performance threshold (percentile)",
                min_value=90,
                max_value=99,
                value=95,
                help="Requests slower than this percentile of all requests are considered performance degradations. For example, P95 means requests slower than 95% of all requests."
            )
        
        # Show current threshold value
        if include_performance and not traces_df.empty:
            # Calculate P95 for successful requests only (matching what RCA does)
            successful_df = traces_df[traces_df['status'] == 'success']
            if not successful_df.empty:
                successful_p95 = successful_df['duration_ms'].quantile(performance_threshold / 100)
                all_p95 = traces_df['duration_ms'].quantile(performance_threshold / 100)
                
                st.caption(f"💡 P{performance_threshold} of successful requests: {successful_p95:.1f}ms - requests slower than this are flagged")
                if abs(successful_p95 - all_p95) > 100:  # Significant difference
                    st.info(f"ℹ️ Note: P{performance_threshold} of all requests (including failures) is {all_p95:.1f}ms")
        
        # Run analysis - use filtered traces to match the stats
        with st.spinner("Analyzing failures and performance issues..."):
            # Convert filtered DataFrame back to TraceMetrics objects
            filtered_traces = [analytics_module.TraceMetrics(**row) for _, row in traces_df.iterrows()]
            
            rca_results = rca.analyze_failures(
                filtered_traces,  # Use filtered traces instead of all traces
                include_performance=include_performance,
                performance_threshold_percentile=performance_threshold
            )
        
        if rca_results and 'summary' in rca_results:
            summary = rca_results['summary']
            total_issues = summary.get('failed_traces', 0) + summary.get('performance_degraded', 0)
            
            # Debug info to understand the discrepancy
            with st.expander("🔍 Analysis Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Request Breakdown:**")
                    st.write(f"Total requests analyzed: {summary.get('total_traces', 0)}")
                    st.write(f"Failed requests: {summary.get('failed_traces', 0)}")
                    st.write(f"Successful requests flagged as slow: {summary.get('performance_degraded', 0)}")
                
                with col2:
                    if 'performance_analysis' in rca_results and 'threshold' in rca_results['performance_analysis']:
                        st.write("**Performance Thresholds:**")
                        st.write(f"P{performance_threshold} of successful requests: {rca_results['performance_analysis']['threshold']:.1f}ms")
                        st.write(f"P{performance_threshold} of all requests: {stats.get('response_time', {}).get('p95', 0):.1f}ms")
                        st.write("*Performance analysis excludes failed requests when calculating thresholds*")
            
            # Check if there are any issues to analyze
            if total_issues == 0:
                st.info("🎉 Great news! No failures or performance issues detected in the selected time range.")
                st.markdown("The system appears to be running smoothly with:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Traces Analyzed", summary.get('total_traces', 0))
                with col2:
                    st.metric("Failure Rate", "0%")
            else:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Issues",
                        total_issues
                    )
                
                with col2:
                    st.metric(
                        "Failed Traces",
                        summary.get('failed_traces', 0)
                    )
                
                with col3:
                    failure_rate = summary.get('failure_rate', 0) * 100
                    st.metric(
                        "Failure Rate",
                        f"{failure_rate:.1f}%"
                    )
            
            st.markdown("---")
            
            # Root causes and recommendations
            if 'root_causes' in rca_results and rca_results['root_causes']:
                st.subheader("Root Cause Hypotheses")
                
                for i, hypothesis in enumerate(rca_results['root_causes'][:5], 1):
                    # Handle both dict and object types
                    if hasattr(hypothesis, 'hypothesis'):
                        # It's a RootCauseHypothesis object
                        with st.expander(
                            f"{i}. {hypothesis.hypothesis} "
                            f"(Confidence: {hypothesis.confidence:.0%})"
                        ):
                            if hypothesis.evidence:
                                st.write("**Evidence:**")
                                for evidence in hypothesis.evidence:
                                    # Format time ranges in evidence
                                    if "Time range:" in evidence:
                                        # Extract and format time range
                                        parts = evidence.split("Time range:", 1)
                                        if len(parts) == 2:
                                            time_part = parts[1].strip()
                                            # Look for ISO timestamps
                                            import re
                                            iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)'
                                            timestamps = re.findall(iso_pattern, time_part)
                                            if len(timestamps) >= 2:
                                                formatted_range = format_time_range(timestamps[0], timestamps[1])
                                                evidence = f"{parts[0]}Time range: {formatted_range}"
                                    st.write(f"- {evidence}")
                            
                            if hypothesis.affected_traces:
                                st.write(f"**Affected traces:** {len(hypothesis.affected_traces)}")
                                # Show trace links
                                if len(hypothesis.affected_traces) > 0:
                                    trace_links = []
                                    for trace_id in hypothesis.affected_traces[:5]:  # Show up to 5
                                        trace_links.append(f"{trace_id[:8]}...")
                                    st.write("Sample trace IDs: " + ", ".join(trace_links))
                                    
                                    # Show Phoenix query for trace IDs
                                    trace_ids_query = " or ".join([f'trace_id == "{tid}"' for tid in hypothesis.affected_traces[:5]])
                                    st.code(trace_ids_query, language="python")
                                    
                                    phoenix_base_url = "http://localhost:6006"
                                    import base64
                                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                                    phoenix_link = f"{phoenix_base_url}/projects/{project_encoded}/traces"
                                    st.caption(f"☝️ Copy this query to find these specific traces in [Phoenix]({phoenix_link})")
                            
                            if hypothesis.suggested_action:
                                st.write(f"**Suggested Action:** {hypothesis.suggested_action}")
                            
                            if hypothesis.category:
                                st.write(f"**Category:** {hypothesis.category}")
                    else:
                        # It's a dictionary
                        with st.expander(
                            f"{i}. {hypothesis.get('hypothesis', 'Unknown')} "
                            f"(Confidence: {hypothesis.get('confidence', 0):.0%})"
                        ):
                            if 'evidence' in hypothesis:
                                st.write("**Evidence:**")
                                for evidence in hypothesis['evidence']:
                                    # Format time ranges in evidence
                                    if "Time range:" in evidence:
                                        # Extract and format time range
                                        parts = evidence.split("Time range:", 1)
                                        if len(parts) == 2:
                                            time_part = parts[1].strip()
                                            # Look for ISO timestamps
                                            import re
                                            iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)'
                                            timestamps = re.findall(iso_pattern, time_part)
                                            if len(timestamps) >= 2:
                                                formatted_range = format_time_range(timestamps[0], timestamps[1])
                                                evidence = f"{parts[0]}Time range: {formatted_range}"
                                    st.write(f"- {evidence}")
                            
                            if 'affected_traces' in hypothesis:
                                st.write(f"**Affected traces:** {len(hypothesis['affected_traces'])}")
                            
                            if 'suggested_action' in hypothesis:
                                st.write(f"**Suggested Action:** {hypothesis['suggested_action']}")
            
            # Show recommendations
            if 'recommendations' in rca_results and rca_results['recommendations']:
                st.subheader("📋 Recommendations")
                
                for i, rec in enumerate(rca_results['recommendations'][:5], 1):
                    if isinstance(rec, dict):
                        # Handle structured recommendation format
                        priority = rec.get('priority', 'medium')
                        category = rec.get('category', 'general')
                        recommendation = rec.get('recommendation', 'Unknown recommendation')
                        details = rec.get('details', [])
                        affected = rec.get('affected_components', [])
                        
                        # Set priority color
                        priority_colors = {
                            'high': '🔴',
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        priority_icon = priority_colors.get(priority, '⚪')
                        
                        with st.expander(f"{priority_icon} {i}. {recommendation} ({category.title()})"):
                            st.write(f"**Priority:** {priority.upper()}")
                            
                            if details:
                                st.write("**Action Items:**")
                                for detail in details:
                                    st.write(f"• {detail}")
                            
                            if affected:
                                st.write("\n**Affected Components:**")
                                for component in affected:
                                    st.write(f"• {component}")
                    else:
                        # Handle simple string recommendations
                        st.write(f"{i}. {rec}")
            
            # Failure analysis
            if 'failure_analysis' in rca_results and rca_results['failure_analysis']:
                st.markdown("---")
                st.subheader("🔍 Detailed Analysis")
                st.markdown("##### Failure Analysis")
                
                failure_data = rca_results['failure_analysis']
                
                # Add link to view all failed traces
                if summary.get('failed_traces', 0) > 0:
                    phoenix_base_url = "http://localhost:6006"
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    st.markdown(f"[📊 View all {summary.get('failed_traces', 0)} failed traces in Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces)")
                    
                    # Show exact Phoenix query
                    phoenix_query = 'status_code == "ERROR"'
                    st.code(phoenix_query, language="python")
                    st.caption(f"☝️ Copy this query and paste it in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
                
                # Show error types
                if 'error_types' in failure_data and failure_data['error_types']:
                    st.write("**Error Types Distribution:**")
                    
                    # Create DataFrame for error types
                    error_data = []
                    for error_type, count in failure_data['error_types'].most_common():
                        formatted_type = error_type.replace('_', ' ').title()
                        error_data.append({
                            "Error Type": formatted_type,
                            "Count": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    error_df = pd.DataFrame(error_data)
                    st.dataframe(error_df, hide_index=True, use_container_width=True)
                
                # Show failed operations
                if 'failed_operations' in failure_data and failure_data['failed_operations']:
                    st.markdown("---")
                    st.write("**Failed Operations:**")
                    
                    # Create DataFrame for failed operations
                    ops_data = []
                    for op, count in failure_data['failed_operations'].most_common():
                        ops_data.append({
                            "Operation": op,
                            "Failures": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    ops_df = pd.DataFrame(ops_data)
                    st.dataframe(ops_df, hide_index=True, use_container_width=True)
                
                # Show failed profiles
                if 'failed_profiles' in failure_data and failure_data['failed_profiles']:
                    st.markdown("---")
                    st.write("**Failed Profiles:**")
                    
                    # Create DataFrame for failed profiles
                    profile_data = []
                    for profile, count in failure_data['failed_profiles'].most_common():
                        profile_data.append({
                            "Profile": profile,
                            "Failures": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    profile_df = pd.DataFrame(profile_data)
                    st.dataframe(profile_df, hide_index=True, use_container_width=True)
                
                # Show temporal patterns if any
                if 'temporal_patterns' in failure_data and failure_data['temporal_patterns']:
                    st.markdown("---")
                    st.write("**Temporal Patterns (Failure Bursts):**")
                    
                    # Create DataFrame for temporal patterns
                    burst_data = []
                    phoenix_base_url = "http://localhost:6006"
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    
                    for i, pattern in enumerate(failure_data['temporal_patterns']):
                        if pattern['type'] == 'burst':
                            burst_data.append({
                                "Burst #": i + 1,
                                "Failures": pattern['failure_count'],
                                "Duration": f"{pattern['duration_minutes']:.1f} min",
                                "Start Time": format_timestamp(pattern['start_time']),
                                "End Time": format_timestamp(pattern.get('end_time', pattern['start_time'])),
                                "Phoenix Link": f"{phoenix_base_url}/projects/{project_encoded}/traces"
                            })
                    
                    if burst_data:
                        burst_df = pd.DataFrame(burst_data)
                        st.dataframe(
                            burst_df, 
                            hide_index=True, 
                            use_container_width=True,
                            column_config={
                                "Phoenix Link": st.column_config.LinkColumn(
                                    "View in Phoenix",
                                    display_text="Open Phoenix"
                                )
                            }
                        )
                        
                        # Show Phoenix queries for each burst
                        with st.expander("📋 Phoenix Queries for Time Ranges"):
                            for i, pattern in enumerate(failure_data['temporal_patterns']):
                                if pattern['type'] == 'burst':
                                    st.write(f"**Burst {i+1}:**")
                                    # Format timestamps for Phoenix query
                                    start_iso = pattern['start_time']
                                    end_iso = pattern.get('end_time', pattern['start_time'])
                                    phoenix_time_query = f'timestamp >= "{start_iso}" and timestamp <= "{end_iso}"'
                                    st.code(phoenix_time_query, language="python")
                            phoenix_base_url = "http://localhost:6006"
                            import base64
                            project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                            st.caption(f"☝️ Copy these queries and paste them in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
            
            # Performance analysis
            if 'performance_analysis' in rca_results and rca_results['performance_analysis']:
                st.subheader("📊 Performance Analysis")
                perf = rca_results['performance_analysis']
                
                # Add link to view slow traces
                if summary.get('performance_degraded', 0) > 0 and 'threshold' in perf:
                    phoenix_base_url = "http://localhost:6006"
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    st.markdown(f"[📊 View {summary.get('performance_degraded', 0)} slow traces in Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces)")
                    
                    # Show exact Phoenix query for slow traces
                    phoenix_duration_query = f'latency_ms > {perf["threshold"]:.0f}'
                    st.code(phoenix_duration_query, language="python")
                    st.caption(f"☝️ Copy this query and paste it in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
                
                # Show threshold information
                if include_performance:
                    st.info(f"ℹ️ Performance threshold: P{performance_threshold} (requests slower than this percentile are flagged)")
                
                if 'slow_operations' in perf and perf['slow_operations']:
                    st.write("**Operations Exceeding Performance Threshold:**")
                    
                    # Show threshold info if available
                    threshold_info = ""
                    percentile = 95  # Default
                    if 'threshold' in perf:
                        threshold = perf['threshold']
                        percentile = perf.get('threshold_percentile', 95)
                        threshold_info = f" (Threshold: P{percentile} = {threshold:.1f}ms)"
                    
                    # Try to get baseline stats from main statistics
                    baseline_info = ""
                    if 'response_time' in stats:
                        p95 = stats['response_time'].get('p95', 0)
                        median = stats['response_time'].get('median', 0)
                        baseline_info = f" | Current view stats - Median: {median:.1f}ms, P95: {p95:.1f}ms"
                    
                    # Show explanation if there's a discrepancy
                    if 'threshold' in perf and 'response_time' in stats:
                        current_p95 = stats['response_time'].get('p95', 0)
                        if abs(perf['threshold'] - current_p95) > 100:  # Significant difference
                            st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}")
                            st.info(f"ℹ️ Note: The threshold ({perf['threshold']:.1f}ms) is based on successful requests only. All requests (including failures) have Median: {stats['response_time'].get('median', 0):.1f}ms, P95: {current_p95:.1f}ms")
                        else:
                            st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}{baseline_info}")
                    else:
                        st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}{baseline_info}")
                    
                    for op, op_stats in perf['slow_operations'].items():
                        if isinstance(op_stats, dict):
                            avg_duration = op_stats.get('avg_duration', 0)
                            count = op_stats.get('count', 0)
                            min_duration = op_stats.get('min_duration', 0)
                            max_duration = op_stats.get('max_duration', 0)
                            
                            col1, col2, col3 = st.columns([3, 2, 2])
                            with col1:
                                st.write(f"**{op}**")
                            with col2:
                                if min_duration == max_duration:
                                    st.write(f"{min_duration:.1f}ms ({count} calls)")
                                else:
                                    st.write(f"Range: {min_duration:.1f}-{max_duration:.1f}ms")
                            with col3:
                                st.write(f"Avg: {avg_duration:.1f}ms")
                            
                            # Show sample durations if available
                            if 'durations' in op_stats and op_stats['durations']:
                                sample = op_stats['durations'][:3]
                                st.caption(f"   Sample durations: {', '.join(f'{d:.1f}ms' for d in sample)}")
                        elif isinstance(op_stats, (int, float)):
                            # This is a count from Counter, not a duration
                            st.write(f"- **{op}**: {int(op_stats)} occurrences")
                        else:
                            st.write(f"- **{op}**: {op_stats}")
                
                # Show performance degradation patterns if available
                if 'degradation_patterns' in perf:
                    st.write("\n**Performance Degradation Patterns:**")
                    for pattern, details in perf['degradation_patterns'].items():
                        st.write(f"- {pattern}: {details}")
        else:
            st.info("No data available for root cause analysis. Ensure there are traces in the selected time range.")

# Export functionality
if show_raw_data:
    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(traces_df)

# Evaluation Tab
with main_tabs[1]:
    # Import and use the tabbed evaluation tab (like HTML report)
    from phoenix_dashboard_evaluation_tab_tabbed import render_evaluation_tab
    render_evaluation_tab()

# Embedding Atlas Tab
with main_tabs[2]:
    st.header("🗺️ Embedding Visualization")
    
    # Import the embedding visualization module
    from embedding_atlas_tab import render_embedding_atlas_tab
    render_embedding_atlas_tab()

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🔥 Phoenix Dashboard - Cogniverse Evaluation Framework")