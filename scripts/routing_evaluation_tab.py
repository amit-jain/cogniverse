#!/usr/bin/env python3
"""
Routing Evaluation Tab for Phoenix Dashboard

Displays routing-specific metrics from RoutingEvaluator including:
- Routing accuracy and confidence calibration
- Per-agent performance metrics
- Temporal analysis of routing decisions
- Confidence distribution and analysis
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px_express
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogniverse_agents.routing.annotation_agent import (
    AnnotationAgent,
    AnnotationPriority,
)
from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    LLMAutoAnnotator,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator
from cogniverse_foundation.telemetry.config import SERVICE_NAME_ORCHESTRATION

logger = logging.getLogger(__name__)


def run_async_in_streamlit(coro):
    """Helper to run async operations in Streamlit"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)


def render_routing_evaluation_tab():
    """Render the routing evaluation tab with metrics and visualizations"""
    st.subheader("🎯 Routing Evaluation Dashboard")

    # Configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tenant_id = st.text_input("Tenant ID", value="default", help="Tenant to analyze")
        with col2:
            lookback_hours = st.number_input(
                "Lookback Period (hours)", min_value=1, max_value=168, value=24
            )

        # Project name - using unified orchestration project
        project_name = f"cogniverse-{tenant_id}-{SERVICE_NAME_ORCHESTRATION}"
        st.info(f"📊 Querying spans from project: `{project_name}`")

    # Initialize evaluator with telemetry provider
    try:
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        telemetry_manager = get_telemetry_manager()
        provider = telemetry_manager.get_provider(tenant_id=tenant_id)
        evaluator = RoutingEvaluator(provider=provider, project_name=project_name)
    except Exception as e:
        st.error(f"❌ Failed to initialize RoutingEvaluator: {e}")
        return

    # Check telemetry provider connectivity
    try:
        # Test provider connectivity by attempting to fetch spans
        async def check_provider():
            try:
                await provider.traces.get_spans(
                    project=project_name,
                    start_time=datetime.now() - timedelta(minutes=1),
                    end_time=datetime.now(),
                    limit=1,
                )
                return True
            except Exception:
                return False

        provider_available = run_async_in_streamlit(check_provider())
    except Exception:
        provider_available = False

    if not provider_available:
        st.warning("⚠️ Telemetry provider is not available")
        st.info("Check your telemetry configuration and ensure the provider is running")
        return

    # Time range for query
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)

    # Fetch and evaluate routing spans
    with st.spinner("Fetching routing spans from telemetry..."):
        try:
            # Query routing spans from telemetry (async method)
            routing_spans = run_async_in_streamlit(
                evaluator.query_routing_spans(
                    start_time=start_time, end_time=end_time, limit=1000
                )
            )

            # Calculate metrics from spans
            if not routing_spans:
                st.warning(
                    f"📭 No routing decisions found in the last {lookback_hours} hours. "
                    "Make sure the routing agent has been processing requests and telemetry is capturing traces."
                )
                return

            metrics = evaluator.calculate_metrics(routing_spans)
        except Exception as e:
            st.error(f"❌ Failed to evaluate routing decisions: {e}")
            st.exception(e)
            return

    # Check if we have data
    if metrics.total_decisions == 0:
        st.warning(
            f"📭 No routing decisions found in the last {lookback_hours} hours. "
            "Generate some routing spans by making requests to the routing agent."
        )
        return

    # Display metrics
    _render_summary_metrics(metrics)
    _render_per_agent_metrics(metrics)
    _render_confidence_analysis(evaluator, start_time, end_time)
    _render_temporal_analysis(evaluator, start_time, end_time)

    # Annotation section
    st.divider()
    _render_annotation_section(tenant_id, project_name, lookback_hours)


def _render_summary_metrics(metrics):
    """Render summary metrics cards"""
    st.subheader("📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Routing Accuracy",
            value=f"{metrics.routing_accuracy:.1%}",
            help="Percentage of routing decisions that led to successful outcomes",
        )

    with col2:
        st.metric(
            label="Confidence Calibration",
            value=f"{metrics.confidence_calibration:.3f}",
            help="Correlation between routing confidence and actual success rate",
        )

    with col3:
        st.metric(
            label="Avg Routing Latency",
            value=f"{metrics.avg_routing_latency:.0f}ms",
            help="Average time to make a routing decision",
        )

    with col4:
        st.metric(
            label="Total Decisions",
            value=f"{metrics.total_decisions}",
            delta=f"{metrics.ambiguous_count} ambiguous" if metrics.ambiguous_count > 0 else None,
            help="Total routing decisions evaluated",
        )


def _render_per_agent_metrics(metrics):
    """Render per-agent performance metrics"""
    st.subheader("🤖 Per-Agent Performance")

    # Create DataFrame for per-agent metrics
    agent_data = []
    for agent in metrics.per_agent_precision.keys():
        agent_data.append(
            {
                "Agent": agent,
                "Precision": metrics.per_agent_precision.get(agent, 0.0),
                "Recall": metrics.per_agent_recall.get(agent, 0.0),
                "F1 Score": metrics.per_agent_f1.get(agent, 0.0),
            }
        )

    if not agent_data:
        st.info("No per-agent metrics available yet.")
        return

    df = pd.DataFrame(agent_data)

    # Display table
    st.dataframe(
        df.style.format(
            {"Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}
        ).background_gradient(cmap="RdYlGn", subset=["Precision", "Recall", "F1 Score"]),
        use_container_width=True,
    )

    # Visualize as bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="Precision", x=df["Agent"], y=df["Precision"], marker_color="lightblue")
    )
    fig.add_trace(
        go.Bar(name="Recall", x=df["Agent"], y=df["Recall"], marker_color="lightgreen")
    )
    fig.add_trace(
        go.Bar(name="F1 Score", x=df["Agent"], y=df["F1 Score"], marker_color="orange")
    )

    fig.update_layout(
        title="Per-Agent Performance Metrics",
        xaxis_title="Agent",
        yaxis_title="Score",
        barmode="group",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_analysis(evaluator, start_time, end_time):
    """Render confidence distribution and calibration analysis"""
    st.subheader("📈 Confidence Analysis")

    # Get detailed span data for confidence analysis
    try:
        # Fetch spans using provider abstraction (async call)
        async def fetch_spans():
            return await evaluator.provider.traces.get_spans(
                project=evaluator.project_name, start_time=start_time, end_time=end_time
            )

        spans_df = run_async_in_streamlit(fetch_spans())

        if spans_df.empty:
            st.info("No span data available for confidence analysis.")
            return

        # Filter for routing spans
        from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]

        if routing_spans.empty:
            st.info("No routing spans found for confidence analysis.")
            return

        # Extract confidence values
        confidences = []
        successes = []

        for _, span_row in routing_spans.iterrows():
            try:
                # Extract routing attributes
                routing_attrs = span_row.get("attributes.routing")
                if routing_attrs and isinstance(routing_attrs, dict):
                    confidence = routing_attrs.get("confidence")
                    if confidence is not None:
                        confidences.append(float(confidence))
                        # Determine success from span status
                        success = span_row.get("status") == "OK"
                        successes.append(success)
            except Exception:
                continue

        if not confidences:
            st.info("No confidence data available in routing spans.")
            return

        # Create two columns for visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Confidence distribution histogram
            fig_hist = px_express.histogram(
                x=confidences,
                nbins=20,
                title="Confidence Score Distribution",
                labels={"x": "Confidence", "y": "Count"},
            )
            fig_hist.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Confidence vs success rate
            confidence_df = pd.DataFrame({"confidence": confidences, "success": successes})

            # Bin confidences into groups
            confidence_df["confidence_bin"] = pd.cut(
                confidence_df["confidence"], bins=10, labels=False
            )
            binned = (
                confidence_df.groupby("confidence_bin")
                .agg({"success": ["mean", "count"], "confidence": "mean"})
                .reset_index()
            )
            binned.columns = ["bin", "success_rate", "count", "avg_confidence"]

            fig_calib = go.Figure()
            fig_calib.add_trace(
                go.Scatter(
                    x=binned["avg_confidence"],
                    y=binned["success_rate"],
                    mode="lines+markers",
                    name="Actual Success Rate",
                    marker=dict(size=binned["count"] * 2, sizemode="area", sizemin=4),
                )
            )
            # Add diagonal line for perfect calibration
            fig_calib.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(dash="dash", color="gray"),
                )
            )

            fig_calib.update_layout(
                title="Confidence Calibration",
                xaxis_title="Routing Confidence",
                yaxis_title="Actual Success Rate",
                height=350,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_calib, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering confidence analysis: {e}")
        logger.exception("Confidence analysis error")


def _render_temporal_analysis(evaluator, start_time, end_time):
    """Render temporal analysis of routing decisions"""
    st.subheader("📅 Temporal Analysis")

    try:
        # Fetch spans using provider abstraction (async call)
        async def fetch_spans():
            return await evaluator.provider.traces.get_spans(
                project=evaluator.project_name, start_time=start_time, end_time=end_time
            )

        spans_df = run_async_in_streamlit(fetch_spans())

        if spans_df.empty:
            st.info("No span data available for temporal analysis.")
            return

        # Filter for routing spans
        from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]

        if routing_spans.empty:
            st.info("No routing spans found for temporal analysis.")
            return

        # Extract temporal data
        temporal_data = []
        for _, span_row in routing_spans.iterrows():
            try:
                routing_attrs = span_row.get("attributes.routing")
                if routing_attrs and isinstance(routing_attrs, dict):
                    chosen_agent = routing_attrs.get("chosen_agent")
                    confidence = routing_attrs.get("confidence")
                    timestamp = span_row.get("start_time")
                    success = span_row.get("status") == "OK"

                    if chosen_agent and confidence is not None and timestamp:
                        temporal_data.append(
                            {
                                "timestamp": timestamp,
                                "agent": chosen_agent,
                                "confidence": float(confidence),
                                "success": success,
                            }
                        )
            except Exception:
                continue

        if not temporal_data:
            st.info("No temporal data available.")
            return

        df = pd.DataFrame(temporal_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Time series of routing decisions
        col1, col2 = st.columns(2)

        with col1:
            # Decisions over time by agent
            decisions_over_time = (
                df.groupby([pd.Grouper(key="timestamp", freq="1H"), "agent"])
                .size()
                .reset_index(name="count")
            )

            fig_time = px_express.line(
                decisions_over_time,
                x="timestamp",
                y="count",
                color="agent",
                title="Routing Decisions Over Time",
                labels={"timestamp": "Time", "count": "Decisions", "agent": "Agent"},
            )
            fig_time.update_layout(height=350)
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            # Success rate over time
            success_over_time = (
                df.groupby(pd.Grouper(key="timestamp", freq="1H"))["success"]
                .mean()
                .reset_index()
            )

            fig_success = px_express.line(
                success_over_time,
                x="timestamp",
                y="success",
                title="Success Rate Over Time",
                labels={"timestamp": "Time", "success": "Success Rate"},
            )
            fig_success.update_layout(height=350, yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_success, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering temporal analysis: {e}")
        logger.exception("Temporal analysis error")


def _render_annotation_section(tenant_id: str, project_name: str, lookback_hours: int):
    """Render annotation interface for human review"""
    st.subheader("📝 Routing Decision Annotations")

    # Initialize agents
    try:
        annotation_agent = AnnotationAgent(
            tenant_id=tenant_id,
            confidence_threshold=0.6
        )
        llm_annotator = LLMAutoAnnotator()
        annotation_storage = RoutingAnnotationStorage(tenant_id=tenant_id)
    except Exception as e:
        st.error(f"❌ Failed to initialize annotation agents: {e}")
        return

    # Configuration
    with st.expander("⚙️ Annotation Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Spans below this confidence need review"
            )
        with col2:
            max_annotations = st.number_input(
                "Max Annotations to Show",
                min_value=1,
                max_value=100,
                value=20,
                help="Maximum number of annotation requests to display"
            )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔍 Find Spans Needing Annotation", use_container_width=True):
            with st.spinner("Identifying spans..."):
                try:
                    annotation_agent.confidence_threshold = confidence_threshold
                    annotation_agent.max_annotations_per_run = max_annotations
                    requests = annotation_agent.identify_spans_needing_annotation(
                        lookback_hours=lookback_hours
                    )
                    st.session_state["annotation_requests"] = requests
                    st.success(f"✅ Found {len(requests)} spans needing annotation")
                except Exception as e:
                    st.error(f"❌ Error identifying spans: {e}")

    with col2:
        if st.button("🤖 Generate LLM Annotations", use_container_width=True):
            if "annotation_requests" not in st.session_state:
                st.warning("⚠️ First find spans needing annotation")
            else:
                requests = st.session_state["annotation_requests"]
                with st.spinner(f"Generating annotations for {len(requests)} spans..."):
                    try:
                        auto_annotations = llm_annotator.batch_annotate(requests)
                        st.session_state["auto_annotations"] = auto_annotations
                        st.success(f"✅ Generated {len(auto_annotations)} LLM annotations")
                    except Exception as e:
                        st.error(f"❌ Error generating annotations: {e}")

    with col3:
        stats = annotation_storage.get_annotation_statistics()
        st.metric(
            label="Total Annotations",
            value=stats.get("total", 0),
            delta=f"{stats.get('pending_review', 0)} pending review"
        )

    # Display annotation requests and allow human review
    if "annotation_requests" in st.session_state and st.session_state["annotation_requests"]:
        st.subheader("📋 Annotation Queue")

        requests = st.session_state["annotation_requests"]
        auto_annotations = st.session_state.get("auto_annotations", [])

        # Create mapping of span_id to auto_annotation
        auto_annotation_map = {
            ann.span_id: ann for ann in auto_annotations
        }

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=[p.value for p in AnnotationPriority],
                default=[p.value for p in AnnotationPriority]
            )
        with col2:
            show_annotated = st.checkbox("Show LLM-Annotated", value=True)

        # Filter requests
        filtered_requests = [
            r for r in requests
            if r.priority.value in priority_filter
            and (show_annotated or r.span_id not in auto_annotation_map)
        ]

        st.info(f"Showing {len(filtered_requests)} of {len(requests)} annotation requests")

        # Display each request
        for idx, request in enumerate(filtered_requests):
            with st.expander(
                f"{'🔴' if request.priority == AnnotationPriority.HIGH else '🟡' if request.priority == AnnotationPriority.MEDIUM else '🟢'} "
                f"{request.query[:80]}... "
                f"({request.chosen_agent}, conf: {request.routing_confidence:.2f})"
            ):
                # Request details
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Query:**")
                    st.text(request.query)

                    st.markdown("**Routing Decision:**")
                    st.write(f"- Agent: `{request.chosen_agent}`")
                    st.write(f"- Confidence: `{request.routing_confidence:.2f}`")
                    st.write(f"- Outcome: `{request.outcome.value}`")

                    st.markdown("**Reason for Annotation:**")
                    st.write(request.reason)

                with col2:
                    st.markdown("**Metadata:**")
                    st.write(f"Priority: {request.priority.value}")
                    st.write(f"Timestamp: {request.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"Span ID: `{request.span_id[:16]}...`")

                # Show LLM annotation if available
                auto_annotation = auto_annotation_map.get(request.span_id)
                if auto_annotation:
                    st.markdown("---")
                    st.markdown("**🤖 LLM Annotation:**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Label: `{auto_annotation.label.value}`")
                    with col2:
                        st.write(f"Confidence: `{auto_annotation.confidence:.2f}`")
                    with col3:
                        st.write(f"Review Needed: {'Yes' if auto_annotation.requires_human_review else 'No'}")

                    st.markdown("**Reasoning:**")
                    st.info(auto_annotation.reasoning)

                    if auto_annotation.suggested_correct_agent:
                        st.write(f"Suggested Agent: `{auto_annotation.suggested_correct_agent}`")

                # Human review interface
                st.markdown("---")
                st.markdown("**👤 Human Review:**")

                form_key = f"annotation_form_{idx}_{request.span_id}"
                with st.form(form_key):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        human_label = st.selectbox(
                            "Your Annotation",
                            options=[label.value for label in AnnotationLabel],
                            index=0 if not auto_annotation else [label.value for label in AnnotationLabel].index(auto_annotation.label.value),
                            key=f"label_{idx}"
                        )

                    with col2:
                        suggested_agent = st.text_input(
                            "Suggested Agent (if wrong)",
                            value=auto_annotation.suggested_correct_agent if auto_annotation and auto_annotation.suggested_correct_agent else "",
                            key=f"agent_{idx}"
                        )

                    reasoning = st.text_area(
                        "Reasoning",
                        value=auto_annotation.reasoning if auto_annotation else "",
                        height=100,
                        key=f"reasoning_{idx}"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        submit_button = st.form_submit_button("✅ Submit Annotation", use_container_width=True)
                    with col2:
                        if auto_annotation and not auto_annotation.requires_human_review:
                            approve_button = st.form_submit_button("👍 Approve LLM Annotation", use_container_width=True)
                        else:
                            approve_button = False

                    if submit_button:
                        try:
                            success = annotation_storage.store_human_annotation(
                                span_id=request.span_id,
                                label=AnnotationLabel(human_label),
                                reasoning=reasoning,
                                suggested_agent=suggested_agent if suggested_agent else None
                            )
                            if success:
                                st.success("✅ Annotation saved!")
                            else:
                                st.error("❌ Failed to save annotation")
                        except Exception as e:
                            st.error(f"❌ Error saving annotation: {e}")

                    if approve_button:
                        try:
                            success = annotation_storage.approve_llm_annotation(
                                span_id=request.span_id
                            )
                            if success:
                                st.success("✅ LLM annotation approved!")
                            else:
                                st.error("❌ Failed to approve annotation")
                        except Exception as e:
                            st.error(f"❌ Error approving annotation: {e}")

    else:
        st.info("👆 Click 'Find Spans Needing Annotation' to start")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Routing Evaluation", page_icon="🎯", layout="wide")
    render_routing_evaluation_tab()
