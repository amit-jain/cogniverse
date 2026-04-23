"""
Orchestration Annotation Tab

Provides UI for human annotation of orchestration workflow quality.
Implements Phase 7.5 orchestration optimization feedback loop.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import streamlit as st


def _extract_orchestration_attrs(span_row: Any) -> Dict[str, Any]:
    """Extract orchestration attributes from a span DataFrame row.

    Phoenix returns orchestration attributes in one of three formats:
    1. Single 'attributes.orchestration' column containing a dict
       (e.g., {"query": "...", "workflow_id": "...", "pattern": "..."})
    2. Flattened columns: attributes.orchestration.query, etc.
    3. Single 'attributes' column as JSON string or nested dict

    This normalizes all to: {"orchestration.query": "...", "orchestration.workflow_id": "..."}
    """
    attrs: Dict[str, Any] = {}

    if not hasattr(span_row, "index"):
        return attrs

    # Format 1: single 'attributes.orchestration' column with a dict value
    if "attributes.orchestration" in span_row.index:
        raw = span_row["attributes.orchestration"]
        if isinstance(raw, dict):
            for k, v in raw.items():
                attrs[f"orchestration.{k}"] = v
            return attrs
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        attrs[f"orchestration.{k}"] = v
                    return attrs
            except (json.JSONDecodeError, TypeError):
                pass

    # Format 2: flattened columns (attributes.orchestration.query, etc.)
    for col in span_row.index:
        col_str = str(col)
        if col_str.startswith("attributes.orchestration."):
            key = col_str.replace("attributes.", "", 1)
            val = span_row[col]
            if val is not None and str(val) != "nan":
                attrs[key] = val

    return attrs


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotation,
    OrchestrationAnnotationStorage,
)
from cogniverse_foundation.telemetry.manager import get_telemetry_manager


def run_async_in_streamlit(coro):
    """Helper to run async operations in Streamlit"""
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


def render_orchestration_annotation_tab():
    """Render the orchestration annotation UI tab"""
    st.header("🔄 Orchestration Workflow Annotation")

    st.markdown(
        """
    Annotate orchestration workflows to improve future routing and orchestration decisions.
    Your annotations become ground truth for DSPy optimization.
    """
    )

    # Tenant comes from the gate-validated session state
    tenant_id = st.session_state["current_tenant"]
    st.caption(f"Annotations scoped to tenant **{tenant_id}**")
    storage = OrchestrationAnnotationStorage(tenant_id=tenant_id)

    st.subheader("📋 Recent Orchestration Workflows")

    col1, col2 = st.columns(2)
    with col1:
        lookback_hours = st.slider("Lookback (hours)", 1, 24, 6)
    with col2:
        max_workflows = st.slider("Max workflows to show", 5, 50, 20)

    if st.button("🔄 Refresh Workflows"):
        with st.spinner("Fetching orchestration spans..."):
            try:
                telemetry_manager = get_telemetry_manager()
                provider = telemetry_manager.get_provider(tenant_id=tenant_id)

                end_time = datetime.now()
                start_time = end_time - timedelta(hours=lookback_hours)

                async def fetch_spans():
                    return await provider.traces.get_spans(
                        project=f"cogniverse-{tenant_id}",
                        start_time=start_time,
                        end_time=end_time,
                    )

                spans_df = run_async_in_streamlit(fetch_spans())

                if spans_df.empty:
                    st.warning(
                        f"No orchestration spans found in last {lookback_hours} hours"
                    )
                    st.session_state.orch_spans = None
                else:
                    # Filter for orchestration spans
                    orch_spans = spans_df[
                        spans_df["name"] == "cogniverse.orchestration"
                    ].head(max_workflows)
                    st.session_state.orch_spans = orch_spans
                    st.success(f"Found {len(orch_spans)} orchestration workflows")

            except Exception as e:
                st.error(f"Error fetching spans: {e}")
                st.session_state.orch_spans = None

    if st.session_state.get("orch_spans") is not None:
        orch_spans = st.session_state.orch_spans

        if len(orch_spans) == 0:
            st.info("No workflows available for annotation")
            return

        workflow_options = []
        for idx, (_, span) in enumerate(orch_spans.iterrows()):
            attrs = _extract_orchestration_attrs(span)
            query = attrs.get("orchestration.query", "Unknown query")
            wf_id = attrs.get("orchestration.workflow_id", f"workflow-{idx}")
            pattern = attrs.get("orchestration.pattern", "unknown")
            workflow_options.append(f"{wf_id}: {query[:50]}... ({pattern})")

        selected_idx = st.selectbox(
            "Select workflow to annotate",
            range(len(workflow_options)),
            format_func=lambda x: workflow_options[x],
        )

        span_row = orch_spans.iloc[selected_idx]
        attrs = _extract_orchestration_attrs(span_row)

        st.subheader("📊 Workflow Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pattern", attrs.get("orchestration.pattern", "N/A"))
        with col2:
            st.metric(
                "Execution Time", f"{attrs.get('orchestration.execution_time', 0):.2f}s"
            )
        with col3:
            st.metric("Tasks Completed", attrs.get("orchestration.tasks_completed", 0))

        st.text_area(
            "Query", attrs.get("orchestration.query", ""), height=100, disabled=True
        )

        agents_raw = attrs.get("orchestration.agents_used", "[]")
        execution_raw = attrs.get("orchestration.execution_order", "[]")
        try:
            agents_used = (
                json.loads(agents_raw) if isinstance(agents_raw, str) else agents_raw
            )
        except (json.JSONDecodeError, TypeError):
            agents_used = [agents_raw] if agents_raw else []
        try:
            execution_order = (
                json.loads(execution_raw)
                if isinstance(execution_raw, str)
                else execution_raw
            )
        except (json.JSONDecodeError, TypeError):
            execution_order = [execution_raw] if execution_raw else []

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Agents Used:**")
            for agent in agents_used:
                st.write(f"- {agent}")
        with col2:
            st.write("**Execution Order:**")
            for i, agent in enumerate(execution_order, 1):
                st.write(f"{i}. {agent}")

        st.subheader("✍️ Your Annotation")

        with st.form("annotation_form"):
            st.markdown("#### Orchestration Pattern")
            pattern_is_optimal = st.radio(
                "Is the orchestration pattern optimal?",
                ["Yes", "No", "Unsure"],
                horizontal=True,
            )

            suggested_pattern = None
            if pattern_is_optimal == "No":
                suggested_pattern = st.selectbox(
                    "Suggested pattern",
                    ["parallel", "sequential", "conditional", "mixed"],
                )
                pattern_feedback = st.text_area(
                    "Why is this pattern better?", height=80
                )
            else:
                pattern_feedback = None

            st.markdown("#### Agent Selection")
            agents_are_correct = st.radio(
                "Are the right agents selected?", ["Yes", "No"], horizontal=True
            )

            missing_agents = []
            unnecessary_agents = []
            suggested_agents = list(agents_used)

            if agents_are_correct == "No":
                missing_agents_input = st.text_input(
                    "Missing agents (comma-separated)", ""
                )
                if missing_agents_input:
                    missing_agents = [
                        a.strip() for a in missing_agents_input.split(",")
                    ]
                    suggested_agents.extend(missing_agents)

                unnecessary_agents_input = st.text_input(
                    "Unnecessary agents (comma-separated)", ""
                )
                if unnecessary_agents_input:
                    unnecessary_agents = [
                        a.strip() for a in unnecessary_agents_input.split(",")
                    ]
                    suggested_agents = [
                        a for a in suggested_agents if a not in unnecessary_agents
                    ]

            st.markdown("#### Execution Order")
            execution_order_is_optimal = st.radio(
                "Is the execution order optimal?", ["Yes", "No"], horizontal=True
            )

            suggested_execution_order = None
            if execution_order_is_optimal == "No":
                suggested_order_input = st.text_area(
                    "Suggested execution order (one agent per line)", height=100
                )
                if suggested_order_input:
                    suggested_execution_order = [
                        line.strip()
                        for line in suggested_order_input.split("\n")
                        if line.strip()
                    ]
                execution_order_feedback = st.text_area(
                    "Why is this order better?", height=80
                )
            else:
                execution_order_feedback = None

            st.markdown("#### Overall Quality")
            quality_label = st.select_slider(
                "Workflow quality",
                options=["failed", "poor", "acceptable", "good", "excellent"],
                value="acceptable",
            )

            quality_score = st.slider("Quality score (0.0-1.0)", 0.0, 1.0, 0.7, 0.05)

            st.markdown("#### Feedback")
            col1, col2 = st.columns(2)
            with col1:
                what_went_well = st.text_area("What went well?", height=100)
            with col2:
                what_went_wrong = st.text_area("What went wrong?", height=100)

            improvement_notes = st.text_area("Improvement suggestions", height=100)

            annotator_id = st.text_input("Your name/ID", value="annotator")

            submitted = st.form_submit_button("💾 Submit Annotation")

            if submitted:
                try:
                    annotation = OrchestrationAnnotation(
                        workflow_id=attrs.get("orchestration.workflow_id"),
                        span_id=span_row.get("context.span_id"),
                        query=attrs.get("orchestration.query"),
                        orchestration_pattern=attrs.get("orchestration.pattern"),
                        agents_used=agents_used,
                        execution_order=execution_order,
                        execution_time=float(
                            attrs.get("orchestration.execution_time", 0.0)
                        ),
                        pattern_is_optimal=(pattern_is_optimal == "Yes"),
                        suggested_pattern=suggested_pattern,
                        pattern_feedback=pattern_feedback,
                        agents_are_correct=(agents_are_correct == "Yes"),
                        missing_agents=missing_agents,
                        unnecessary_agents=unnecessary_agents,
                        suggested_agents=suggested_agents,
                        execution_order_is_optimal=(
                            execution_order_is_optimal == "Yes"
                        ),
                        suggested_execution_order=suggested_execution_order,
                        execution_order_feedback=execution_order_feedback,
                        workflow_quality_label=quality_label,
                        quality_score=quality_score,
                        improvement_notes=improvement_notes,
                        what_went_well=what_went_well,
                        what_went_wrong=what_went_wrong,
                        annotator_id=annotator_id,
                        workflow_succeeded=(span_row.get("status_code") == "OK"),
                        error_details=span_row.get("status_message"),
                    )

                    result = run_async_in_streamlit(
                        storage.store_annotation(annotation)
                    )

                    if result:
                        st.success("✅ Annotation saved successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to save annotation")

                except Exception as e:
                    st.error(f"Error saving annotation: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    else:
        st.info(
            "Click 'Refresh Workflows' to load orchestration workflows for annotation"
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Orchestration Annotation", layout="wide")
    render_orchestration_annotation_tab()
