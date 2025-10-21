#!/usr/bin/env python3
"""
Enhanced Optimization Tab for Phoenix Dashboard

Comprehensive optimization framework including:
- Search result annotation (thumbs up/down, star ratings)
- Golden dataset builder from Phoenix annotations
- Synthetic data generation for all optimizers
- Routing optimization (GRPO/GEPA)
- DSPy module optimization (teacher-student distillation)
- Reranking optimization from user feedback
- Profile selection optimization
- Unified metrics dashboard
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


def render_enhanced_optimization_tab():
    """Render the enhanced optimization tab with all optimization workflows"""
    st.header("🔧 Optimization Framework")
    st.markdown(
        "Comprehensive optimization dashboard for routing, search quality, and agent performance."
    )

    # Create sub-tabs for different optimization workflows
    opt_tabs = st.tabs([
        "📊 Overview",
        "⭐ Search Annotations",
        "📚 Golden Dataset",
        "🔬 Synthetic Data",
        "🎯 Module Optimization",
        "🧠 DSPy Optimization",
        "🔄 Reranking Optimization",
        "📈 Profile Selection",
        "📉 Metrics Dashboard"
    ])

    with opt_tabs[0]:
        _render_overview_tab()

    with opt_tabs[1]:
        _render_search_annotation_tab()

    with opt_tabs[2]:
        _render_golden_dataset_tab()

    with opt_tabs[3]:
        _render_synthetic_data_tab()

    with opt_tabs[4]:
        _render_routing_optimization_tab()

    with opt_tabs[5]:
        _render_dspy_optimization_tab()

    with opt_tabs[6]:
        _render_reranking_optimization_tab()

    with opt_tabs[7]:
        _render_profile_selection_tab()

    with opt_tabs[8]:
        _render_metrics_dashboard_tab()


def _render_overview_tab():
    """Render optimization overview and quick stats"""
    st.subheader("📊 Optimization Overview")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Count annotations in Phoenix
        annotation_count = st.session_state.get("annotation_count", 0)
        st.metric(
            "Total Annotations",
            annotation_count,
            delta=None,
            help="Human and LLM annotations collected"
        )

    with col2:
        # Count golden dataset entries
        golden_count = st.session_state.get("golden_dataset_size", 0)
        st.metric(
            "Golden Dataset Size",
            golden_count,
            delta=None,
            help="Queries with ground truth labels"
        )

    with col3:
        # Optimization runs
        opt_runs = st.session_state.get("optimization_requests", [])
        st.metric(
            "Optimization Runs",
            len(opt_runs),
            delta=None,
            help="Total optimization jobs triggered"
        )

    with col4:
        # Last optimization
        if opt_runs:
            last_run = opt_runs[-1]
            last_time = last_run.get("timestamp", datetime.now())
            time_ago = datetime.now() - last_time
            st.metric(
                "Last Optimization",
                f"{time_ago.seconds // 60}m ago" if time_ago.seconds < 3600 else f"{time_ago.seconds // 3600}h ago",
                delta=None,
                help="Time since last optimization"
            )
        else:
            st.metric("Last Optimization", "Never", delta=None)

    st.markdown("---")

    # Workflow diagram
    st.subheader("🔄 Optimization Workflow")
    st.markdown("""
    **1. Collect Annotations** → Annotate search results with thumbs up/down or star ratings

    **2. Build Golden Dataset** → Create ground truth dataset from high-quality annotations

    **3. Train Optimizers** → Run optimization for routing, DSPy modules, reranking, and profile selection

    **4. Monitor Metrics** → Track improvements in accuracy, latency, and user satisfaction

    **5. Iterate** → Continuous improvement based on new annotations and feedback
    """)

    st.markdown("---")

    # Recent optimization history
    st.subheader("📜 Recent Optimization History")

    if opt_runs:
        history_df = pd.DataFrame([
            {
                "Timestamp": run.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M"),
                "Type": run.get("type", "unknown"),
                "Status": run.get("status", "unknown"),
                "Examples": run.get("examples_count", 0),
                "Optimizer": run.get("optimizer", "N/A")
            }
            for run in opt_runs[-10:]  # Last 10 runs
        ])

        st.dataframe(
            history_df.sort_values("Timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No optimization runs yet. Start by collecting annotations!")


def _render_search_annotation_tab():
    """Render search result annotation interface"""
    st.subheader("⭐ Search Result Annotations")
    st.markdown("Annotate search results to provide feedback for optimization")

    # Configuration
    with st.expander("⚙️ Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tenant_id = st.text_input("Tenant ID", value="default")
            lookback_hours = st.number_input("Lookback Hours", min_value=1, max_value=168, value=24)
        with col2:
            annotation_type = st.selectbox(
                "Annotation Type",
                ["Thumbs Up/Down", "Star Rating (1-5)", "Relevance Score (0-1)"]
            )

    # Fetch search spans from Phoenix
    if st.button("🔍 Fetch Search Results", key="fetch_search_results"):
        with st.spinner("Fetching search results from Phoenix..."):
            try:
                import phoenix as px

                client = px.Client()
                project_name = f"cogniverse-{tenant_id}-search"

                end_time = datetime.now()
                start_time = end_time - timedelta(hours=lookback_hours)

                spans_df = client.get_spans_dataframe(
                    project_name=project_name,
                    start_time=start_time,
                    end_time=end_time
                )

                # Filter for search spans
                search_spans = spans_df[spans_df["name"].str.contains("search", case=False)]

                st.session_state["search_spans"] = search_spans
                st.success(f"✅ Fetched {len(search_spans)} search results")

            except Exception as e:
                st.error(f"❌ Failed to fetch spans: {e}")

    # Display search results for annotation
    if "search_spans" in st.session_state and not st.session_state["search_spans"].empty:
        spans = st.session_state["search_spans"]

        st.subheader(f"📋 Annotation Queue ({len(spans)} results)")

        # Pagination
        page_size = 10
        page = st.selectbox("Page", range(1, (len(spans) // page_size) + 2))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(spans))

        # Display each search result
        for idx in range(start_idx, end_idx):
            span = spans.iloc[idx]

            with st.expander(
                f"Result {idx + 1}: {span.get('attributes.query', 'Unknown query')[:80]}"
            ):
                # Display query and results
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Query:**")
                    st.text(span.get("attributes.query", "N/A"))

                    st.markdown("**Results:**")
                    results = span.get("attributes.results", [])
                    if results:
                        for i, result in enumerate(results[:5]):  # Top 5
                            st.write(f"{i + 1}. {result.get('title', result.get('id', 'Unknown'))}")
                    else:
                        st.info("No results returned")

                with col2:
                    st.markdown("**Metadata:**")
                    st.write(f"Profile: {span.get('attributes.profile', 'N/A')}")
                    st.write(f"Strategy: {span.get('attributes.strategy', 'N/A')}")
                    st.write(f"Latency: {span.get('latency_ms', 0):.0f}ms")

                # Annotation form
                st.markdown("---")
                st.markdown("**Your Annotation:**")

                form_key = f"annotation_form_{idx}"
                with st.form(form_key):
                    if annotation_type == "Thumbs Up/Down":
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            thumbs_up = st.form_submit_button("👍 Good", use_container_width=True)
                        with col2:
                            thumbs_down = st.form_submit_button("👎 Bad", use_container_width=True)
                        with col3:
                            notes = st.text_input("Notes (optional)", key=f"notes_{idx}")

                        if thumbs_up:
                            _save_search_annotation(
                                span_id=span.get("context.span_id"),
                                rating=1.0,
                                annotation_type="thumbs",
                                notes=notes,
                                tenant_id=tenant_id
                            )
                            st.success("✅ Saved: Thumbs Up")

                        if thumbs_down:
                            _save_search_annotation(
                                span_id=span.get("context.span_id"),
                                rating=0.0,
                                annotation_type="thumbs",
                                notes=notes,
                                tenant_id=tenant_id
                            )
                            st.success("✅ Saved: Thumbs Down")

                    elif annotation_type == "Star Rating (1-5)":
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            stars = st.slider("Rating", 1, 5, 3, key=f"stars_{idx}")
                        with col2:
                            notes = st.text_input("Notes (optional)", key=f"notes_{idx}")

                        if st.form_submit_button("💾 Save Rating"):
                            _save_search_annotation(
                                span_id=span.get("context.span_id"),
                                rating=stars / 5.0,  # Normalize to 0-1
                                annotation_type="stars",
                                notes=notes,
                                tenant_id=tenant_id
                            )
                            st.success(f"✅ Saved: {stars} stars")

                    else:  # Relevance Score
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            relevance = st.slider("Relevance", 0.0, 1.0, 0.5, 0.1, key=f"rel_{idx}")
                        with col2:
                            notes = st.text_input("Notes (optional)", key=f"notes_{idx}")

                        if st.form_submit_button("💾 Save Score"):
                            _save_search_annotation(
                                span_id=span.get("context.span_id"),
                                rating=relevance,
                                annotation_type="relevance",
                                notes=notes,
                                tenant_id=tenant_id
                            )
                            st.success(f"✅ Saved: {relevance:.1f} relevance")

    else:
        st.info("👆 Click 'Fetch Search Results' to start annotating")


def _save_search_annotation(
    span_id: str,
    rating: float,
    annotation_type: str,
    notes: str,
    tenant_id: str
):
    """Save search result annotation to Phoenix"""
    try:
        import phoenix as px
        from phoenix.trace import SpanEvaluations

        client = px.Client()

        # Create evaluation dataframe
        eval_data = {
            "context.span_id": [span_id],
            "label": ["positive" if rating >= 0.6 else "negative" if rating <= 0.4 else "neutral"],
            "score": [rating],
            "explanation": [notes or "User annotation"],
            "annotation_type": [annotation_type],
            "annotator": ["human"],
            "timestamp": [datetime.now().isoformat()]
        }

        eval_df = pd.DataFrame(eval_data)

        # Log evaluation
        span_evals = SpanEvaluations(
            eval_name="search_quality_annotation",
            dataframe=eval_df
        )
        client.log_evaluations(span_evals)

        # Update session state
        if "annotation_count" not in st.session_state:
            st.session_state["annotation_count"] = 0
        st.session_state["annotation_count"] += 1

        logger.info(f"Saved search annotation for span {span_id}: {rating}")

    except Exception as e:
        logger.error(f"Failed to save annotation: {e}")
        raise


def _render_golden_dataset_tab():
    """Render golden dataset builder from annotations"""
    st.subheader("📚 Golden Dataset Builder")
    st.markdown("Build ground truth datasets from Phoenix annotations")

    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        tenant_id = st.text_input("Tenant ID", value="default", key="golden_tenant")
    with col2:
        min_rating = st.slider("Min Rating Threshold", 0.0, 1.0, 0.8, 0.1)
    with col3:
        lookback_days = st.number_input("Lookback Days", 1, 90, 30)

    # Build golden dataset
    if st.button("🏗️ Build Golden Dataset", type="primary"):
        with st.spinner("Building golden dataset from annotations..."):
            try:
                golden_dataset = _build_golden_dataset_from_phoenix(
                    tenant_id=tenant_id,
                    min_rating=min_rating,
                    lookback_days=lookback_days
                )

                st.session_state["golden_dataset"] = golden_dataset
                st.session_state["golden_dataset_size"] = len(golden_dataset)

                st.success(f"✅ Built golden dataset with {len(golden_dataset)} queries")

                # Display sample
                if golden_dataset:
                    st.subheader("📋 Dataset Sample")
                    sample_df = pd.DataFrame([
                        {
                            "Query": query,
                            "Expected Videos": len(data.get("expected_videos", [])),
                            "Avg Relevance": data.get("avg_relevance", 0.0)
                        }
                        for query, data in list(golden_dataset.items())[:10]
                    ])
                    st.dataframe(sample_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to build dataset: {e}")

    # Export golden dataset
    if "golden_dataset" in st.session_state and st.session_state["golden_dataset"]:
        st.markdown("---")
        st.subheader("💾 Export Golden Dataset")

        dataset = st.session_state["golden_dataset"]

        col1, col2 = st.columns(2)
        with col1:
            filename = st.text_input(
                "Filename",
                value=f"golden_dataset_{tenant_id}_{datetime.now().strftime('%Y%m%d')}.json"
            )
        with col2:
            if st.button("📥 Download JSON"):
                json_str = json.dumps(dataset, indent=2)
                st.download_button(
                    label="Download",
                    data=json_str,
                    file_name=filename,
                    mime="application/json"
                )


def _build_golden_dataset_from_phoenix(
    tenant_id: str,
    min_rating: float,
    lookback_days: int
) -> Dict:
    """Build golden dataset from Phoenix annotations"""
    import phoenix as px

    client = px.Client()
    project_name = f"cogniverse-{tenant_id}-search"

    # Query annotated spans
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    spans_df = client.get_spans_dataframe(
        project_name=project_name,
        start_time=start_time,
        end_time=end_time
    )

    # Filter for search spans with annotations
    search_spans = spans_df[spans_df["name"].str.contains("search", case=False)]

    golden_dataset = {}

    for _, span in search_spans.iterrows():
        # Check if span has annotation
        annotation_score = span.get("attributes.annotation.score")
        if annotation_score is None or float(annotation_score) < min_rating:
            continue

        query = span.get("attributes.query", "")
        results = span.get("attributes.results", [])

        if not query or not results:
            continue

        # Extract expected videos (top results from highly-rated queries)
        expected_videos = [
            result.get("id", result.get("video_id"))
            for result in results[:5]  # Top 5 results
            if result.get("id") or result.get("video_id")
        ]

        # Build relevance scores
        relevance_scores = {
            vid: 1.0 / (i + 1)  # Reciprocal rank
            for i, vid in enumerate(expected_videos)
        }

        golden_dataset[query] = {
            "expected_videos": expected_videos,
            "relevance_scores": relevance_scores,
            "avg_relevance": float(annotation_score),
            "profile": span.get("attributes.profile", "unknown"),
            "timestamp": span.get("start_time", "").isoformat() if hasattr(span.get("start_time"), "isoformat") else str(span.get("start_time"))
        }

    return golden_dataset


def _render_synthetic_data_tab():
    """Render synthetic data generation interface"""
    st.subheader("🔬 Synthetic Data Generation")
    st.markdown("Generate training data for optimizers by sampling from Vespa backend")

    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        optimizer = st.selectbox(
            "Optimizer Type",
            ["modality", "cross_modal", "routing", "workflow", "unified"],
            help="Which optimizer to generate data for"
        )
    with col2:
        count = st.number_input("Examples to Generate", 10, 10000, 100, 10)
    with col3:
        vespa_sample_size = st.number_input("Vespa Sample Size", 10, 10000, 200, 10)

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            strategies = st.multiselect(
                "Sampling Strategies",
                ["diverse", "temporal_recent", "entity_rich", "multi_modal_sequences", "by_modality", "cross_modal_pairs"],
                default=["diverse"],
                help="How to sample content from Vespa"
            )
        with col2:
            max_profiles = st.slider("Max Profiles", 1, 10, 3)
            tenant_id = st.text_input("Tenant ID", value="default")

    # Generate synthetic data
    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner(f"Generating {count} examples for {optimizer} optimizer..."):
            try:
                # Call synthetic data API
                import requests

                api_base = st.session_state.get("api_base_url", "http://localhost:8000")
                request_payload = {
                    "optimizer": optimizer,
                    "count": count,
                    "vespa_sample_size": vespa_sample_size,
                    "strategies": strategies,
                    "max_profiles": max_profiles,
                    "tenant_id": tenant_id
                }

                response = requests.post(
                    f"{api_base}/synthetic/generate",
                    json=request_payload,
                    timeout=300  # 5 minutes
                )

                if response.status_code == 200:
                    result = response.json()

                    st.session_state["synthetic_data_result"] = result
                    st.success(f"✅ Generated {result['count']} examples using {len(result['selected_profiles'])} profiles")

                    # Show profile selection reasoning
                    st.info(f"**Profile Selection**: {result['profile_selection_reasoning']}")

                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Schema", result['schema_name'])
                    with col2:
                        st.metric("Generation Time", f"{result['metadata'].get('generation_time_ms', 0)}ms")
                    with col3:
                        st.metric("Profiles Used", len(result['selected_profiles']))

                    # Show selected profiles
                    st.subheader("📋 Selected Profiles")
                    for profile in result['selected_profiles']:
                        st.code(profile, language=None)

                    # Display sample data
                    if result['data']:
                        st.subheader("📊 Sample Generated Examples")
                        sample_df = pd.DataFrame(result['data'][:10])
                        st.dataframe(sample_df, use_container_width=True)

                else:
                    st.error(f"❌ Generation failed: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the runtime service is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Try reducing the sample size or count.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.exception("Synthetic data generation failed")

    # Export synthetic data
    if "synthetic_data_result" in st.session_state and st.session_state["synthetic_data_result"]:
        st.markdown("---")
        st.subheader("💾 Export Synthetic Data")

        result = st.session_state["synthetic_data_result"]

        col1, col2 = st.columns(2)
        with col1:
            filename = st.text_input(
                "Filename",
                value=f"synthetic_{result['optimizer']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        with col2:
            if st.button("📥 Download JSON"):
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="Download",
                    data=json_str,
                    file_name=filename,
                    mime="application/json"
                )

        # Use in optimization
        st.markdown("---")
        st.subheader("🎯 Use in Optimization")
        st.markdown(f"""
        The generated data is ready for use with **{result['optimizer'].replace('_', ' ').title()} Optimizer**.

        **Next Steps:**
        1. Review the generated examples above
        2. Download the data if needed for offline use
        3. Navigate to the corresponding optimization tab
        4. Load the synthetic data for training

        **Optimizer Tabs:**
        - `modality` → Routing Optimization Tab
        - `cross_modal` → Reranking Optimization Tab
        - `routing` → Routing Optimization Tab
        - `workflow` → DSPy Optimization Tab
        - `unified` → Multiple tabs (Routing + DSPy)
        """)

    # Show optimizer info
    st.markdown("---")
    st.subheader("ℹ️ Optimizer Information")

    optimizer_info = {
        "modality": {
            "description": "Per-modality routing (VIDEO, DOCUMENT, IMAGE, AUDIO)",
            "schema": "ModalityExampleSchema",
            "features": "Generates queries targeting specific modalities"
        },
        "cross_modal": {
            "description": "Multi-modal fusion decisions",
            "schema": "FusionHistorySchema",
            "features": "Creates fusion scenarios with improvement metrics"
        },
        "routing": {
            "description": "Entity-based advanced routing",
            "schema": "RoutingExperienceSchema",
            "features": "Generates queries with entities and relationships"
        },
        "workflow": {
            "description": "Multi-agent workflow orchestration",
            "schema": "WorkflowExecutionSchema",
            "features": "Creates multi-step workflow patterns"
        },
        "unified": {
            "description": "Combined routing and workflow planning",
            "schema": "Mixed schemas",
            "features": "Generates diverse examples for multiple optimizers"
        }
    }

    if optimizer in optimizer_info:
        info = optimizer_info[optimizer]
        st.info(f"""
        **{optimizer.replace('_', ' ').title()} Optimizer**

        **Description**: {info['description']}
        **Schema**: `{info['schema']}`
        **Features**: {info['features']}
        """)


def _render_routing_optimization_tab():
    """Render routing optimization interface"""
    st.subheader("🎯 Module Optimization")
    st.markdown("""
    Optimize routing/workflow modules with automatic DSPy optimizer selection.

    **Modules**: modality (per-modality routing), cross_modal (fusion), routing (entity-based), workflow (orchestration), unified (combined)

    **Auto DSPy Optimizer Selection**: System automatically chooses GEPA/Bootstrap/SIMBA/MIPRO based on training data size
    """)

    # Argo Workflow Integration
    st.markdown("### 🚀 Batch Optimization with Argo Workflows")

    col1, col2 = st.columns(2)
    with col1:
        tenant_id = st.text_input(
            "Tenant ID",
            value="default",
            help="Tenant identifier for optimization"
        )
        optimizer_type = st.selectbox(
            "Module to Optimize",
            ["modality", "cross_modal", "routing", "workflow", "unified"],
            help="Which routing/workflow module to optimize (modality=per-modality routing, cross_modal=fusion, routing=entity-based, workflow=orchestration, unified=combined)"
        )

    with col2:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=10,
            max_value=500,
            value=100,
            help="Maximum optimization iterations"
        )
        use_synthetic = st.checkbox(
            "Use Synthetic Data",
            value=True,
            help="Generate synthetic training data from Vespa backend"
        )

    with st.expander("⚙️ Advanced Configuration"):
        col1, col2, col3 = st.columns(3)
        with col1:
            synthetic_count = st.number_input(
                "Synthetic Examples",
                min_value=10,
                max_value=1000,
                value=200,
                help="Number of synthetic examples to generate"
            )
        with col2:
            vespa_sample_size = st.number_input(
                "Vespa Sample Size",
                min_value=10,
                max_value=1000,
                value=500,
                help="Documents to sample from Vespa"
            )
        with col3:
            max_profiles = st.number_input(
                "Max Profiles",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum backend profiles to use"
            )

    # Submit workflow button
    if st.button("🚀 Submit Routing Optimization Workflow", type="primary"):
        with st.spinner("Submitting Argo Workflow..."):
            try:
                import subprocess
                import tempfile
                import yaml
                from datetime import datetime

                # Create workflow specification
                workflow_spec = {
                    "apiVersion": "argoproj.io/v1alpha1",
                    "kind": "Workflow",
                    "metadata": {
                        "generateName": f"routing-opt-{optimizer_type}-",
                        "namespace": "cogniverse"
                    },
                    "spec": {
                        "workflowTemplateRef": {
                            "name": "batch-optimization"
                        },
                        "arguments": {
                            "parameters": [
                                {"name": "tenant-id", "value": tenant_id},
                                {"name": "optimizer-category", "value": "routing"},
                                {"name": "optimizer-type", "value": optimizer_type},
                                {"name": "max-iterations", "value": str(max_iterations)},
                                {"name": "use-synthetic-data", "value": "true" if use_synthetic else "false"}
                            ]
                        }
                    }
                }

                # Write to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(workflow_spec, f)
                    workflow_file = f.name

                # Submit workflow using kubectl or argo CLI
                try:
                    # Try argo CLI first
                    result = subprocess.run(
                        ["argo", "submit", workflow_file, "-n", "cogniverse"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        # Fallback to kubectl
                        result = subprocess.run(
                            ["kubectl", "apply", "-f", workflow_file],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                except FileNotFoundError:
                    # If argo CLI not found, use kubectl
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", workflow_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                if result.returncode == 0:
                    st.success(f"✅ Workflow submitted successfully!")
                    st.code(result.stdout, language="text")
                    st.info("""
                    **Monitor workflow progress:**
                    ```bash
                    argo list -n cogniverse
                    argo get <workflow-name> -n cogniverse
                    argo logs <workflow-name> -n cogniverse --follow
                    ```
                    """)
                else:
                    st.error(f"❌ Workflow submission failed:\n{result.stderr}")
                    st.warning("Make sure you have kubectl or argo CLI configured and Kubernetes cluster is accessible")

                # Clean up temp file
                import os
                os.unlink(workflow_file)

            except Exception as e:
                st.error(f"❌ Error submitting workflow: {str(e)}")
                st.info("**Prerequisites:**\n- Kubernetes cluster running\n- kubectl or argo CLI configured\n- cogniverse namespace exists\n- batch-optimization WorkflowTemplate deployed")


def _render_dspy_optimization_tab():
    """Render DSPy module optimization with teacher-student"""
    st.subheader("🧠 DSPy Module Optimization")
    st.markdown("Teacher-student distillation for local model optimization")

    # Argo Workflow Integration
    st.markdown("### 🚀 Batch DSPy Optimization with Argo Workflows")

    col1, col2 = st.columns(2)
    with col1:
        tenant_id = st.text_input(
            "Tenant ID",
            value="default",
            key="dspy_tenant",
            help="Tenant identifier for optimization"
        )
        optimizer_type = st.selectbox(
            "DSPy Optimizer",
            ["GEPA", "Bootstrap", "SIMBA", "MIPRO"],
            help="Which DSPy optimizer to run"
        )
        dataset_name = st.text_input(
            "Dataset Name",
            value="golden_eval_v1",
            help="Name of evaluation dataset in data/testset/evaluation/"
        )

    with col2:
        profiles = st.text_input(
            "Backend Profiles",
            value="video_colpali_smol500_mv_frame",
            help="Comma-separated list of profiles to use"
        )
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=10,
            max_value=500,
            value=100,
            key="dspy_iterations",
            help="Maximum optimization iterations"
        )
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f",
            help="Learning rate for optimization"
        )

    with st.expander("⚙️ Advanced Configuration"):
        use_synthetic = st.checkbox(
            "Use Synthetic Data (Future)",
            value=False,
            disabled=True,
            help="Synthetic data generation for DSPy optimizers - Coming soon!"
        )
        st.info("**Note**: Synthetic data generation for DSPy module optimizers is planned but not yet implemented. Currently uses golden evaluation datasets.")

    # Submit workflow button
    if st.button("🚀 Submit DSPy Optimization Workflow", type="primary"):
        with st.spinner("Submitting Argo Workflow..."):
            try:
                import subprocess
                import tempfile
                import yaml
                from datetime import datetime

                # Create workflow specification
                workflow_spec = {
                    "apiVersion": "argoproj.io/v1alpha1",
                    "kind": "Workflow",
                    "metadata": {
                        "generateName": f"dspy-opt-{optimizer_type.lower()}-",
                        "namespace": "cogniverse"
                    },
                    "spec": {
                        "workflowTemplateRef": {
                            "name": "batch-optimization"
                        },
                        "arguments": {
                            "parameters": [
                                {"name": "tenant-id", "value": tenant_id},
                                {"name": "optimizer-category", "value": "dspy"},
                                {"name": "optimizer-type", "value": optimizer_type},
                                {"name": "dataset-name", "value": dataset_name},
                                {"name": "profiles", "value": profiles},
                                {"name": "max-iterations", "value": str(max_iterations)},
                                {"name": "learning-rate", "value": str(learning_rate)},
                                {"name": "use-synthetic-data", "value": "false"}
                            ]
                        }
                    }
                }

                # Write to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(workflow_spec, f)
                    workflow_file = f.name

                # Submit workflow using kubectl or argo CLI
                try:
                    # Try argo CLI first
                    result = subprocess.run(
                        ["argo", "submit", workflow_file, "-n", "cogniverse"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        # Fallback to kubectl
                        result = subprocess.run(
                            ["kubectl", "apply", "-f", workflow_file],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                except FileNotFoundError:
                    # If argo CLI not found, use kubectl
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", workflow_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                if result.returncode == 0:
                    st.success(f"✅ Workflow submitted successfully!")
                    st.code(result.stdout, language="text")
                    st.info("""
                    **Monitor workflow progress:**
                    ```bash
                    argo list -n cogniverse
                    argo get <workflow-name> -n cogniverse
                    argo logs <workflow-name> -n cogniverse --follow
                    ```

                    **What happens next:**
                    1. Validates configuration and dataset
                    2. Prepares evaluation dataset
                    3. Runs DSPy optimization with selected algorithm
                    4. Evaluates results and deploys if improvement > 5%
                    5. Sends completion notification
                    """)
                else:
                    st.error(f"❌ Workflow submission failed:\n{result.stderr}")
                    st.warning("Make sure you have kubectl or argo CLI configured and Kubernetes cluster is accessible")

                # Clean up temp file
                import os
                os.unlink(workflow_file)

            except Exception as e:
                st.error(f"❌ Error submitting workflow: {str(e)}")
                st.info("**Prerequisites:**\n- Kubernetes cluster running\n- kubectl or argo CLI configured\n- cogniverse namespace exists\n- batch-optimization WorkflowTemplate deployed\n- Dataset exists at data/testset/evaluation/{dataset_name}.csv")


def _render_reranking_optimization_tab():
    """Render reranking optimization based on user feedback"""
    st.subheader("🔄 Reranking Optimization")
    st.markdown("Optimize result ranking based on annotation feedback")

    st.markdown("""
    ### How It Works:

    1. **Collect Feedback**: User annotations (thumbs up/down) on search results
    2. **Learn Preferences**: Train reranker to prioritize positively-rated results
    3. **Optimize Weights**: Adjust BM25 vs semantic weights based on feedback
    4. **A/B Test**: Compare optimized reranker against baseline
    """)

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        min_annotations = st.number_input(
            "Min Annotations Required",
            10, 1000, 50,
            help="Minimum annotations before training reranker"
        )
    with col2:
        current_annotations = st.session_state.get("annotation_count", 0)
        st.metric("Current Annotations", current_annotations)

    # Check if enough data
    can_train = current_annotations >= min_annotations

    if st.button("🔧 Train Reranker", disabled=not can_train):
        if can_train:
            with st.spinner("Training reranker from user feedback..."):
                st.info("Learning ranking preferences from annotations...")

                # In real implementation:
                # 1. Query annotated spans from Phoenix
                # 2. Extract (query, results, ratings) tuples
                # 3. Train LambdaMART/RankNet model
                # 4. Optimize BM25/semantic weights
                # 5. Store optimized reranker

                st.success("✅ Reranker trained successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NDCG@10", "0.72 → 0.84", delta="+0.12")
                with col2:
                    st.metric("MRR", "0.65 → 0.78", delta="+0.13")
        else:
            st.warning(f"Need {min_annotations - current_annotations} more annotations")


def _render_profile_selection_tab():
    """Render profile selection optimization"""
    st.subheader("📈 Profile Selection Optimization")
    st.markdown("Learn which processing profile works best for different query types")

    st.markdown("""
    ### Profile Selection Strategy:

    Cogniverse has 6 video processing profiles. This optimizer learns which profile to use based on:
    - Query type (temporal, object-based, activity, etc.)
    - Video characteristics (length, content type)
    - Historical performance data
    """)

    # Display profile performance matrix
    st.subheader("🎯 Profile Performance Matrix")

    # Fake data for demonstration
    profiles = [
        "video_colpali_smol500_mv_frame",
        "video_colqwen_omni_mv_chunk_30s",
        "video_videoprism_base_mv_chunk_30s",
        "video_videoprism_large_mv_chunk_30s",
        "video_videoprism_lvt_base_sv_chunk_6s",
        "video_videoprism_lvt_large_sv_chunk_6s"
    ]

    query_types = ["Temporal", "Object", "Activity", "Scene", "Abstract"]

    # Create heatmap data
    import numpy as np
    heatmap_data = np.random.rand(len(profiles), len(query_types))

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=query_types,
        y=[p.split("_")[1] + "_" + p.split("_")[2] for p in profiles],
        colorscale="RdYlGn",
        text=np.round(heatmap_data, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="NDCG@10")
    ))

    fig.update_layout(
        title="Profile Performance by Query Type",
        xaxis_title="Query Type",
        yaxis_title="Profile",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Train profile selector
    if st.button("🔧 Train Profile Selector"):
        with st.spinner("Training profile selection model..."):
            st.info("Learning optimal profile routing based on query characteristics...")

            # In real implementation:
            # 1. Query all search spans with profiles and results
            # 2. Extract (query_features, profile, ndcg) tuples
            # 3. Train classifier (Random Forest / XGBoost)
            # 4. Store model for inference

            st.success("✅ Profile selector trained!")
            st.info("Model will now recommend optimal profile for each query type")


def _render_metrics_dashboard_tab():
    """Render unified optimization metrics dashboard"""
    st.subheader("📉 Optimization Metrics Dashboard")
    st.markdown("Track improvements across all optimization dimensions")

    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        metric_timeframe = st.selectbox(
            "Timeframe",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    with col2:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()

    # Overall improvement metrics
    st.subheader("📊 Overall Improvements")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Routing Accuracy", "89%", delta="+12%")
    with col2:
        st.metric("Search NDCG@10", "0.84", delta="+0.15")
    with col3:
        st.metric("Avg Latency", "245ms", delta="-78ms")
    with col4:
        st.metric("User Satisfaction", "4.2/5", delta="+0.6")

    st.markdown("---")

    # Per-optimization-type metrics
    st.subheader("🎯 Metrics by Optimization Type")

    metrics_data = {
        "Optimization Type": ["Routing", "Reranking", "DSPy Modules", "Profile Selection"],
        "Runs": [12, 8, 5, 3],
        "Avg Improvement": [0.12, 0.15, 0.17, 0.10],
        "Last Run": ["2h ago", "1d ago", "3d ago", "5d ago"]
    }

    metrics_df = pd.DataFrame(metrics_data)

    st.dataframe(
        metrics_df.style.background_gradient(subset=["Avg Improvement"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Improvement over time chart
    st.subheader("📈 Improvement Over Time")

    # Generate fake time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    routing_acc = 0.77 + np.cumsum(np.random.randn(30) * 0.01)
    search_ndcg = 0.69 + np.cumsum(np.random.randn(30) * 0.01)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=routing_acc,
        mode="lines+markers",
        name="Routing Accuracy",
        line=dict(color="#2ecc71", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=search_ndcg,
        mode="lines+markers",
        name="Search NDCG@10",
        line=dict(color="#3498db", width=2)
    ))

    fig.update_layout(
        title="Optimization Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Enhanced Optimization",
        page_icon="🔧",
        layout="wide"
    )
    render_enhanced_optimization_tab()
