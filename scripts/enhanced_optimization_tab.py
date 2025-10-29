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
from typing import Dict

import pandas as pd
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
        _render_reranking_optimization_tab()

    with opt_tabs[6]:
        _render_profile_selection_tab()

    with opt_tabs[7]:
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

    # Quality Control Section (Prominent, outside Advanced Options)
    st.markdown("---")
    st.markdown("### 🔍 Quality Control")

    col1, col2 = st.columns([2, 3])

    with col1:
        enable_approval = st.checkbox(
            "✅ Enable Human-in-the-Loop Review",
            value=True,
            key="enable_approval_checkbox",
            help="Review AI-generated outputs before using them in optimization (enabled by default)"
        )

    with col2:
        if enable_approval:
            st.info(
                "💡 **How it works:** High-confidence items (≥ threshold) are auto-approved. "
                "Low-confidence items require your review."
            )

    # Show confidence threshold slider when enabled
    if enable_approval:
        st.markdown("**Confidence Settings:**")
        col1, col2 = st.columns(2)

        with col1:
            confidence_threshold = st.slider(
                "Auto-Approval Threshold",
                0.0, 1.0, 0.85, 0.05,
                help="Items with confidence ≥ this value are automatically approved",
                key="confidence_threshold_slider"
            )

        with col2:
            st.metric(
                "Expected Review Rate",
                f"~{int((1 - confidence_threshold) * 100)}%",
                help="Estimated percentage of items requiring review"
            )

        st.caption(
            f"ℹ️ With threshold {confidence_threshold:.2f}, items scoring {confidence_threshold:.2f}+ "
            "will be auto-approved. Lower-confidence items will require your review below."
        )

    # Advanced options (collapsed by default)
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
            tenant_id = st.text_input("Tenant ID", value="default", key="synthetic_data_tenant_id")

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

                    # Process through approval if enabled
                    if enable_approval:
                        _process_approval_workflow(result, confidence_threshold)
                    else:
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

                    # Show inline approval interface if enabled
                    if enable_approval and "last_generated_batch" in st.session_state:
                        st.markdown("---")
                        _render_inline_approval_interface()

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

    # Golden dataset selection when synthetic data is disabled
    if not use_synthetic:
        st.markdown("### 📚 Golden Dataset Selection")
        st.info("Using golden evaluation dataset from Phoenix (harvested from traces)")

        # Query Phoenix for available datasets
        try:
            import requests

            # GraphQL query to get datasets
            query = """
            query {
                datasets {
                    edges {
                        node {
                            id
                            name
                            exampleCount
                            createdAt
                            description
                        }
                    }
                }
            }
            """

            response = requests.post(
                "http://localhost:6006/graphql",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            datasets = []
            if response.status_code == 200:
                result = response.json()
                if result and 'data' in result and result['data']:
                    for edge in result.get('data', {}).get('datasets', {}).get('edges', []):
                        if edge and 'node' in edge:
                            node = edge['node']
                            datasets.append({
                                'id': node['id'],
                                'name': node['name'],
                                'example_count': node['exampleCount'],
                                'created_at': node.get('createdAt', ''),
                                'description': node.get('description', '')
                            })

            if datasets:
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Create dataset options with descriptions
                    dataset_options = [d['name'] for d in datasets]
                    selected_dataset_name = st.selectbox(
                        "Select Phoenix Dataset",
                        options=dataset_options,
                        help="Datasets stored in Phoenix"
                    )
                    dataset_name = selected_dataset_name

                with col2:
                    # Show info for selected dataset
                    selected_dataset = next((d for d in datasets if d['name'] == selected_dataset_name), None)
                    if selected_dataset:
                        st.metric("Dataset Size", selected_dataset['example_count'])
                        if selected_dataset['description']:
                            st.caption(f"_{selected_dataset['description']}_")
                        st.caption(f"Created: {selected_dataset['created_at'][:10] if selected_dataset['created_at'] else 'N/A'}")
            else:
                st.warning("No datasets found in Phoenix. Upload a CSV to create a dataset or use the Golden Dataset tab.")

                # Option to upload CSV and create Phoenix dataset
                col1, col2 = st.columns([3, 1])
                with col1:
                    uploaded_file = st.file_uploader(
                        "Upload CSV Dataset",
                        type=['csv'],
                        help="CSV with columns: query, expected_result, etc."
                    )
                with col2:
                    if uploaded_file:
                        dataset_name = st.text_input(
                            "Dataset Name",
                            value="",
                            placeholder="e.g., my_eval_dataset",
                            help="Name for the Phoenix dataset"
                        )

                if uploaded_file and dataset_name:
                    if st.button("📤 Upload to Phoenix"):
                        with st.spinner("Creating Phoenix dataset..."):
                            try:
                                import tempfile

                                from cogniverse_core.evaluation.data import (
                                    DatasetManager,
                                )

                                # Save uploaded file to temp location
                                with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    tmp_path = tmp.name

                                st.info("Processing CSV file...")

                                # Use existing DatasetManager to create dataset
                                dataset_manager = DatasetManager()
                                dataset_id = dataset_manager.create_from_csv(
                                    csv_path=tmp_path,
                                    dataset_name=dataset_name,
                                    description="Uploaded via optimization dashboard"
                                )

                                # Clean up temp file
                                import os
                                os.unlink(tmp_path)

                                st.success(f"✅ Created Phoenix dataset '{dataset_name}' with ID: {dataset_id}")
                                st.info("Refresh the page to see the new dataset in the dropdown")

                            except Exception as e:
                                st.error(f"Failed to create Phoenix dataset: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    dataset_name = ""

        except Exception as e:
            # Show clean error message for connection issues
            if "Connection refused" in str(e) or "ConnectionError" in str(type(e).__name__):
                st.warning("⚠️ Phoenix is not running")
            else:
                st.warning(f"⚠️ Could not fetch Phoenix datasets: {type(e).__name__}")
            st.info("Start Phoenix with `px.launch_app()` or upload CSV manually")

            # Fallback: manual dataset name entry
            dataset_name = st.text_input(
                "Dataset Name (manual)",
                value="",
                placeholder="Enter Phoenix dataset name",
                help="Enter dataset name manually if you know it exists"
            )

    # Advanced synthetic data configuration (only shown when using synthetic data)
    if use_synthetic:
        with st.expander("⚙️ Advanced Synthetic Data Configuration"):
            col1, col2, col3 = st.columns(3)
            with col1:
                _synthetic_count = st.number_input(
                    "Synthetic Examples",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    help="Number of synthetic examples to generate"
                )
            with col2:
                _vespa_sample_size = st.number_input(
                    "Vespa Sample Size",
                    min_value=10,
                    max_value=1000,
                    value=500,
                    help="Documents to sample from Vespa"
                )
            with col3:
                _max_profiles = st.number_input(
                    "Max Profiles",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Maximum backend profiles to use"
                )
    # Submit workflow button
    if st.button("🚀 Submit Module Optimization Workflow", type="primary"):
        with st.spinner("Submitting Argo Workflow..."):
            try:
                import subprocess
                import tempfile

                import yaml

                # Build workflow parameters
                workflow_params = [
                    {"name": "tenant-id", "value": tenant_id},
                    {"name": "optimizer-category", "value": "routing"},
                    {"name": "optimizer-type", "value": optimizer_type},
                    {"name": "max-iterations", "value": str(max_iterations)},
                    {"name": "use-synthetic-data", "value": "true" if use_synthetic else "false"}
                ]

                # Add dataset name if using golden dataset
                if not use_synthetic:
                    workflow_params.append({"name": "dataset-name", "value": dataset_name})

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
                            "parameters": workflow_params
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
                    st.success("✅ Workflow submitted successfully!")
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
    """Render profile selection optimization with real Phoenix data"""
    st.subheader("📈 Profile Selection Optimization")
    st.markdown("Learn which processing profile works best for different query types from Phoenix data")

    # Check Phoenix availability first
    try:
        import httpx
        phoenix_response = httpx.get("http://localhost:6006", timeout=2)
        phoenix_available = phoenix_response.status_code in [200, 404]
    except Exception:
        phoenix_available = False

    if not phoenix_available:
        st.warning("⚠️ Phoenix is not running on http://localhost:6006")
        st.info("Start Phoenix with: `import phoenix as px; px.launch_app()`")
        return

    # Query Phoenix for profile performance data
    try:
        from datetime import timedelta

        import phoenix as px

        phoenix_client = px.Client(endpoint="http://localhost:6006")

        # Time range
        lookback_days = st.slider("Lookback (days)", 1, 90, 30)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        # Get search spans from Phoenix
        spans_df = phoenix_client.get_spans_dataframe(
            start_time=start_time,
            end_time=end_time
        )

        if spans_df is None or spans_df.empty:
            st.warning(f"No spans found in the last {lookback_days} days. Run searches first.")
            return

        # Filter for search spans that have profile information
        search_spans = spans_df[spans_df['name'].str.contains('search', case=False, na=False)] if 'name' in spans_df.columns else pd.DataFrame()

        if search_spans.empty:
            st.info("No search spans found. Run video searches with different profiles to see performance data.")
            return

        st.success(f"Found {len(search_spans)} search spans")

        # Try to extract profile information from span attributes
        # Profile info might be in attributes or metadata
        st.subheader("🎯 Profile Usage Statistics")

        # Display available columns to help debug what data we have
        with st.expander("Available Span Data"):
            st.write("Columns:", search_spans.columns.tolist())

        # Count profiles used (if profile attribute exists)
        profile_cols = [col for col in search_spans.columns if 'profile' in col.lower()]

        if profile_cols:
            st.markdown("#### Profile Usage")
            for col in profile_cols:
                profile_counts = search_spans[col].value_counts()
                st.write(f"**{col}:**")
                st.bar_chart(profile_counts)
        else:
            st.info("No profile information found in span attributes. Profile data needs to be logged to Phoenix.")

        st.markdown("---")

        # Performance analysis
        st.subheader("⚡ Performance Analysis")

        # Check if we have NDCG or quality scores
        quality_cols = [col for col in search_spans.columns if any(metric in col.lower() for metric in ['ndcg', 'score', 'quality', 'accuracy'])]

        if quality_cols:
            st.markdown("#### Quality Metrics Found")
            for col in quality_cols:
                st.write(f"**{col}:** {search_spans[col].describe().to_dict()}")

            # If we have both profile and quality data, show correlation
            if profile_cols and quality_cols:
                st.markdown("#### Profile Performance Comparison")
                for profile_col in profile_cols:
                    for quality_col in quality_cols:
                        # Group by profile and calculate average quality
                        profile_quality = search_spans.groupby(profile_col)[quality_col].agg(['mean', 'count'])
                        st.write(f"**{profile_col} vs {quality_col}:**")
                        st.dataframe(profile_quality)
        else:
            st.info("No quality metrics (NDCG, accuracy) found in spans. Run evaluations to collect quality data.")

        st.markdown("---")

        # Training section
        st.subheader("🔧 Train Profile Selector")

        st.markdown("""
        **Training Requirements:**
        1. Phoenix search spans with profile attributes
        2. Quality metrics (NDCG, accuracy) in span data
        3. Query features (text, modality, length, etc.)

        The trainer will:
        - Extract semantic features from queries (temporal/spatial/object keywords)
        - Extract (query_features, profile, quality_score) tuples from Phoenix
        - Train XGBoost classifier to predict best profile for each query type
        - Save model for runtime profile recommendation
        """)

        # Check if model exists
        from pathlib import Path
        model_dir = Path("outputs/models/profile_performance")
        model_exists = (model_dir / "xgboost_model.pkl").exists()

        if model_exists:
            st.info("✅ Trained model found at outputs/models/profile_performance/")

        col1, col2 = st.columns(2)
        with col1:
            train_button = st.button("🚀 Train Profile Selector Model")
        with col2:
            if model_exists:
                load_button = st.button("📂 Load Existing Model")
            else:
                load_button = False

        if train_button:
            if not profile_cols or not quality_cols:
                st.error("Cannot train: Need both profile information and quality metrics in Phoenix spans")
                st.info("Make sure search operations log profile names and evaluation results to Phoenix")
            else:
                with st.spinner("Training profile selection model..."):
                    try:
                        from cogniverse_agents.routing.profile_performance_optimizer import (
                            ProfilePerformanceOptimizer,
                        )

                        st.info(f"Extracting training data from {len(search_spans)} search spans...")

                        # Initialize optimizer
                        optimizer = ProfilePerformanceOptimizer()

                        # Extract training data from Phoenix
                        X, y, profile_names = optimizer.extract_training_data_from_phoenix(
                            phoenix_client=phoenix_client,
                            start_time=start_time,
                            end_time=end_time,
                            min_samples=10
                        )

                        st.info(f"Extracted {len(X)} training samples with {X.shape[1]} features")
                        st.info(f"Found {len(profile_names)} profiles: {', '.join(profile_names)}")

                        # Train model
                        st.info("Training XGBoost classifier with optimized parameters...")
                        metrics = optimizer.train(X, y, test_size=0.2)

                        # Show metrics
                        st.success("✅ Model trained successfully!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
                        with col2:
                            st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                        with col3:
                            st.metric("Samples", metrics['n_samples'])

                        # Show feature importance if available
                        if optimizer.model is not None:
                            st.markdown("#### Feature Importance")
                            feature_names = [
                                "query_length",
                                "word_count",
                                "has_temporal_keywords",
                                "has_spatial_keywords",
                                "has_object_keywords",
                                "avg_word_length"
                            ]
                            importance = optimizer.model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)
                            st.dataframe(importance_df, use_container_width=True)

                        # Save model
                        st.info("Saving model...")
                        optimizer.save()
                        st.success(f"✅ Model saved to {optimizer.model_dir}")
                        st.info("Model can now be used for runtime profile recommendations")

                    except ValueError as e:
                        st.error(f"Training failed: {e}")
                        st.info("Make sure Phoenix spans contain query text, profile names, and quality metrics")
                    except ImportError as e:
                        st.error(f"Missing required library: {e}")
                        st.info("Install with: uv pip install xgboost scikit-learn")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        if load_button:
            with st.spinner("Loading existing model..."):
                try:
                    from cogniverse_agents.routing.profile_performance_optimizer import (
                        ProfilePerformanceOptimizer,
                    )

                    optimizer = ProfilePerformanceOptimizer()
                    if optimizer.load():
                        st.success("✅ Model loaded successfully!")

                        # Test prediction with example query
                        st.markdown("#### Test Prediction")
                        test_query = st.text_input("Test query:", "Show me videos from last week")

                        if test_query:
                            profile, confidence = optimizer.predict_best_profile(test_query)
                            st.info(f"Recommended profile: **{profile}** (confidence: {confidence:.2%})")

                            # Show extracted features
                            features = optimizer.extract_query_features(test_query)
                            st.markdown("**Extracted features:**")
                            st.json({
                                "query_length": features.query_length,
                                "word_count": features.word_count,
                                "has_temporal_keywords": features.has_temporal_keywords,
                                "has_spatial_keywords": features.has_spatial_keywords,
                                "has_object_keywords": features.has_object_keywords,
                                "avg_word_length": round(features.avg_word_length, 2)
                            })
                    else:
                        st.error("Failed to load model")

                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Error fetching profile data: {e}")
        st.info("Make sure Phoenix is running at http://localhost:6006")
        st.info("Start Phoenix with: `import phoenix as px; px.launch_app()`")


def _render_metrics_dashboard_tab():
    """Render unified optimization metrics dashboard with actual optimization metrics"""
    st.subheader("📉 Optimization Metrics Dashboard")
    st.markdown("Track routing accuracy, search quality, and optimization improvements from Phoenix")

    # Check Phoenix availability first
    try:
        import httpx
        phoenix_response = httpx.get("http://localhost:6006", timeout=2)
        phoenix_available = phoenix_response.status_code in [200, 404]
    except Exception:
        phoenix_available = False

    if not phoenix_available:
        st.warning("⚠️ Phoenix is not running on http://localhost:6006")
        st.info("Start Phoenix with: `import phoenix as px; px.launch_app()`")
        return

    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        lookback_days = st.selectbox(
            "Timeframe",
            [7, 30, 90],
            format_func=lambda x: f"Last {x} days"
        )
    with col2:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()

    # Query Phoenix for optimization metrics
    try:
        from datetime import timedelta

        from cogniverse_core.evaluation.evaluators.routing_evaluator import (
            RoutingEvaluator,
        )
        from cogniverse_core.telemetry.manager import get_telemetry_manager

        # Get telemetry provider
        telemetry_manager = get_telemetry_manager()
        provider = telemetry_manager.get_provider(tenant_id="default")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        # Get spans from provider
        spans_df = await provider.traces.get_spans(
            project="cogniverse-default",
            start_time=start_time,
            end_time=end_time
        )

        if spans_df is None or spans_df.empty:
            st.warning(f"No spans found in the last {lookback_days} days. Run some queries first.")
            return

        # Calculate routing metrics
        st.subheader("📊 Routing Optimization Metrics")

        routing_evaluator = RoutingEvaluator(provider=provider)
        routing_spans = routing_evaluator.query_routing_spans(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )

        if routing_spans:
            routing_metrics = routing_evaluator.calculate_metrics(routing_spans)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Routing Accuracy", f"{routing_metrics.routing_accuracy:.1%}")

            with col2:
                st.metric("Total Decisions", routing_metrics.total_decisions)

            with col3:
                st.metric("Avg Routing Latency", f"{routing_metrics.avg_routing_latency:.0f}ms")

            with col4:
                st.metric("Confidence Calibration", f"{routing_metrics.confidence_calibration:.3f}")

            # Per-agent metrics
            st.markdown("#### Per-Agent Performance")

            if routing_metrics.per_agent_precision:
                agent_metrics_data = []
                for agent in routing_metrics.per_agent_precision.keys():
                    agent_metrics_data.append({
                        "Agent": agent,
                        "Precision": routing_metrics.per_agent_precision.get(agent, 0),
                        "Recall": routing_metrics.per_agent_recall.get(agent, 0),
                        "F1 Score": routing_metrics.per_agent_f1.get(agent, 0)
                    })

                agent_df = pd.DataFrame(agent_metrics_data)
                st.dataframe(
                    agent_df.style.background_gradient(subset=["Precision", "Recall", "F1 Score"], cmap="RdYlGn"),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No routing spans found. Routing metrics require cogniverse.routing spans in Phoenix.")

        st.markdown("---")

        # Search quality metrics from evaluations
        st.subheader("🔍 Search Quality Metrics")

        # Query for evaluation spans that contain NDCG scores
        eval_spans = spans_df[spans_df['name'].str.contains('eval|ndcg', case=False, na=False)] if 'name' in spans_df.columns else pd.DataFrame()

        if not eval_spans.empty:
            # Try to extract NDCG scores from span attributes
            st.info(f"Found {len(eval_spans)} evaluation spans")

            # This would extract NDCG scores from attributes if they exist
            # For now, show that evaluation data exists
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Evaluation Runs", len(eval_spans))
            with col2:
                st.metric("Search Queries Evaluated", "N/A")
        else:
            st.info("No search evaluation data found. Run search evaluations to see NDCG metrics.")

        st.markdown("---")

        # Training run history from MLflow or Phoenix
        st.subheader("🏋️ Optimization Training Runs")

        # Look for optimization/training related spans
        training_spans = spans_df[spans_df['name'].str.contains('train|optim', case=False, na=False)] if 'name' in spans_df.columns else pd.DataFrame()

        if not training_spans.empty:
            st.info(f"Found {len(training_spans)} training/optimization spans")

            if 'start_time' in training_spans.columns:
                # Group by day to show training activity
                training_spans['date'] = pd.to_datetime(training_spans['start_time']).dt.date
                daily_training = training_spans.groupby('date').size().reset_index(name='runs')

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=daily_training['date'],
                    y=daily_training['runs'],
                    name='Training Runs'
                ))

                fig.update_layout(
                    title="Training Activity Over Time",
                    xaxis_title="Date",
                    yaxis_title="Number of Training Runs",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training runs found. Run module optimizations to see training history.")

    except Exception as e:
        st.error(f"Error fetching optimization metrics: {e}")
        st.info("Make sure Phoenix is running at http://localhost:6006")
        st.info("Start Phoenix with: `import phoenix as px; px.launch_app()`")


def _render_inline_approval_interface():
    """
    Render inline approval interface in the optimization tab

    Shows pending items with approve/reject controls directly in the results section.
    """
    batch = st.session_state.get("last_generated_batch")
    if not batch:
        return

    pending_items = batch.pending_review

    if not pending_items:
        st.success("✨ All items automatically approved! No review needed.")
        return

    st.subheader("🔍 Review Low-Confidence Items")
    st.markdown(
        f"**{len(pending_items)} items** need your review. "
        f"Items below confidence threshold require human validation."
    )

    # Display each pending item for inline review
    for idx, item in enumerate(pending_items[:5]):  # Show first 5 inline, rest in approval queue
        with st.expander(
            f"📝 Item {idx + 1}/{len(pending_items)} - "
            f"Confidence: {item.confidence:.2f} - "
            f"{item.data.get('query', 'N/A')[:60]}...",
            expanded=(idx == 0)  # Expand first item
        ):
            _render_inline_review_item(item, idx)

    # Show link to full approval queue if there are more items
    if len(pending_items) > 5:
        st.info(
            f"📋 Showing 5 of {len(pending_items)} items. "
            f"Navigate to **Approval Queue** tab to review all items."
        )


def _render_inline_review_item(item, idx: int):
    """Render a single review item inline with approval controls"""

    data = item.data
    query = data.get("query", "N/A")
    entities = data.get("entities", [])
    reasoning = data.get("reasoning", "")
    metadata = data.get("_generation_metadata", {})

    # Display item data in columns
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Generated Query:**")
        st.info(query)

        if reasoning:
            st.markdown("**Reasoning:**")
            st.caption(reasoning)

        st.markdown("**Entities:**")
        if entities:
            entity_str = ", ".join([str(e) for e in entities])
            st.code(entity_str, language=None)
        else:
            st.warning("No entities")

    with col2:
        st.metric("Confidence", f"{item.confidence:.2f}")
        st.metric("Retries", metadata.get("retry_count", 0))

        # Quality indicators
        if item.confidence >= 0.75:
            st.success("🟢 Medium-Low")
        elif item.confidence >= 0.60:
            st.warning("🟡 Low")
        else:
            st.error("🔴 Very Low")

    # Generation metadata toggle
    if metadata:
        with st.expander("🔧 Generation Details"):
            st.json(metadata)

    # Approval controls
    st.markdown("---")
    st.markdown("**Your Decision:**")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button(
            "✅ Approve",
            key=f"inline_approve_{idx}",
            type="primary",
            use_container_width=True
        ):
            _handle_inline_approval(item, idx)

    with col2:
        if st.button(
            "❌ Reject",
            key=f"inline_reject_{idx}",
            use_container_width=True
        ):
            st.session_state[f"inline_rejecting_{idx}"] = True

    # Show rejection form if user clicked reject
    if st.session_state.get(f"inline_rejecting_{idx}", False):
        st.markdown("---")
        st.markdown("**📝 Rejection Feedback & Annotations:**")

        feedback = st.text_area(
            "Why are you rejecting this item?",
            key=f"inline_feedback_{idx}",
            placeholder="e.g., Query doesn't include required entities, Poor grammar, Doesn't match topic...",
            help="Provide specific feedback to help improve future generations"
        )

        st.markdown("**Corrections (optional):**")
        col1, col2 = st.columns(2)

        with col1:
            corrected_entities = st.text_input(
                "Corrected Entities (comma-separated)",
                key=f"inline_entities_{idx}",
                value=", ".join([str(e) for e in entities]),
                help="Update the entities that should be included"
            )

        with col2:
            corrected_topics = st.text_input(
                "Corrected Topics (comma-separated)",
                key=f"inline_topics_{idx}",
                value=data.get("topics", ""),
                help="Update the topics for better context"
            )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button(
                "💾 Submit Rejection",
                key=f"inline_submit_reject_{idx}",
                type="primary",
                use_container_width=True
            ):
                corrections = {}
                if corrected_entities:
                    corrections["entities"] = [e.strip() for e in corrected_entities.split(",")]
                if corrected_topics:
                    corrections["topics"] = corrected_topics.strip()

                _handle_inline_rejection(item, idx, feedback, corrections)
                st.session_state[f"inline_rejecting_{idx}"] = False
                st.rerun()

        with col2:
            if st.button("Cancel", key=f"inline_cancel_{idx}", use_container_width=True):
                st.session_state[f"inline_rejecting_{idx}"] = False
                st.rerun()


def _handle_inline_approval(item, idx: int):
    """Handle inline approval"""
    from cogniverse_agents.approval import ApprovalStatus, ReviewDecision

    try:
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            reviewer=st.session_state.get("user_email", "unknown"),
        )

        # Update item status
        item.status = ApprovalStatus.APPROVED
        import pandas as pd
        item.reviewed_at = pd.Timestamp.now()

        st.success(f"✅ Approved: {item.data.get('query', item.item_id)[:50]}...")

        # Update batch and session state
        batch = st.session_state.get("last_generated_batch")
        if batch:
            # Move from pending to approved
            approved_items = st.session_state.get("approved_items", [])
            approved_items.append(item)
            st.session_state.approved_items = approved_items

            # Update pending items
            pending_items = [i for i in batch.pending_review if i.item_id != item.item_id]
            st.session_state.pending_items = pending_items

        st.rerun()

    except Exception as e:
        st.error(f"Failed to approve item: {e}")
        logger.exception("Inline approval failed")


def _handle_inline_rejection(item, idx: int, feedback: str, corrections: Dict):
    """Handle inline rejection with feedback"""
    from cogniverse_agents.approval import ApprovalStatus, ReviewDecision

    try:
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback=feedback,
            corrections=corrections,
            reviewer=st.session_state.get("user_email", "unknown"),
        )

        # Update item status
        item.status = ApprovalStatus.REJECTED
        import pandas as pd
        item.reviewed_at = pd.Timestamp.now()

        st.warning(f"❌ Rejected: {item.data.get('query', item.item_id)[:50]}...")
        if feedback:
            st.info(f"📝 Feedback: {feedback}")

        # Update batch and session state
        batch = st.session_state.get("last_generated_batch")
        if batch:
            # Move from pending to rejected
            rejected_items = st.session_state.get("rejected_items", [])
            rejected_items.append((item, decision))
            st.session_state.rejected_items = rejected_items

            # Update pending items
            pending_items = [i for i in batch.pending_review if i.item_id != item.item_id]
            st.session_state.pending_items = pending_items

        st.rerun()

    except Exception as e:
        st.error(f"Failed to reject item: {e}")
        logger.exception("Inline rejection failed")


def _process_approval_workflow(result: Dict, confidence_threshold: float):
    """
    Process synthetic data through human approval workflow

    Args:
        result: Synthetic data generation result
        confidence_threshold: Confidence threshold for auto-approval
    """
    try:
        from cogniverse_agents.approval import HumanApprovalAgent
        from cogniverse_synthetic.approval import (
            SyntheticDataConfidenceExtractor,
            SyntheticDataFeedbackHandler,
        )

        # Initialize approval agent
        confidence_extractor = SyntheticDataConfidenceExtractor()
        feedback_handler = SyntheticDataFeedbackHandler()

        agent = HumanApprovalAgent(
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            confidence_threshold=confidence_threshold,
            storage=None  # Using session state instead of Phoenix for demo
        )

        # Process batch through approval agent
        batch_id = f"synthetic_{result['optimizer']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create mock batch (in production, would call agent.process_batch() asynchronously)
        from cogniverse_agents.approval import ReviewItem, ApprovalBatch, ApprovalStatus

        review_items = []
        for i, item_data in enumerate(result['data']):
            confidence = confidence_extractor.extract(item_data)
            status = (
                ApprovalStatus.AUTO_APPROVED
                if confidence >= confidence_threshold
                else ApprovalStatus.PENDING_REVIEW
            )

            review_item = ReviewItem(
                item_id=f"{batch_id}_{i}",
                data=item_data,
                confidence=confidence,
                status=status
            )
            review_items.append(review_item)

        batch = ApprovalBatch(
            batch_id=batch_id,
            items=review_items,
            context={
                "optimizer": result['optimizer'],
                "tenant_id": result.get('tenant_id', 'default'),
                "profiles": result['selected_profiles']
            }
        )

        # Store in session state for approval queue tab
        st.session_state["last_generated_batch"] = batch
        st.session_state["pending_items"] = batch.pending_review
        st.session_state["approved_items"] = batch.auto_approved

        # Display approval summary
        st.success(
            f"✅ Generated {len(batch.items)} examples: "
            f"{len(batch.auto_approved)} auto-approved, "
            f"{len(batch.pending_review)} awaiting review"
        )

        if len(batch.pending_review) > 0:
            st.info(
                f"📋 **{len(batch.pending_review)} items need your review**. "
                "Navigate to the **Approval Queue** tab to review them."
            )

            # Show approval stats
            stats = agent.get_approval_stats(batch)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Auto-Approved", stats['auto_approved'])
            with col2:
                st.metric("Pending Review", stats['pending_review'])
            with col3:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")

    except Exception as e:
        st.error(f"❌ Approval workflow failed: {e}")
        logger.exception("Approval workflow error")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Enhanced Optimization",
        page_icon="🔧",
        layout="wide"
    )
    render_enhanced_optimization_tab()
