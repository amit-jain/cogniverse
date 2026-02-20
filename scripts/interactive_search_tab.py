#!/usr/bin/env python3
"""
Interactive Search Tab

Live search testing and evaluation with multiple ranking strategies
and real-time results, routed through the Video Search Agent.
"""

import time

import httpx
import streamlit as st


def render_interactive_search_tab(agent_status: dict):
    """Render the interactive search tab interface."""
    st.header("🔍 Interactive Search Interface")
    st.markdown("Live search testing and evaluation with multiple ranking strategies and real-time results.")

    # Search Interface
    st.subheader("🔎 Search Interface")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g. basketball dunk, person throwing discus, game highlights",
            help="Enter natural language queries to search through ingested videos",
        )

    with col2:
        # Check if video search agent is available
        video_search_agent_available = (
            "error" not in agent_status
            and agent_status.get("Video Search Agent", {}).get("status") == "online"
        )
        search_button_disabled = not search_query or not video_search_agent_available

        if not video_search_agent_available:
            st.warning("🔧 Video Search Agent is offline")
        else:
            st.success("Agent connected")

        search_button = st.button("🔍 Search", type="primary", disabled=search_button_disabled)

    # Search Configuration
    st.subheader("⚙️ Search Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_profile = st.selectbox(
            "Processing Profile",
            ["video_colpali_smol500_mv_frame", "video_colqwen_omni_mv_chunk_30s",
             "video_videoprism_base_mv_chunk_30s", "video_videoprism_lvt_base_sv_chunk_6s"],
            help="Select the video processing profile for search",
        )

    with col2:
        ranking_strategies = st.multiselect(
            "Ranking Strategies",
            ["binary_binary", "float_float", "binary_float", "float_binary"],
            default=["binary_binary", "float_float"],
            help="Compare different ranking strategies",
        )

    with col3:
        top_k = st.slider("Number of Results", 1, 20, 5)
        _confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Search Results
    if search_button and search_query:
        st.subheader("🎯 Search Results")

        # Get agent URL
        video_search_url = agent_status.get("Video Search Agent", {}).get("url", "")

        with st.spinner(f"Searching for '{search_query}' via Video Search Agent..."):
            start_time = time.time()
            results = _call_agent_search(
                agent_url=video_search_url,
                query=search_query,
                profile=selected_profile,
                strategies=ranking_strategies,
                top_k=top_k,
            )
            elapsed_ms = (time.time() - start_time) * 1000

        if results is None or results.get("status") == "error":
            error_msg = results.get("message", "Unknown error") if results else "Request failed"
            st.error(f"Search error: {error_msg}")
        else:
            result_count = results.get("results_count", 0)
            result_list = results.get("results", [])

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Results", result_count)
            with m2:
                st.metric("Latency", f"{elapsed_ms:.0f}ms")
            with m3:
                st.metric("Profile", results.get("profile", selected_profile))

            if not result_list:
                st.info("No results found for this query.")
            else:
                for i, result in enumerate(result_list):
                    metadata = result.get("metadata", {})
                    temporal = result.get("temporal_info", {})
                    video_id = metadata.get("video_id", "unknown")
                    score = result.get("score", 0.0)
                    doc_id = result.get("document_id", "")
                    start_t = temporal.get("start_time", 0)
                    end_t = temporal.get("end_time", 0)

                    with st.expander(f"Result #{i+1} — {doc_id} (score: {score:.4f})", expanded=(i < 3)):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**Video ID:** `{video_id}`")
                            st.markdown(f"**Segment:** {metadata.get('segment_id', '?')}")
                            st.markdown(f"**Time:** {start_t:.2f}s — {end_t:.2f}s")
                        with col_b:
                            st.markdown(f"**Score:** {score:.6f}")
                            st.markdown(f"**Profile:** {selected_profile}")
                            st.markdown(f"**Schema:** {metadata.get('sddocname', 'N/A')}")

                            # Relevance annotation
                            _relevance = st.radio(
                                "Relevance",
                                ["Highly Relevant", "Relevant", "Not Relevant"],
                                key=f"relevance_{i}",
                                horizontal=True,
                            )
                            if st.button("💾 Save Annotation", key=f"save_{i}"):
                                st.success(f"Annotation saved for result #{i+1}")

            # Store search in session state for statistics
            if "search_history" not in st.session_state:
                st.session_state["search_history"] = []
            st.session_state["search_history"].append({
                "query": search_query,
                "results_count": result_count,
                "latency_ms": elapsed_ms,
                "profile": selected_profile,
            })
    else:
        st.info("👆 Enter a query and click Search to see results")

    # Statistics
    st.markdown("---")
    st.subheader("📊 Search Statistics")

    history = st.session_state.get("search_history", [])
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Searches", len(history))
    with stats_col2:
        st.metric("Annotations", "0")
    with stats_col3:
        if history:
            avg_lat = sum(h["latency_ms"] for h in history) / len(history)
            st.metric("Avg. Latency", f"{avg_lat:.0f}ms")
        else:
            st.metric("Avg. Latency", "N/A")

    # Documentation
    with st.expander("ℹ️ About Interactive Search"):
        st.markdown("""
        This tab provides an interactive interface for testing video search
        via the Video Search Agent.

        **Features:**
        - Real-time search with natural language queries
        - Multiple ranking strategies comparison
        - Relevance annotation for search results
        - Performance metrics and statistics

        **Ranking Strategies:**
        - `binary_binary`: Binary query, binary document embeddings
        - `float_float`: Float query, float document embeddings
        - `binary_float`: Binary query, float document embeddings
        - `float_binary`: Float query, binary document embeddings
        """)


def _call_agent_search(
    agent_url: str,
    query: str,
    profile: str,
    strategies: list,
    top_k: int,
) -> dict:
    """Call Video Search Agent's /search endpoint."""
    try:
        response = httpx.post(
            f"{agent_url}/search",
            json={
                "query": query,
                "profile": profile,
                "strategies": strategies,
                "top_k": top_k,
            },
            timeout=60.0,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
