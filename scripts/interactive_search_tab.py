#!/usr/bin/env python3
"""
Interactive Search Tab

Live search testing via the unified Runtime search API.
Session-aware with cross-session conversational memory.
"""

import time
import uuid

import httpx
import streamlit as st

# Search strategies available in Vespa ColPali profiles
SEARCH_STRATEGIES = [
    "float_float",
    "binary_binary",
    "float_binary",
    "default",
    "phased",
    "hybrid_float_bm25",
    "hybrid_binary_bm25",
    "bm25_only",
]


def render_interactive_search_tab(agent_status: dict):
    """Render the interactive search tab interface."""
    st.header("Interactive Search Interface")
    st.markdown("Search via unified Runtime API (`POST /search/`).")

    if "current_tenant" not in st.session_state:
        st.error("No tenant selected. Set an Active Tenant in the sidebar first.")
        return

    # Generate session_id for conversational memory (persists per browser session)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    # Determine runtime URL from session state (set by app.py from SystemConfig)
    runtime_url = st.session_state.get("runtime_url", "")
    if not runtime_url:
        for agent_info in agent_status.values():
            if isinstance(agent_info, dict) and agent_info.get("status") == "online":
                runtime_url = agent_info.get("url", "")
                break

    runtime_available = bool(runtime_url)

    # Search Interface
    st.subheader("Search Interface")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g. basketball dunk, person throwing discus, game highlights",
            help="Enter natural language queries to search through ingested videos",
        )

    with col2:
        if runtime_available:
            st.success("Runtime connected")
        else:
            st.warning("Runtime not available")

        st.button(
            "Search", type="primary", disabled=not search_query or not runtime_available,
            on_click=lambda: st.session_state.update(trigger_search=True),
        )

    # Search Configuration
    st.subheader("Search Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of Results", 1, 20, 5)

    with col2:
        # Profile — use the tenant's deployed schema
        current_tenant = st.session_state["current_tenant"]
        profile = st.text_input(
            "Profile",
            value="video_colpali_smol500_mv_frame",
            help="Search profile (schema name without tenant suffix)",
        )

    with col3:
        strategy = st.selectbox("Strategy", SEARCH_STRATEGIES, index=0)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.text(f"Session: {st.session_state['session_id'][:8]}...")
    with col_s2:
        if st.button("New Session"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()

    # Search Results — execute search and store in session state
    search_button = st.session_state.pop("trigger_search", False)
    if search_button and search_query:
        # Clear stale results before executing new search
        st.session_state.pop("search_results", None)
        current_tenant = st.session_state["current_tenant"]
        session_id = st.session_state["session_id"]

        with st.spinner(f"Searching for '{search_query}'..."):
            start_time = time.time()
            results = _call_runtime_search(
                runtime_url=runtime_url,
                query=search_query,
                top_k=top_k,
                profile=profile,
                strategy=strategy,
                tenant_id=current_tenant,
                session_id=session_id,
            )
            elapsed_ms = (time.time() - start_time) * 1000

        st.session_state["search_results"] = results
        st.session_state["search_elapsed_ms"] = elapsed_ms
        st.session_state["search_query_text"] = search_query

    # Display search results from session state (survives Streamlit rerenders)
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        elapsed_ms = st.session_state.get("search_elapsed_ms", 0)
        st.subheader("Search Results")

        if results is None or results.get("status") == "error":
            error_msg = (
                results.get("message", "Unknown error") if results else "Request failed"
            )
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
                st.metric("Profile", results.get("profile", "auto"))

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

                    with st.expander(
                        f"Result #{i + 1} — {doc_id} (score: {score:.4f})",
                        expanded=(i < 3),
                    ):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**Video ID:** `{video_id}`")
                            st.markdown(
                                f"**Segment:** {metadata.get('segment_id', '?')}"
                            )
                            st.markdown(f"**Time:** {start_t:.2f}s — {end_t:.2f}s")
                        with col_b:
                            st.markdown(f"**Score:** {score:.6f}")
                            st.markdown(f"**Profile:** {result.get('profile', 'auto')}")
                            st.markdown(
                                f"**Schema:** {metadata.get('sddocname', 'N/A')}"
                            )

                            # Relevance annotation
                            _relevance = st.radio(
                                "Relevance",
                                ["Highly Relevant", "Relevant", "Not Relevant"],
                                key=f"relevance_{i}",
                                horizontal=True,
                            )
                            if st.button("💾 Save Annotation", key=f"save_{i}"):
                                if "search_annotations" not in st.session_state:
                                    st.session_state["search_annotations"] = []
                                st.session_state["search_annotations"].append(
                                    {
                                        "query": search_query,
                                        "result_rank": i + 1,
                                        "video_id": video_id,
                                        "doc_id": doc_id,
                                        "score": score,
                                        "relevance": _relevance,
                                    }
                                )
                                st.success(
                                    f"Annotation saved for result #{i + 1}: {_relevance}"
                                )

                                # Persist to Phoenix via runtime for optimization
                                try:
                                    import httpx as _httpx

                                    _httpx.post(
                                        f"{runtime_url}/agents/routing_agent/process",
                                        json={
                                            "agent_name": "routing_agent",
                                            "query": search_query,
                                            "context": {
                                                "tenant_id": "default",
                                                "action": "optimize_routing",
                                                "examples": [
                                                    {
                                                        "query": search_query,
                                                        "chosen_agent": "search_agent",
                                                        "confidence": score,
                                                        "search_quality": (
                                                            0.9
                                                            if _relevance
                                                            == "Highly Relevant"
                                                            else 0.5
                                                            if _relevance == "Relevant"
                                                            else 0.1
                                                        ),
                                                        "agent_success": _relevance
                                                        != "Not Relevant",
                                                    }
                                                ],
                                            },
                                        },
                                        timeout=10.0,
                                    )
                                except Exception:
                                    pass  # Non-blocking

            # Summarize Results (Streaming)
            if result_count > 0 and st.button("📝 Summarize Results (Streaming)"):
                from phoenix_dashboard_standalone import display_streaming_result

                descriptions = []
                for item in results.get("results", [])[:5]:
                    descriptions.append(
                        f"{item.get('video_id', 'unknown')}: {item.get('description', '')}"
                    )
                summary_query = (
                    f"Summarize search results for '{search_query}': "
                    + "; ".join(descriptions[:10])
                )
                st.subheader("📄 Summary")
                final = display_streaming_result(
                    agent_name="summarizer_agent",
                    query=summary_query,
                )
                if final and "summary" in final:
                    st.markdown("### Key Points")
                    for point in final.get("key_points", []):
                        st.markdown(f"- {point}")

            # Store search in session state for statistics
            if "search_history" not in st.session_state:
                st.session_state["search_history"] = []
            st.session_state["search_history"].append(
                {
                    "query": search_query,
                    "results_count": result_count,
                    "latency_ms": elapsed_ms,
                    "profile": results.get("profile", "auto"),
                }
            )
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
        annotation_count = len(st.session_state.get("search_annotations", []))
        st.metric("Annotations", str(annotation_count))
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
        routed through the OrchestratorAgent with session-aware conversational memory.

        **Features:**
        - Real-time search with natural language queries
        - Automatic profile selection via ProfileSelectionAgent
        - Cross-session conversational memory via MemoryAwareMixin
        - Relevance annotation for search results
        - Performance metrics and statistics

        **Search Flow:**
        1. Query sent to OrchestratorAgent (or direct to Video Search Agent as fallback)
        2. OrchestratorAgent plans execution: query enhancement, entity extraction, profile selection, search
        3. Results aggregated and returned with profile and latency metadata
        """)


def _call_runtime_search(
    runtime_url: str,
    query: str,
    top_k: int,
    profile: str,
    strategy: str,
    tenant_id: str = "default",
    session_id: str | None = None,
) -> dict:
    """Call the unified Runtime's POST /search/ endpoint."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "profile": profile,
            "strategy": strategy,
            "tenant_id": tenant_id,
            "session_id": session_id,
        }
        response = httpx.post(
            f"{runtime_url}/search/",
            json=payload,
            timeout=120.0,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}",
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
