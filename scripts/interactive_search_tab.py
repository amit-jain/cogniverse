#!/usr/bin/env python3
"""
Interactive Search Tab

Live search testing and evaluation with multiple ranking strategies
and real-time results.
"""

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
            help="Enter natural language queries to search through ingested videos"
        )
    
    with col2:
        # Check if video search agent is available
        video_search_agent_available = (
            "error" not in agent_status and
            agent_status.get("Video Search Agent", {}).get("status") == "online"
        )
        search_button_disabled = not search_query or not video_search_agent_available

        if not video_search_agent_available:
            st.warning("🔧 Video Search Agent is offline")

        search_button = st.button("🔍 Search", type="primary", disabled=search_button_disabled)
    
    # Search Configuration
    st.subheader("⚙️ Search Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_profile = st.selectbox(
            "Processing Profile",
            ["video_colpali_smol500_mv_frame", "video_colqwen_omni_mv_chunk_30s",
             "video_videoprism_base_mv_chunk_30s", "video_videoprism_lvt_base_sv_chunk_6s"],
            help="Select the video processing profile for search"
        )
    
    with col2:
        ranking_strategies = st.multiselect(
            "Ranking Strategies",
            ["binary_binary", "float_float", "binary_float", "float_binary"],
            default=["binary_binary", "float_float"],
            help="Compare different ranking strategies"
        )
    
    with col3:
        top_k = st.slider("Number of Results", 1, 20, 5)
        _confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    # Search Results
    if search_button and search_query:
        st.subheader("🎯 Search Results")
        
        with st.spinner(f"Searching for '{search_query}'..."):
            st.info("💡 This is a placeholder. Integration with actual search backend pending.")
            
            # Placeholder results
            st.markdown("### Results Preview")
            for i in range(min(top_k, 3)):
                with st.expander(f"Result #{i+1} - Video {i+1}"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.markdown("**Frame Preview**")
                        st.info(f"Frame {i*10}")
                    with col_b:
                        st.markdown(f"**Match Score:** {(1.0 - i*0.1):.2f}")
                        st.markdown(f"**Profile:** {selected_profile}")
                        st.markdown(f"**Strategy:** {ranking_strategies[0] if ranking_strategies else 'N/A'}")
                        
                        # Relevance annotation
                        _relevance = st.radio(
                            "Relevance",
                            ["Highly Relevant", "Relevant", "Not Relevant"],
                            key=f"relevance_{i}",
                            horizontal=True
                        )
                        if st.button("💾 Save Annotation", key=f"save_{i}"):
                            st.success(f"✅ Annotation saved for result #{i+1}")
    else:
        st.info("👆 Enter a query and click Search to see results")
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Search Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Searches", "0")
    with stats_col2:
        st.metric("Annotations", "0")
    with stats_col3:
        st.metric("Avg. Latency", "N/A")
    
    # Documentation
    with st.expander("ℹ️ About Interactive Search"):
        st.markdown("""
        This tab provides an interactive interface for testing video search.
        
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
