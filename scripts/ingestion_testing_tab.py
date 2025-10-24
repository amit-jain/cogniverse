#!/usr/bin/env python3
"""
Ingestion Testing Tab

Interactive testing and configuration of video ingestion pipelines
with different processing profiles.
"""

import streamlit as st


def render_ingestion_testing_tab(agent_status: dict):
    """Render the ingestion testing tab interface."""
    st.header("📥 Ingestion Pipeline Testing")
    st.markdown("Interactive testing and configuration of video ingestion pipelines with different processing profiles.")
    
    # Video Upload Section
    st.subheader("🎬 Video Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_video = st.file_uploader(
            "Upload test video for ingestion",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video file to test different ingestion configurations"
        )
    
    with col2:
        st.markdown("**Processing Profiles:**")
        selected_profiles = st.multiselect(
            "Select profiles to test",
            ["video_colpali_smol500_mv_frame", "video_colqwen_omni_mv_chunk_30s", 
             "video_videoprism_base_mv_chunk_30s", "video_videoprism_large_mv_chunk_30s",
             "video_videoprism_lvt_base_sv_chunk_6s", "video_videoprism_lvt_large_sv_chunk_6s"],
            default=["video_colpali_smol500_mv_frame"]
        )
    
    # Pipeline Configuration
    if uploaded_video:
        st.subheader("⚙️ Pipeline Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_frames = st.slider("Max Frames per Video", 1, 50, 10)
            _chunk_duration = st.slider("Chunk Duration (s)", 5, 60, 30)

        with col2:
            _enable_transcription = st.checkbox("Enable Audio Transcription", True)
            _enable_descriptions = st.checkbox("Enable Frame Descriptions", True)

        with col3:
            _keyframe_method = st.selectbox("Keyframe Extraction", ["fps", "scene_detection", "uniform"])
            _embedding_precision = st.selectbox("Embedding Precision", ["float32", "binary"])
        
        # Process Video Button
        video_processing_agent_available = (
            "error" not in agent_status and
            agent_status.get("Video Processing Agent", {}).get("status") == "online"
        )
        process_button_disabled = not selected_profiles or not video_processing_agent_available

        if not video_processing_agent_available:
            st.warning("🔧 Video Processing Agent is offline")

        if st.button("🚀 Process Video", type="primary", disabled=process_button_disabled):
            with st.spinner(f"Processing video with {len(selected_profiles)} profile(s)..."):
                st.info("💡 This is a placeholder. Integration with actual ingestion pipeline pending.")
                
                # Placeholder progress bar
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                st.success(f"✅ Video processed with {len(selected_profiles)} profile(s)")
                
                # Results
                st.subheader("📊 Processing Results")
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Frames Extracted", max_frames)
                with result_col2:
                    st.metric("Embeddings Generated", max_frames * len(selected_profiles))
                with result_col3:
                    st.metric("Storage Used", f"{max_frames * len(selected_profiles) * 0.5:.1f} MB")
    else:
        st.info("👆 Upload a video file to start testing the ingestion pipeline")
    
    # Pipeline Status
    st.subheader("📈 Pipeline Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**Recent Jobs**")
        st.info("No recent jobs. Process a video to see results here.")
    
    with status_col2:
        st.markdown("**Agent Status**")
        if agent_status.get("Video Search Agent", {}).get("status") == "online":
            st.success("✅ Backend: Healthy")
        else:
            st.error("❌ Backend: Offline")
    
    # Documentation
    st.markdown("---")
    with st.expander("ℹ️ About Ingestion Testing"):
        st.markdown("""
        This tab provides an interactive way to test video ingestion pipelines.
        
        **Features:**
        - Upload and process videos with multiple profiles
        - Configure keyframe extraction, transcription, and embeddings
        - Monitor processing progress in real-time
        - View ingestion results and metrics
        
        **Processing Profiles:**
        - `video_colpali_smol500_mv_frame`: ColPali model, frame-based
        - `video_colqwen_omni_mv_chunk_30s`: ColQwen Omni, 30s chunks
        - `video_videoprism_base_mv_chunk_30s`: VideoPrism base, 30s chunks
        - `video_videoprism_lvt_base_sv_chunk_6s`: VideoPrism LVT, 6s chunks
        """)
    
    st.subheader("📊 System Status")
    st.info("System status is displayed in the sidebar based on real agent connectivity checks.")
