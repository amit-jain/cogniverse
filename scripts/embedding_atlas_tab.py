"""
Embedding Atlas visualization tab for Dashboard
"""

# Fix protobuf issue - must be before other imports
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import search backend
try:
    from src.backends.vespa.search_backend import VespaSearchBackend as SearchBackend
    VESPA_AVAILABLE = True
except ImportError:
    VESPA_AVAILABLE = False


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_videos() -> Dict[str, Dict]:
    """
    Query backend to get available videos with metadata
    Returns dict: {video_id: {frame_count, duration, strategy, profile}}
    """
    if not VESPA_AVAILABLE:
        return {}

    try:
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )
        config_manager = create_default_config_manager()
        config = get_config(tenant_id="default", config_manager=config_manager)
        backend = SearchBackend(
            url=config.get("backend_url", "http://localhost"),
            port=config.get("backend_port", 8080),
            schema=config.get("schema_name", "video_frame")
        )
        
        # Query to get unique videos with aggregation
        yql = """
        select video_id, video_title, source_type, 
               count(*) as frame_count,
               max(timestamp) as duration
        from video_frame 
        where true
        limit 0 | 
        all(group(video_id) max(100) each(
            output(count() as frame_count)
            max(1) each(output(summary()))
        ))
        """
        
        # Simpler query if aggregation fails
        simple_yql = "select video_id, video_title, timestamp, frame_number from video_frame where true limit 1000"
        
        try:
            response = backend.app.query(yql=yql)
        except Exception:
            # Fallback to simple query
            response = backend.app.query(yql=simple_yql)
        
        videos = {}
        
        # Process response to extract unique videos
        if hasattr(response, 'hits'):
            video_data = {}
            for hit in response.hits:
                fields = hit.get("fields", {})
                vid = fields.get("video_id")
                if vid and vid not in video_data:
                    video_data[vid] = {
                        "frame_count": 1,
                        "duration": fields.get("timestamp", 0),
                        "title": fields.get("video_title", vid),
                        "strategy": fields.get("ranking_strategy", "default"),
                        "profile": fields.get("search_profile", "unknown")
                    }
                elif vid:
                    video_data[vid]["frame_count"] += 1
                    video_data[vid]["duration"] = max(
                        video_data[vid]["duration"],
                        fields.get("timestamp", 0)
                    )
            
            videos = video_data
        
        return videos
    
    except Exception as e:
        st.error(f"Error fetching videos from backend: {str(e)}")
        return {}


@st.cache_data(ttl=300)
def get_available_strategies() -> Dict[str, list]:
    """
    Get available search strategies/profiles from backend
    Returns dict: {strategy: [video_ids]}
    """
    if not VESPA_AVAILABLE:
        return {}
    
    try:
        videos = get_available_videos()
        
        # Group by strategy
        strategies = {}
        for video_id, info in videos.items():
            strategy = info.get("strategy", "default")
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(video_id)
        
        return strategies
    
    except Exception:
        return {}


def render_embedding_atlas_tab():
    """Render the embedding atlas visualization tab"""
    
    # Introduction
    st.markdown("""
    ### Export and Visualize Embeddings
    
    **What gets exported:**
    • High-dimensional embeddings  
    • Metadata (titles, descriptions, timestamps)  
    • Automatically reduced to 2D/3D for visualization  
    • Clustering applied for pattern discovery
    
    **Insights you'll get:**
    • Clustering patterns in your content  
    • Semantic relationships between items  
    • Distribution and coverage analysis
    """)
    
    # Create columns for controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📥 Export Settings")
        
        # Schema selection - read from configs
        import json
        from pathlib import Path
        
        # Get available schemas from config directory
        schema_dir = Path("configs/schemas")
        available_schemas = []
        
        if schema_dir.exists():
            for schema_file in schema_dir.glob("*_schema.json"):
                if schema_file.name != "ranking_strategies.json":
                    try:
                        with open(schema_file, 'r') as f:
                            schema_data = json.load(f)
                            schema_name = schema_data.get("name", schema_file.stem.replace("_schema", ""))
                            available_schemas.append(schema_name)
                    except Exception:
                        pass

        # Default to video_frame if no schemas found
        if not available_schemas:
            available_schemas = ["video_frame"]
        
        selected_schema = st.selectbox(
            "Schema",
            sorted(available_schemas),
            help="Select the schema to export embeddings from"
        )
        
        # Profile selection - CRITICAL for query encoding
        available_profiles = [
            "frame_based_colpali",
            "direct_video_frame", 
            "direct_video_frame_large",
            "colqwen_chunks"
        ]
        
        selected_profile = st.selectbox(
            "Encoder Profile",
            available_profiles,
            help="IMPORTANT: Select the profile that was used to encode these documents"
        )
        
        # Export options
        export_type = st.selectbox(
            "Export Type",
            ["Sample (Fast)", "Filtered", "Full Dataset"],
            help="Choose what to export from backend"
        )
        
        if export_type == "Sample (Fast)":
            max_docs = st.number_input(
                "Number of documents",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of random documents to sample"
            )
        elif export_type == "Filtered":
            # Video selection with dropdown
            video_selection_mode = st.radio(
                "Video Selection",
                ["Browse Available", "Manual Entry"],
                horizontal=True,
                help="Browse videos from backend or enter manually"
            )
            
            if video_selection_mode == "Browse Available":
                # Get available videos from backend
                available_videos = get_available_videos()
                available_strategies = get_available_strategies()
                
                if available_videos:
                    # First, let user select filtering approach
                    filter_by = st.radio(
                        "Filter by",
                        ["Individual Video", "Strategy/Profile", "All Videos"],
                        horizontal=True
                    )
                    
                    if filter_by == "Individual Video":
                        # Create formatted options with metadata
                        video_options = []
                        video_map = {}
                        
                        for vid, info in available_videos.items():
                            display_name = f"{info.get('title', vid)} ({info.get('frame_count', 0)} frames)"
                            video_options.append(display_name)
                            video_map[display_name] = vid
                        
                        selected_display = st.selectbox(
                            "Select Video",
                            sorted(video_options),
                            help="Choose from videos indexed in backend"
                        )
                        
                        video_id = video_map.get(selected_display)
                        
                        # Show video metadata
                        if video_id and video_id in available_videos:
                            video_info = available_videos[video_id]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frames", video_info.get('frame_count', 'N/A'))
                            with col2:
                                st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
                            with col3:
                                st.metric("Strategy", video_info.get('strategy', 'default'))
                    
                    elif filter_by == "Strategy/Profile":
                        if available_strategies:
                            selected_strategy = st.selectbox(
                                "Select Strategy/Profile",
                                sorted(available_strategies.keys()),
                                help="Filter by indexing strategy or search profile"
                            )
                            
                            # Show videos in this strategy
                            strategy_videos = available_strategies.get(selected_strategy, [])
                            st.info(f"Found {len(strategy_videos)} videos with strategy: {selected_strategy}")
                            
                            # Option to select specific video from strategy
                            video_in_strategy = st.selectbox(
                                "Select specific video (optional)",
                                ["All in Strategy"] + sorted(strategy_videos)
                            )
                            
                            if video_in_strategy != "All in Strategy":
                                video_id = video_in_strategy
                            else:
                                # We'll handle multiple videos in export
                                video_id = f"strategy:{selected_strategy}"
                        else:
                            st.warning("No strategy information available")
                            video_id = None
                    
                    else:  # All Videos
                        video_id = None
                        total_videos = len(available_videos)
                        total_frames = sum(v.get('frame_count', 0) for v in available_videos.values())
                        st.info(f"Will export from all {total_videos} videos ({total_frames} total frames)")
                    
                else:
                    st.warning("Could not fetch video list from backend")
                    video_id = st.text_input(
                        "Video ID (manual)",
                        placeholder="e.g., sample_video_001",
                        help="Enter video ID manually"
                    )
            else:
                video_id = st.text_input(
                    "Video ID (optional)",
                    placeholder="e.g., sample_video_001",
                    help="Filter by specific video ID"
                )
            
            query = st.text_input(
                "Search Query (optional)",
                placeholder="e.g., outdoor sports",
                help="Filter by search query"
            )
            
            max_docs = st.number_input(
                "Max documents",
                min_value=100,
                max_value=50000,
                value=5000,
                step=500
            )
        else:  # Full Dataset
            max_docs = st.number_input(
                "Max documents (safety limit)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Maximum documents to export (for safety)"
            )
    
    with col2:
        st.subheader("📋 Schema Fields")
        
        # Show fields that will be exported based on selected schema
        if selected_schema == "video_frame":
            st.info("""
            **Fields exported:**
            • video_id, frame_number
            • video_title, frame_description  
            • timestamp, duration
            • embedding vectors
            • relevance_score (if available)
            """)
        else:
            # Generic fields for other schemas
            st.info("""
            **Fields exported:**
            • Document ID
            • Title/description fields
            • Embedding vectors
            • Metadata fields
            • Relevance scores
            """)
    
    with col3:
        st.subheader("🚀 Actions")
        
        # Export button
        if st.button("🔄 Export & Visualize", type="primary", use_container_width=True):
            export_and_visualize(
                schema=selected_schema,
                profile=selected_profile,
                export_type=export_type,
                max_docs=max_docs,
                video_id=video_id if export_type == "Filtered" else None,
                query=query if export_type == "Filtered" else None
            )
        
        # Open external viewer - directly launch if we have data
        if "embedding_data" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                viewer_type = st.selectbox(
                    "Visualization Type",
                    ["Embedding Atlas", "Custom 3D Visualization"],
                    help="Choose visualization library"
                )
            with col2:
                if st.button("🌐 Open Visualization", use_container_width=True):
                    if viewer_type == "Embedding Atlas":
                        open_atlas_viewer("apple")
                    else:
                        open_atlas_viewer("custom")
    
    # Display area for results
    st.markdown("---")
    
    # Check for existing exports
    display_existing_exports()


def export_and_visualize(
    schema: str,
    profile: str,
    export_type: str,
    max_docs: int,
    video_id: str = None,
    query: str = None
):
    """Export embeddings from backend and prepare for visualization"""
    
    # Construct tenant-scoped schema name
    # Documents live in {base_schema}_{tenant_id} where ':' is replaced with '_'
    current_tenant = st.session_state.get("current_tenant", "default")
    tenant_suffix = current_tenant.replace(":", "_")
    tenant_schema = f"{schema}_{tenant_suffix}"

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"vespa_embeddings_{timestamp}.parquet"

    # Set reduction method
    reduction_method = "umap"  # Default to UMAP for dimension reduction

    # Build export command using backend abstraction
    cmd = [
        "uv", "run", "python", "scripts/export_backend_embeddings.py",
        "--backend", "vespa",  # Can be made configurable
        "--backend-url", "http://localhost:8080",
        "--output", str(output_file),
        "--schema", tenant_schema,
        "--profile", profile,  # Always pass the profile
        "--max-documents", str(max_docs),
        "--reduction-method", reduction_method
        # Note: dimension reduction is enabled by default (no --no-reduction flag)
    ]
    
    if video_id:
        cmd.extend(["--filter-key", "video_id", "--filter-value", video_id])
    
    # Note: query-based filtering would need to be handled differently
    # For now, we'll skip text query filtering in export
    
    # Show progress
    with st.spinner(f"Exporting {max_docs} documents from backend..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run export
            status_text.text("Downloading embeddings...")
            progress_bar.progress(20)
            
            # Run export without timeout by default (wait for completion)
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                st.error(f"Export failed: {error_msg}")
                # Check common issues
                if "Connection refused" in error_msg or "Failed to connect" in error_msg:
                    st.warning("⚠️ Backend appears to be offline. Please ensure backend is running on http://localhost:8080")
                return
            
            progress_bar.progress(60)
            status_text.text("Processing embeddings...")
            
            # Load the exported data
            df = pd.read_parquet(output_file)
            
            progress_bar.progress(80)
            status_text.text("Preparing visualization...")
            
            # Store in session state
            st.session_state.embedding_data = {
                "df": df,
                "file": str(output_file),
                "timestamp": timestamp,
                "method": reduction_method,
                "n_docs": len(df)
            }
            # Profile is already stored in the dataframe itself via export script
            
            progress_bar.progress(100)
            status_text.text("✅ Export complete!")
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Successfully exported {len(df)} documents to {output_file.name}")
            
            # Show export statistics in a cleaner format
            st.markdown("### 📊 Export Statistics")
            
            # Use wider columns for better display
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **📄 Documents Exported:** {len(df):,}  
                **🎬 Unique Videos:** {df['video_id'].nunique() if 'video_id' in df.columns else 0:,}
                """)
            with col2:
                cluster_count = df['cluster'].nunique() if 'cluster' in df.columns else 0
                embed_dim = len(df.iloc[0]['embedding']) if 'embedding' in df.columns and len(df) > 0 else 0
                st.info(f"""
                **🎯 Clusters Generated:** {cluster_count:,}  
                **📏 Embedding Dimensions:** {embed_dim:,}
                """)
                    
        except Exception as e:
            st.error(f"Error during export: {str(e)}")
            progress_bar.empty()
            status_text.empty()



def display_existing_exports():
    """Display existing export files"""
    
    export_dir = Path("outputs/embeddings")
    if not export_dir.exists():
        return
    
    parquet_files = list(export_dir.glob("*.parquet"))
    
    if parquet_files:
        st.subheader("📁 Existing Exports")
        
        # Sort by modification time
        parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Create a table of exports
        export_data = []
        for file in parquet_files[:10]:  # Show last 10
            try:
                df = pd.read_parquet(file)
                export_data.append({
                    "File": file.name,
                    "Documents": len(df),
                    "Size (MB)": f"{file.stat().st_size / 1024 / 1024:.1f}",
                    "Modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            except Exception:
                continue

        if export_data:
            export_df = pd.DataFrame(export_data)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_file = st.selectbox(
                    "Load existing export",
                    [None] + [f["File"] for f in export_data],
                    format_func=lambda x: "Select a file..." if x is None else x
                )
            
            with col2:
                # Use markdown with custom height to align with selectbox
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                if st.button("📂 Load Selected", use_container_width=True):
                    if selected_file:
                        load_existing_export(export_dir / selected_file)
            
            st.dataframe(export_df, use_container_width=True, hide_index=True)
    
    # Query Comparison Section - shown when we have loaded embeddings
    if "embedding_data" in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Query Projection")
        
        st.info("""
        **Visualize where your queries land in the embedding space:**
        Add one or more queries to see where they appear among your documents
        """)
        
        # Initialize query list in session state if not present
        if "query_list" not in st.session_state:
            st.session_state.query_list = []
        
        # Input for new query with visible label above
        st.write("**Add Query:**")
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Use a unique key that changes to clear the input after adding
            input_key = f"new_query_input_{len(st.session_state.query_list)}"
            new_query = st.text_input(
                "",  # Empty label since we show it above
                placeholder="Enter query text (e.g., outdoor sports activities)",
                key=input_key,
                help="Query will be added to the visualization with the same encoder used for documents",
                label_visibility="collapsed"  # Hide the label since we have one above
            )
        
        with col2:
            if st.button("➕ Add", use_container_width=True, key="add_query_btn"):
                if new_query and new_query not in st.session_state.query_list:
                    st.session_state.query_list.append(new_query)
                    st.rerun()
                elif new_query in st.session_state.query_list:
                    st.warning("Query already added")
                else:
                    st.warning("Please enter a query text")
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True, key="clear_all_btn"):
                st.session_state.query_list = []
                if "query_comparison" in st.session_state:
                    del st.session_state.query_comparison
                st.rerun()
        
        # Display current queries
        if st.session_state.query_list:
            st.write("**Current queries:**")
            for i, query in enumerate(st.session_state.query_list):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {query}")
                with col2:
                    if st.button("❌", key=f"remove_{i}", help=f"Remove: {query[:30]}..."):
                        st.session_state.query_list.pop(i)
                        st.rerun()
            
            # Project all queries button
            if st.button("🎯 Project All Queries", use_container_width=True, type="primary"):
                schema = st.session_state.get("last_schema", "video_frame")
                compare_multiple_queries_with_embeddings(st.session_state.query_list, schema)
        
        # Show comparison results if available
        if "query_comparison" in st.session_state:
            display_query_comparison()


def load_existing_export(file_path: Path):
    """Load an existing export file"""
    try:
        with st.spinner(f"Loading {file_path.name}..."):
            df = pd.read_parquet(file_path)
            
            # Only compute dimensions if they don't exist
            if "x" not in df.columns or "y" not in df.columns:
                if "embedding" in df.columns:
                    st.info("Computing 2D projection (file was exported without dimension reduction)...")
                    
                    # Get embeddings
                    embeddings = np.vstack(df["embedding"].values)
                    
                    # Use UMAP for dimensionality reduction
                    from umap import UMAP
                    reducer = UMAP(n_components=2, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                    
                    df["x"] = coords[:, 0]
                    df["y"] = coords[:, 1]
                    
                    # Add clustering if not present
                    if "cluster" not in df.columns:
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=min(8, len(df)), random_state=42)
                        df["cluster"] = kmeans.fit_predict(embeddings)
                    
                    # Save the updated dataframe back to the file
                    st.info("Saving computed projections to file...")
                    df.to_parquet(file_path)
                else:
                    st.warning("No embeddings found in file - cannot compute 2D projection")
            
            st.session_state.embedding_data = {
                "df": df,
                "file": str(file_path),
                "timestamp": file_path.stem.split("_")[-1],
                "method": "loaded",
                "n_docs": len(df)
            }
            
            # Try to detect schema from filename or data
            if "video_frame" in str(file_path):
                st.session_state.last_schema = "video_frame"
            elif "global" in str(file_path) or "videoprism" in str(file_path):
                st.session_state.last_schema = "global"
            else:
                # Default based on columns
                if "frame_number" in df.columns:
                    st.session_state.last_schema = "video_frame"
                else:
                    st.session_state.last_schema = "video_frame"  # safe default
            
            # Clear encoder type to force re-detection
            if "encoder_type" in st.session_state:
                del st.session_state.encoder_type
            
            st.success(f"Loaded {len(df)} documents from {file_path.name}")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")


def compare_multiple_queries_with_embeddings(query_texts: list, schema: str):
    """Add multiple query embeddings to the visualization"""
    
    with st.spinner(f"Adding {len(query_texts)} queries to visualization..."):
        try:
            # Get the embedding data
            df = st.session_state.embedding_data["df"].copy()
            
            # Get the strategy/profile that was used - MUST be in the data
            if "encoder_profile" in df.columns and not df["encoder_profile"].isna().all():
                strategy_profile = df["encoder_profile"].iloc[0]
            else:
                # No profile found - this is a problem!
                st.error("❌ Cannot determine encoder profile!")
                st.warning("The embedding file doesn't contain profile information.")
                st.info("Please re-export with the latest version and select the correct encoder profile.")
                return
            
            # Using strategy profile: {strategy_profile}
            
            # Create encoder from the strategy
            from src.app.agents.query_encoders import QueryEncoderFactory
            encoder = QueryEncoderFactory.create_encoder(strategy_profile)
            
            # Process each query
            query_rows = []
            query_embeddings = []
            
            for query_text in query_texts:
                # Encode query
                query_embedding_raw = encoder.encode(query_text)
                query_embeddings.append(query_embedding_raw)
                
                # Create a row for this query - ensure consistent float32 dtype
                if strategy_profile == "frame_based_colpali":
                    # For ColPali, store the raw multi-vector embedding
                    query_flat = query_embedding_raw.flatten().astype(np.float32)
                    query_row = {
                        "text": f"QUERY: {query_text}",
                        "embedding": query_flat,
                        "is_query": True,
                        "encoder_profile": strategy_profile
                    }
                else:
                    # For single-vector models, store as is with float32
                    query_emb = query_embedding_raw.astype(np.float32) if isinstance(query_embedding_raw, np.ndarray) else np.array(query_embedding_raw, dtype=np.float32)
                    query_row = {
                        "text": f"QUERY: {query_text}",
                        "embedding": query_emb,
                        "is_query": True,
                        "encoder_profile": strategy_profile
                    }
                
                query_rows.append(query_row)
            
            # Ensure all documents have is_query field set to False
            if "is_query" not in df.columns:
                df["is_query"] = False
            else:
                # Remove any existing queries from df
                df = df[~df["is_query"]].copy()
                # Ensure remaining docs are marked as not queries
                df["is_query"] = False

            # Add all new queries to dataframe
            query_df = pd.DataFrame(query_rows)
            df = pd.concat([df, query_df], ignore_index=True)

            # Added queries to dataframe

            # Handle ColPali-specific projection
            if strategy_profile == "frame_based_colpali":
                # Get document embeddings (all non-query rows)
                doc_df = df[~df["is_query"]]
                query_df_indices = df[df["is_query"]].index.tolist()
                
                # First compute MaxSim scores to get actual similarities
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Build similarity matrix between queries and documents
                similarity_matrix = []
                
                for query_idx, query_emb_raw in zip(query_df_indices, query_embeddings):
                    if query_emb_raw.ndim == 1:
                        query_tokens = query_emb_raw.reshape(-1, 128)
                    else:
                        query_tokens = query_emb_raw
                    
                    doc_similarities = []
                    for idx, row in doc_df.iterrows():
                        doc_emb = np.array(row["embedding"])
                        if doc_emb.ndim == 1:
                            num_patches = len(doc_emb) // 128
                            doc_patches = doc_emb.reshape(num_patches, 128)
                        else:
                            doc_patches = doc_emb
                        
                        # Use visual patches only for MaxSim
                        if num_patches == 874:
                            doc_patches = doc_patches[42:]
                        elif num_patches == 1139:
                            doc_patches = doc_patches[51:]
                        
                        similarities = cosine_similarity(query_tokens, doc_patches)
                        max_sims = similarities.max(axis=1)
                        maxsim_score = max_sims.mean()
                        doc_similarities.append(maxsim_score)
                    
                    similarity_matrix.append(doc_similarities)
                    # Store similarity scores in dataframe
                    df.loc[doc_df.index, f"query_similarity_{query_idx}"] = doc_similarities
                
                # Create averaged document embeddings for UMAP
                doc_embeddings_avg = []
                for idx, row in doc_df.iterrows():
                    doc_emb = np.array(row["embedding"])
                    if doc_emb.ndim == 1:
                        num_patches = len(doc_emb) // 128
                        doc_patches = doc_emb.reshape(num_patches, 128)
                    else:
                        doc_patches = doc_emb
                    
                    # Skip system tokens and average
                    if num_patches == 874:
                        visual_patches = doc_patches[42:]
                    elif num_patches == 1139:
                        visual_patches = doc_patches[51:]
                    else:
                        visual_patches = doc_patches
                    
                    # Average the patches to get document representation
                    doc_avg = visual_patches.mean(axis=0)
                    doc_embeddings_avg.append(doc_avg)
                
                # Create averaged query embeddings
                query_embeddings_avg = []
                for query_emb_raw in query_embeddings:
                    if query_emb_raw.ndim == 1:
                        query_tokens = query_emb_raw.reshape(-1, 128)
                    else:
                        query_tokens = query_emb_raw
                    query_avg = query_tokens.mean(axis=0)
                    query_embeddings_avg.append(query_avg)
                
                # Combine all averaged embeddings for UMAP
                all_embeddings = np.vstack(doc_embeddings_avg + query_embeddings_avg)
                
                # Apply UMAP with cosine metric to match MaxSim
                from umap import UMAP
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(all_embeddings) - 1),
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                coords_2d = reducer.fit_transform(all_embeddings)
                
                # Assign coordinates to documents
                for i, idx in enumerate(doc_df.index):
                    df.loc[idx, "x"] = np.float32(coords_2d[i, 0])
                    df.loc[idx, "y"] = np.float32(coords_2d[i, 1])
                
                # Assign coordinates to queries
                query_start_idx = len(doc_df)
                for i, query_idx in enumerate(query_df_indices):
                    df.loc[query_idx, "x"] = np.float32(coords_2d[query_start_idx + i, 0])
                    df.loc[query_idx, "y"] = np.float32(coords_2d[query_start_idx + i, 1])
                    # Add self-similarity
                    df.loc[query_idx, f"query_similarity_{query_idx}"] = 1.0
            
            else:
                # For non-ColPali models, compute standard projections
                from umap import UMAP
                
                embeddings = np.vstack(df["embedding"].values)
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(df) - 1),
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                coords_2d = reducer.fit_transform(embeddings)
                df["x"] = coords_2d[:, 0].astype(np.float32)
                df["y"] = coords_2d[:, 1].astype(np.float32)
            
            # Automatically save the file with queries
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"outputs/embeddings/with_{len(query_texts)}_queries_{timestamp}.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Fix dtype consistency for embeddings before saving
            df_save = df.copy()
            
            # Convert embeddings to consistent float32 arrays
            if "embedding" in df_save.columns:
                embeddings_list = []
                for emb in df_save["embedding"]:
                    if isinstance(emb, np.ndarray):
                        embeddings_list.append(emb.astype(np.float32))
                    else:
                        embeddings_list.append(np.array(emb, dtype=np.float32))
                df_save["embedding"] = embeddings_list
            
            # Ensure numeric columns have consistent dtypes
            for col in ["x", "y", "z"]:
                if col in df_save.columns:
                    df_save[col] = df_save[col].astype(np.float32)
            
            # Save with fixed dtypes
            df_save.to_parquet(output_path)
            
            # Update session state with saved data
            st.session_state.embedding_data["df"] = df_save
            st.session_state.embedding_data["file"] = str(output_path)
            st.session_state.query_comparison = {
                "queries": query_texts,
                "results": df_save
            }
            
            st.success(f"✅ Successfully projected {len(query_texts)} queries and saved to: {output_path.name}")
            st.info("Ready to visualize! Use the visualization buttons in the Actions section above")
            
        except Exception as e:
            st.error(f"Error comparing queries: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def compare_query_with_embeddings(query_text: str, schema: str):
    """Add single query embedding to the visualization (backward compatibility)"""
    compare_multiple_queries_with_embeddings([query_text], schema)
    
    with st.spinner("Adding query to visualization..."):
        try:
            # Get the embedding data
            df = st.session_state.embedding_data["df"].copy()
            
            # Get the strategy/profile that was used - MUST be in the data
            if "encoder_profile" in df.columns and not df["encoder_profile"].isna().all():
                strategy_profile = df["encoder_profile"].iloc[0]
            else:
                # No profile found - this is a problem!
                st.error("❌ Cannot determine encoder profile!")
                st.warning("The embedding file doesn't contain profile information.")
                st.info("Please re-export with the latest version and select the correct encoder profile.")
                return
            
            # Using strategy profile: {strategy_profile}
            
            # Create encoder from the strategy
            from src.app.agents.query_encoders import QueryEncoderFactory
            encoder = QueryEncoderFactory.create_encoder(strategy_profile)
            
            # Encode query
            query_embedding_raw = encoder.encode(query_text)
            
            # Debug: Check dimensions
            st.info(f"Raw query embedding shape: {query_embedding_raw.shape}")
            
            # Get a sample document embedding to match format
            sample_doc_emb = df["embedding"].iloc[0]
            doc_dim = len(sample_doc_emb)
            
            # For patch-based models, we need to handle dimension mismatch
            if strategy_profile == "frame_based_colpali" and query_embedding_raw.ndim == 2:
                # ColPali: query has fewer patches than documents
                query_patches, query_dim = query_embedding_raw.shape
                st.info(f"Query: {query_patches} patches × {query_dim} dims")
                
                # Documents are flattened - figure out their patch configuration
                # Common case: 111872 = 128 patches × 874 dims (binary quantized)
                if doc_dim == 111872:
                    doc_patches = 128
                    doc_patch_dim = 874
                    st.info(f"Documents: {doc_patches} patches × {doc_patch_dim} dims (binary quantized)")
                    
                    # The mismatch is both in patches (14 vs 128) and dims (128 vs 874)
                    # 874 comes from binary quantization of 1024 dims
                    # We need to:
                    # 1. Binary quantize the query embeddings (128 -> ~109 bytes)
                    # 2. Pad to 128 patches
                    
                    st.info("Matching query format to documents...")
                    
                    # Binary quantize each patch
                    quantized_patches = []
                    for patch in query_embedding_raw:
                        # Convert to binary (>0 becomes 1, <=0 becomes 0)
                        binary = (patch > 0).astype(np.uint8)
                        # Pack bits into bytes
                        packed = np.packbits(binary)
                        quantized_patches.append(packed)
                    
                    # Convert to array
                    quantized_patches = np.array(quantized_patches)
                    st.info(f"Quantized query patches shape: {quantized_patches.shape}")
                    
                    # Now we have 14 patches × 16 bytes (128 bits packed)
                    # But documents have 874 dims per patch...
                    # 874 * 8 = 6992 bits, but patches are 128 dims = 128 bits
                    # This suggests documents store more than just embeddings
                    
                    # Let's try padding the packed embeddings
                    # Pad each patch to 874 dims with zeros
                    padded_patches = []
                    for patch in quantized_patches:
                        # Pad to 874 dims
                        padded = np.pad(patch, (0, 874 - len(patch)), mode='constant')
                        padded_patches.append(padded)
                    
                    # Now pad to 128 patches (add zero patches)
                    query_array = np.array(padded_patches)  # Shape: (14, 874)
                    zero_patches = np.zeros((doc_patches - query_patches, doc_patch_dim))
                    query_padded = np.vstack([query_array, zero_patches])  # Shape: (128, 874)
                    
                    # Flatten to match document format
                    query_embedding = query_padded.flatten()
                    st.success(f"Padded query to {query_embedding.shape} to match documents")
                    
                else:
                    # Try simpler padding for other configurations
                    query_flat = query_embedding_raw.flatten()
                    
                    if len(query_flat) < doc_dim:
                        # Pad with zeros
                        query_embedding = np.pad(query_flat, (0, doc_dim - len(query_flat)), mode='constant')
                        st.info(f"Padded query from {len(query_flat)} to {doc_dim} dims")
                    else:
                        query_embedding = query_flat
                        st.warning(f"Query has {len(query_flat)} dims, documents have {doc_dim} dims")
            else:
                # Non-patch based or matching dimensions
                query_embedding = query_embedding_raw.flatten()
            
            st.info(f"Final query embedding shape: {query_embedding.shape}")
            
            # Create a query row that matches the document format
            query_row = {
                "text": f"QUERY: {query_text}",
                "embedding": query_embedding,
                "is_query": True,
                "video_id": "QUERY",
                "frame_number": -1,
                "timestamp": -1
            }
            
            # Add any other columns with default values
            for col in df.columns:
                if col not in query_row and col not in ["x", "y", "z"]:
                    query_row[col] = None
            
            # Append query to dataframe - convert query_row to DataFrame with proper dtypes
            query_df = pd.DataFrame([query_row])
            # Ensure dtypes match for concatenation
            for col in df.columns:
                if col in query_df.columns and col != "embedding":
                    try:
                        query_df[col] = query_df[col].astype(df[col].dtype)
                    except Exception:
                        pass  # Keep original dtype if conversion fails
            df = pd.concat([df, query_df], ignore_index=True)
            
            # Mark documents as not queries
            df.loc[df.index[:-1], "is_query"] = False
            
            # Recompute UMAP with query included
            st.info("Recomputing projection with query...")
            
            # Debug: Check document embedding dimensions
            sample_doc_emb = df["embedding"].iloc[0]
            st.info(f"Sample document embedding shape: {np.array(sample_doc_emb).shape}")
            
            # Special handling for ColPali multi-vector embeddings
            if strategy_profile == "frame_based_colpali":
                st.info("🎯 Using joint token projection for ColPali (all visual patches + query tokens)")
                
                # Filter for non-query rows
                if "is_query" in df.columns:
                    doc_df = df[~df["is_query"]]
                else:
                    doc_df = df[:-1]  # All except last row (which is the query we just added)
                
                # Collect ALL visual patches from documents (no sampling!)
                all_visual_patches = []
                doc_indices = []  # Track which doc each patch belongs to
                
                for idx, row in doc_df.iterrows():
                    doc_emb = np.array(row["embedding"])
                    if doc_emb.ndim == 1:
                        # Reshape flattened embedding back to (num_patches, 128)
                        num_patches = len(doc_emb) // 128
                        doc_patches = doc_emb.reshape(num_patches, 128)
                    else:
                        doc_patches = doc_emb
                    
                    # For 1280x720 frames: 874 tokens total
                    # Skip first ~42 system tokens, use visual patches only
                    # For safety, we'll detect based on total patches
                    if num_patches == 874:
                        # Skip system tokens (roughly first 42)
                        visual_patches = doc_patches[42:]  # 832 visual patches
                        st.info(f"Doc {idx}: Using patches 42-874 (832 visual patches)")
                    elif num_patches == 1139:
                        # Square image format - skip first 51 tokens
                        visual_patches = doc_patches[51:]  # 1088 visual patches
                        st.info(f"Doc {idx}: Using patches 51-1139 (1088 visual patches)")
                    else:
                        # Unknown format - use all patches
                        visual_patches = doc_patches
                        st.warning(f"Doc {idx}: Unknown format with {num_patches} patches, using all")
                    
                    all_visual_patches.extend(visual_patches)
                    doc_indices.extend([idx] * len(visual_patches))
                
                # Add query tokens (these are text tokens, but in same embedding space)
                if query_embedding_raw.ndim == 1:
                    query_tokens = query_embedding_raw.reshape(-1, 128)
                else:
                    query_tokens = query_embedding_raw
                
                query_start_idx = len(all_visual_patches)
                all_visual_patches.extend(query_tokens)
                
                # Stack all tokens
                all_tokens = np.vstack(all_visual_patches)
                st.info(f"Projecting {len(all_tokens)} total tokens ({query_start_idx} visual patches + {len(query_tokens)} query tokens)")
                
                # Apply UMAP to ALL tokens together (visual patches + query tokens)
                from sklearn.metrics.pairwise import cosine_similarity
                from umap import UMAP
                
                # Use cosine metric since ColPali uses cosine similarity
                reducer = UMAP(
                    n_components=2, 
                    n_neighbors=min(30, len(all_tokens) - 1),  # More neighbors for better structure
                    metric='cosine',  # Use cosine like ColPali does
                    random_state=42
                )
                coords_2d = reducer.fit_transform(all_tokens)
                
                # Average visual patch positions back to documents
                for idx in doc_df.index:
                    # Find all patches belonging to this document
                    token_mask = [i for i, d in enumerate(doc_indices) if d == idx]
                    if token_mask:
                        doc_coords = coords_2d[token_mask].mean(axis=0)
                        df.loc[idx, "x"] = doc_coords[0]
                        df.loc[idx, "y"] = doc_coords[1]
                
                # Average query token positions
                query_coords = coords_2d[query_start_idx:].mean(axis=0)
                query_idx = df[df["is_query"]].index[0]
                df.loc[query_idx, "x"] = query_coords[0]
                df.loc[query_idx, "y"] = query_coords[1]
                
                # Also compute MaxSim scores for visualization
                st.info("Computing MaxSim scores for similarity visualization...")
                maxsim_scores = []
                for idx, row in doc_df.iterrows():
                    doc_emb = np.array(row["embedding"])
                    if doc_emb.ndim == 1:
                        num_patches = len(doc_emb) // 128
                        doc_patches = doc_emb.reshape(num_patches, 128)
                    else:
                        doc_patches = doc_emb
                    
                    # Use visual patches only for MaxSim
                    if num_patches == 874:
                        doc_patches = doc_patches[42:]
                    elif num_patches == 1139:
                        doc_patches = doc_patches[51:]
                    
                    similarities = cosine_similarity(query_tokens, doc_patches)
                    max_sims = similarities.max(axis=1)
                    maxsim_score = max_sims.mean()
                    maxsim_scores.append(maxsim_score)
                
                # Store similarity scores
                df["query_similarity"] = 0.0
                for i, idx in enumerate(doc_df.index):
                    df.loc[idx, "query_similarity"] = maxsim_scores[i]
                
                top_scores = sorted(maxsim_scores, reverse=True)[:3]
                st.success(f"✅ Joint projection complete! Top MaxSim scores: {[round(s, 3) for s in top_scores]}")
                
            else:
                # Standard approach for non-patch based models
                # Check if dimensions match
                if len(query_embedding) != len(sample_doc_emb):
                    st.error(f"Dimension mismatch! Query: {len(query_embedding)}, Documents: {len(sample_doc_emb)}")
                    st.warning("The encoder may not be producing embeddings in the expected format.")
                    return
                
                embeddings = np.vstack(df["embedding"].values)
                
                from umap import UMAP
                reducer = UMAP(n_components=2, n_neighbors=min(15, len(embeddings) - 1), random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
                
                df["x"] = coords_2d[:, 0]
                df["y"] = coords_2d[:, 1]
            
            # Add z-coordinate for 3D visualization
            if "z" not in df.columns:
                # Add z based on clusters or random
                if "cluster" in df.columns:
                    df["z"] = df["cluster"] * 0.5 + np.random.randn(len(df)) * 0.2
                else:
                    df["z"] = np.random.randn(len(df))
            
            # Ensure query has a z-coordinate
            query_idx = df[df["is_query"]].index[0] if "is_query" in df.columns else -1
            if query_idx >= 0 and pd.isna(df.loc[query_idx, "z"]):
                # Place query slightly above mean for visibility
                mean_z = df[df.index != query_idx]["z"].mean()
                df.loc[query_idx, "z"] = mean_z + 1.0
            
            # Save to new file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"outputs/embeddings/with_query_{timestamp}.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(output_path)
            
            # Update session state
            st.session_state.embedding_data["df"] = df
            st.session_state.embedding_data["file"] = str(output_path)
            
            query_point = df[df["is_query"]].iloc[0]
            st.success(f"✅ Query added at ({query_point['x']:.2f}, {query_point['y']:.2f})")
            st.info(f"Saved to: {output_path.name}")
            
            # Now user can open visualization to see the query point
            
        except Exception as e:
            st.error(f"Failed to add query: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_query_comparison():
    """Display query projection results"""
    
    if "query_comparison" not in st.session_state:
        return
    
    comp = st.session_state.query_comparison
    df = comp["results"]
    queries = comp["queries"]
    
    # Show query positions
    st.subheader("📍 Query Positions in Embedding Space")
    
    query_df = df[df["is_query"]]
    if not query_df.empty:
        # Display query coordinates
        query_info = []
        for idx, row in query_df.iterrows():
            query_text = row["text"].replace("QUERY: ", "")
            query_info.append({
                "Query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                "X": f"{row['x']:.3f}",
                "Y": f"{row['y']:.3f}",
                "Z": f"{row.get('z', 0):.3f}"
            })
        
        st.dataframe(pd.DataFrame(query_info), use_container_width=True, hide_index=True)
    
    # Show top similar documents for each query
    if any(col.startswith("query_similarity_") for col in df.columns):
        st.subheader("🎯 Top Similar Documents")
        
        doc_df = df[~df["is_query"]]

        for i, query_text in enumerate(queries):
            similarity_col = f"query_similarity_{query_df.index[i]}"
            if similarity_col in doc_df.columns:
                st.write(f"**Query {i+1}:** {query_text[:80]}...")
                
                # Get top 5 similar documents
                top_docs = doc_df.nlargest(5, similarity_col)
                
                results = []
                for _, doc in top_docs.iterrows():
                    # Build display text based on available columns
                    if "video_title" in doc and pd.notna(doc["video_title"]):
                        display_text = f"{doc['video_title']}"
                        if "frame_number" in doc:
                            display_text += f" (Frame {doc['frame_number']})"
                    elif "title" in doc and pd.notna(doc["title"]):
                        display_text = doc["title"]
                    elif "text" in doc and pd.notna(doc["text"]):
                        display_text = doc["text"][:100] + "..."
                    else:
                        display_text = f"Document {doc.name}"
                    
                    results.append({
                        "Document": display_text,
                        "Similarity": f"{doc[similarity_col]:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    
    # Show the file that was automatically saved
    if "embedding_data" in st.session_state and "file" in st.session_state.embedding_data:
        st.info(f"📁 Data with queries saved to: {Path(st.session_state.embedding_data['file']).name}")


def open_atlas_viewer(viewer_type="apple"):
    """Open the embedding atlas viewer in browser
    
    Args:
        viewer_type: "apple" for Apple's embedding-atlas, "custom" for plotly visualization
    """
    
    if "embedding_data" in st.session_state:
        file_path = st.session_state.embedding_data["file"]
        
        try:
            # Store file path in session state for the atlas app
            st.session_state.embedding_atlas_file = str(file_path)
            
            # Kill any existing streamlit on port 5858
            subprocess.run(["pkill", "-f", "streamlit.*5858"], capture_output=True)
            import time
            time.sleep(1)
            
            # Choose which viewer to launch
            if viewer_type == "custom":
                script = "scripts/simple_atlas.py"
            else:
                script = "scripts/atlas_viewer.py"
            
            cmd = [
                "uv", "run", "streamlit", "run",
                script,
                "--server.port", "5858"
            ]
            
            # Create log file for debugging
            log_file = Path("/tmp/atlas_streamlit.log")
            
            # Run in background
            with open(log_file, "w") as log:
                _process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, 'STREAMLIT_EMBEDDING_FILE': str(file_path)}
                )
            
            # Wait a bit for the app to start
            time.sleep(2)
            
            viewer_name = "Embedding Atlas" if viewer_type == "apple" else "Custom 3D Visualization"
            st.success(f"✅ {viewer_name} visualization available at http://localhost:5858")
            
        except Exception as e:
            st.error(f"Failed to launch: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("No embedding data loaded. Please export or load data first.")
