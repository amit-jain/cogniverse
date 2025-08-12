"""
Embedding Atlas visualization tab for Dashboard
"""

# Fix protobuf issue - must be before other imports
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime
import sys
from typing import Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import search backend
try:
    from src.search.vespa_search_backend import VespaSearchBackend as SearchBackend
    from src.tools.config import get_config
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
        config = get_config()
        backend = SearchBackend(
            url=config.get("vespa_url", "http://localhost"),
            port=config.get("vespa_port", 8080),
            schema=config.get("vespa_schema", "video_frame")
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
        except:
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
    
    except Exception as e:
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
                    except:
                        pass
        
        # Default to video_frame if no schemas found
        if not available_schemas:
            available_schemas = ["video_frame"]
        
        selected_schema = st.selectbox(
            "Schema",
            sorted(available_schemas),
            help="Select the schema to export embeddings from"
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
    export_type: str,
    max_docs: int,
    video_id: str = None,
    query: str = None
):
    """Export embeddings from backend and prepare for visualization"""
    
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
        "--schema", schema,
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
            except:
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
                if st.button("📂 Load Selected"):
                    if selected_file:
                        load_existing_export(export_dir / selected_file)
            
            st.dataframe(export_df, use_container_width=True, hide_index=True)


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
            
            st.success(f"Loaded {len(df)} documents from {file_path.name}")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")


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
            
            # Build URL with file parameter
            import urllib.parse
            file_param = urllib.parse.quote(str(file_path))
            
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
                process = subprocess.Popen(
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