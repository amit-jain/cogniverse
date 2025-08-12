"""
Embedding Atlas visualization tab for Phoenix Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def render_embedding_atlas_tab():
    """Render the embedding atlas visualization tab"""
    
    # Introduction
    st.markdown("""
    This tab allows you to visualize embeddings from Vespa using Apple's Embedding Atlas.
    The visualization helps you understand:
    - **Clustering patterns** in your video frames
    - **Semantic relationships** between different content
    - **Outliers and anomalies** in the embedding space
    - **Coverage gaps** in your indexed content
    """)
    
    # Create columns for controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📥 Export Settings")
        
        # Export options
        export_type = st.selectbox(
            "Export Type",
            ["Sample (Fast)", "Filtered", "Full Dataset"],
            help="Choose what to export from Vespa"
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
        st.subheader("🎨 Visualization Settings")
        
        # Dimension reduction settings
        reduction_method = st.selectbox(
            "Dimension Reduction",
            ["umap", "tsne", "pca"],
            help="Method for reducing high-dimensional embeddings to 2D/3D"
        )
        
        n_components = st.radio(
            "Visualization Dimensions",
            [2, 3],
            format_func=lambda x: f"{x}D",
            horizontal=True,
            help="2D for standard plots, 3D for interactive exploration"
        )
        
        # Clustering
        enable_clustering = st.checkbox(
            "Enable automatic clustering",
            value=True,
            help="Automatically identify clusters in the embedding space"
        )
        
        if enable_clustering:
            n_clusters = st.slider(
                "Number of clusters",
                min_value=3,
                max_value=20,
                value=8,
                help="Number of clusters to identify"
            )
    
    with col3:
        st.subheader("🚀 Actions")
        
        # Export button
        if st.button("🔄 Export & Visualize", type="primary", use_container_width=True):
            export_and_visualize(
                export_type=export_type,
                max_docs=max_docs,
                video_id=video_id if export_type == "Filtered" else None,
                query=query if export_type == "Filtered" else None,
                reduction_method=reduction_method,
                n_components=n_components,
                enable_clustering=enable_clustering,
                n_clusters=n_clusters if enable_clustering else None
            )
        
        # Open external viewer
        if st.button("🌐 Open Atlas Viewer", use_container_width=True):
            open_atlas_viewer()
    
    # Display area for results
    st.markdown("---")
    
    # Check for existing exports
    display_existing_exports()
    
    # Visualization area
    if "embedding_data" in st.session_state:
        display_embedding_visualization()


def export_and_visualize(
    export_type: str,
    max_docs: int,
    video_id: str = None,
    query: str = None,
    reduction_method: str = "umap",
    n_components: int = 2,
    enable_clustering: bool = True,
    n_clusters: int = 8
):
    """Export embeddings from Vespa and prepare for visualization"""
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"vespa_embeddings_{timestamp}.parquet"
    
    # Build export command
    cmd = [
        "uv", "run", "python", "scripts/export_vespa_embeddings.py",
        "--output", str(output_file),
        "--max-documents", str(max_docs),
        "--reduction-method", reduction_method
    ]
    
    if video_id:
        cmd.extend(["--video-id", video_id])
    
    if query:
        cmd.extend(["--query", query])
    
    if n_components == 3:
        # Modify script to support 3D (would need to update the export script)
        pass
    
    # Show progress
    with st.spinner(f"Exporting {max_docs} documents from Vespa..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run export
            status_text.text("Connecting to Vespa...")
            progress_bar.progress(20)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                st.error(f"Export failed: {result.stderr}")
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
            
            # Show quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents", f"{len(df):,}")
            with col2:
                unique_videos = df['video_id'].nunique() if 'video_id' in df.columns else 0
                st.metric("Unique Videos", f"{unique_videos:,}")
            with col3:
                if 'cluster' in df.columns:
                    st.metric("Clusters", f"{df['cluster'].nunique():,}")
            with col4:
                if 'embedding' in df.columns and len(df) > 0:
                    embed_dim = len(df.iloc[0]['embedding'])
                    st.metric("Embedding Dim", embed_dim)
            
        except Exception as e:
            st.error(f"Error during export: {str(e)}")
            progress_bar.empty()
            status_text.empty()


def display_embedding_visualization():
    """Display the embedding visualization"""
    
    if "embedding_data" not in st.session_state:
        return
    
    data = st.session_state.embedding_data
    df = data["df"]
    
    st.subheader("📊 Embedding Visualization")
    
    # Create tabs for different views
    viz_tabs = st.tabs(["Scatter Plot", "Density Map", "Cluster Analysis", "Data Explorer"])
    
    # Scatter Plot
    with viz_tabs[0]:
        if "x" in df.columns and "y" in df.columns:
            # Color by options
            color_by = st.selectbox(
                "Color by",
                ["cluster", "video_id", "relevance_score", "frame_number"],
                key="scatter_color"
            )
            
            if color_by in df.columns:
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color=color_by,
                    hover_data=["video_title", "frame_description", "timestamp"],
                    title=f"Embedding Space ({data['method'].upper()})",
                    height=600
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Column '{color_by}' not found in data")
        else:
            st.info("2D coordinates not available. Run export with dimension reduction enabled.")
    
    # Density Map
    with viz_tabs[1]:
        if "x" in df.columns and "y" in df.columns:
            fig = px.density_heatmap(
                df,
                x="x",
                y="y",
                title="Embedding Density Map",
                height=600,
                nbinsx=50,
                nbinsy=50
            )
            
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Analysis
    with viz_tabs[2]:
        if "cluster" in df.columns:
            # Cluster distribution
            cluster_counts = df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    title="Documents per Cluster",
                    labels={"x": "Cluster ID", "y": "Document Count"}
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Select cluster to explore
                selected_cluster = st.selectbox(
                    "Select cluster to explore",
                    sorted(df['cluster'].unique())
                )
                
                cluster_df = df[df['cluster'] == selected_cluster]
                st.write(f"**Cluster {selected_cluster}**: {len(cluster_df)} documents")
                
                # Show sample documents from cluster
                st.write("Sample documents:")
                sample_cols = ["video_title", "frame_description", "timestamp"]
                available_cols = [col for col in sample_cols if col in cluster_df.columns]
                if available_cols:
                    st.dataframe(
                        cluster_df[available_cols].head(5),
                        use_container_width=True
                    )
        else:
            st.info("Clustering not enabled. Re-export with clustering to see this analysis.")
    
    # Data Explorer
    with viz_tabs[3]:
        st.write(f"**Total documents**: {len(df):,}")
        
        # Search/filter
        search_term = st.text_input("Search in text", key="data_search")
        if search_term and "text" in df.columns:
            filtered_df = df[df['text'].str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(filtered_df)} matching documents")
            
            if len(filtered_df) > 0:
                display_cols = ["video_id", "text", "timestamp", "relevance_score"]
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                st.dataframe(
                    filtered_df[available_cols].head(20),
                    use_container_width=True
                )
        else:
            # Show raw data sample
            st.write("Data sample:")
            display_cols = ["video_id", "video_title", "frame_description", "timestamp"]
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                st.dataframe(
                    df[available_cols].head(20),
                    use_container_width=True
                )


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
            
            st.session_state.embedding_data = {
                "df": df,
                "file": str(file_path),
                "timestamp": file_path.stem.split("_")[-1],
                "method": "unknown",
                "n_docs": len(df)
            }
            
            st.success(f"Loaded {len(df)} documents from {file_path.name}")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")


def open_atlas_viewer():
    """Open the embedding atlas viewer"""
    
    if "embedding_data" in st.session_state:
        file_path = st.session_state.embedding_data["file"]
        
        # Try to launch embedding-atlas
        try:
            # Check if embedding-atlas is installed
            result = subprocess.run(
                ["which", "embedding-atlas"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Launch in background
                subprocess.Popen(
                    ["embedding-atlas", file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                st.success("Launching Embedding Atlas viewer...")
                st.info(f"If the viewer doesn't open, run in terminal:\n`embedding-atlas {file_path}`")
            else:
                st.warning("embedding-atlas not found. Install with: `pip install embedding-atlas`")
                st.code(f"embedding-atlas {file_path}", language="bash")
        except Exception as e:
            st.error(f"Error launching viewer: {str(e)}")
            st.code(f"embedding-atlas {file_path}", language="bash")
    else:
        st.warning("No embedding data loaded. Export data first.")