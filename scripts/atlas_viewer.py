#!/usr/bin/env python3
"""
Proper embedding atlas visualization using Apple's embedding-atlas library
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from embedding_atlas.streamlit import embedding_atlas
from umap import UMAP

st.set_page_config(page_title="Embedding Atlas", layout="wide")

# Title
st.title("🗺️ Embedding Analysis")

# Get file path from environment or session state
file_path = None

if 'embedding_atlas_file' in st.session_state:
    file_path = Path(st.session_state.embedding_atlas_file)
elif 'STREAMLIT_EMBEDDING_FILE' in os.environ:
    file_path = Path(os.environ['STREAMLIT_EMBEDDING_FILE'])

# File selector
if not file_path:
    parquet_dir = Path("outputs/embeddings")
    if parquet_dir.exists():
        files = sorted(parquet_dir.glob("*.parquet"))
        if files:
            file_path = st.selectbox(
                "Choose embedding file:",
                files,
                format_func=lambda x: x.name,
                index=len(files)-1
            )

# Load data
if file_path and file_path.exists():
    # Load the parquet file
    df = pd.read_parquet(file_path)
    
    # Only compute UMAP if x,y don't exist (shouldn't happen if exported from dashboard)
    if ('x' not in df.columns or 'y' not in df.columns) and 'embedding' in df.columns:
        with st.spinner("Computing 2D projection (this should have been done during export)..."):
            embeddings = np.array(df['embedding'].tolist())
            n_neighbors = min(15, len(df) - 1)
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            coords = reducer.fit_transform(embeddings)
            df['x'] = coords[:, 0]
            df['y'] = coords[:, 1]
    
    # Display useful info about the data
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
    st.info(f"""
    **Features:** Interactive 2D projection • Automatic clustering • Rich data tables • Advanced filtering
    
    **Data:** {len(df):,} points • UMAP projection • {file_path.name} ({file_size:.2f} MB)
    """)
    
    # Use the embedding-atlas component
    # It requires x,y coordinates - we've computed them above if needed
    result = embedding_atlas(
        data_frame=df,
        x="x",  # Required - we ensure this exists
        y="y",  # Required - we ensure this exists  
        text="text" if "text" in df.columns else None,
        labels="automatic",  # Automatic clustering and labeling
        show_table=True,
        show_charts=True,
        show_embedding=True
    )
    
    # Display selection info if any
    if result and result.get("predicate"):
        st.info(f"Current selection: {result['predicate']}")
        
else:
    st.error("No file selected or file not found")