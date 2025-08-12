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
    
    # Check if this is a comparison file with queries
    has_query = "is_query" in df.columns and df["is_query"].any()
    query_info = ""
    if has_query:
        query_df = df[df["is_query"] == True]
        num_queries = len(query_df)
        if num_queries > 0:
            if num_queries == 1:
                query_text = query_df.iloc[0].get('text', 'N/A').replace('QUERY: ', '')
                query_info = f" • **1 query included:** '{query_text[:50]}...'"
            else:
                query_info = f" • **{num_queries} queries included**"
    
    st.info(f"""
    **Features:** Interactive 2D projection • Automatic clustering • Rich data tables • Advanced filtering
    
    **Data:** {len(df):,} points • UMAP projection • {file_path.name} ({file_size:.2f} MB){query_info}
    """)
    
    # Prepare data for visualization with enhanced query support
    # Add a category column for queries if present
    if has_query:
        # Create category column to distinguish queries from documents
        df['point_type'] = df['is_query'].apply(lambda x: 'Query' if x else 'Document')
        
        # If multiple queries, give them unique labels
        query_indices = df[df['is_query'] == True].index
        for i, idx in enumerate(query_indices):
            df.loc[idx, 'point_type'] = f'Query {i+1}'
        
        # The labels parameter will handle coloring
        labels_column = "point_type"
    else:
        # Use automatic clustering for color
        labels_column = "automatic"
    
    # Use the embedding-atlas component
    # It requires x,y coordinates - we've computed them above if needed
    result = embedding_atlas(
        data_frame=df,
        x="x",  # Required - we ensure this exists
        y="y",  # Required - we ensure this exists  
        text="text" if "text" in df.columns else None,
        labels=labels_column,  # This handles both labeling and coloring
        show_table=True,
        show_charts=True,
        show_embedding=True
    )
    
    # Display additional query information if present
    if has_query:
        st.markdown("---")
        st.subheader("🎯 Query Analysis")
        
        query_df = df[df["is_query"] == True]
        doc_df = df[df["is_query"] != True]
        
        # Display query details
        for i, (idx, query_row) in enumerate(query_df.iterrows()):
            query_text = query_row.get('text', '').replace('QUERY: ', '')
            st.write(f"**Query {i+1}:** {query_text}")
            
            # Check for similarity columns for this query
            similarity_col = f"query_similarity_{idx}"
            if similarity_col in doc_df.columns:
                # Show top similar documents
                top_k = min(3, len(doc_df))
                top_docs = doc_df.nlargest(top_k, similarity_col)
                
                with st.expander(f"Top {top_k} similar documents"):
                    for _, doc in top_docs.iterrows():
                        # Get display text
                        if "video_title" in doc and pd.notna(doc["video_title"]):
                            display_text = f"{doc['video_title']}"
                            if "frame_number" in doc:
                                display_text += f" (Frame {doc['frame_number']})"
                        elif "text" in doc and pd.notna(doc["text"]):
                            display_text = doc["text"][:100] + "..."
                        else:
                            display_text = f"Document {doc.name}"
                        
                        st.write(f"• {display_text} (Similarity: {doc[similarity_col]:.3f})")
    
    # Display selection info if any
    if result and result.get("predicate"):
        st.info(f"Current selection: {result['predicate']}")
        
else:
    st.error("No file selected or file not found")