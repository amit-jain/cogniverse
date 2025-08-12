#!/usr/bin/env python3
"""
Simple working embedding atlas visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

# Import dimension reduction tools if available
try:
    from umap import UMAP
    from sklearn.cluster import KMeans
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.warning("UMAP not available. Install with: pip install umap-learn scikit-learn")

# Fix for environment variable
if 'STREAMLIT_EMBEDDING_FILE' in os.environ:
    default_file = os.environ['STREAMLIT_EMBEDDING_FILE']
else:
    default_file = None

st.set_page_config(page_title="Embedding Atlas", layout="wide")

# Title
st.title("🗺️ Embedding Visualization")

# Get file path from session state or command line
file_path = None

# Check if launched from dashboard with file in session state
if 'embedding_atlas_file' in st.session_state:
    file_path = Path(st.session_state.embedding_atlas_file)
    st.sidebar.success(f"Loaded from dashboard: {file_path.name}")
elif default_file:
    file_path = Path(default_file)
    if file_path.exists():
        st.sidebar.success(f"Loaded from environment: {file_path.name}")

# If not from dashboard, show file selector
if not file_path:
    with st.sidebar:
        st.header("📁 Select Data")
        
        # List available parquet files
        parquet_dir = Path("outputs/embeddings")
        if parquet_dir.exists():
            files = sorted(parquet_dir.glob("*.parquet"))
            if files:
                file_path = st.selectbox(
                    "Choose embedding file:",
                    files,
                    format_func=lambda x: x.name,
                    index=len(files)-1  # Default to most recent
                )
            else:
                st.error("No parquet files found in outputs/embeddings/")
                st.stop()
        else:
            st.error("outputs/embeddings/ directory not found")
            st.stop()

# Load the selected data
@st.cache_data
def load_data(path):
    """Load embeddings from parquet file"""
    df = pd.read_parquet(path)
    
    # Check if we need to compute 2D projection
    if 'x' not in df.columns or 'y' not in df.columns:
        if 'embedding' in df.columns and len(df) > 0:
            st.info("Computing 2D projection for visualization...")
            
            try:
                import numpy as np
                from umap import UMAP
                from sklearn.cluster import KMeans
                
                # Extract embeddings
                embeddings = np.vstack(df['embedding'].values)
                
                # Compute UMAP projection
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(df) - 1),
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                coords = reducer.fit_transform(embeddings)
                
                # Add coordinates to dataframe
                df['x'] = coords[:, 0]
                df['y'] = coords[:, 1]
                
                # Add clustering if not present
                if 'cluster' not in df.columns and len(df) > 20:
                    n_clusters = min(8, len(df) // 10)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df['cluster'] = kmeans.fit_predict(embeddings)
                    
                st.success("✅ 2D projection computed successfully!")
                
            except Exception as e:
                st.error(f"Failed to compute 2D projection: {str(e)}")
                st.stop()
        else:
            st.error("No embedding column found in the data. Cannot create visualization.")
            st.stop()
    
    return df

if file_path and file_path.exists():
    df = load_data(file_path)
    
    # Display useful info about the data
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
    # Count queries if present
    num_queries = df["is_query"].sum() if "is_query" in df.columns else 0
    
    info_text = f"""
    **Features:** 3D scatter with rotation • 2D UMAP projection • Rich hover details • Cluster visualization
    
    **Data:** {len(df):,} points
    """
    
    if num_queries > 0:
        info_text += f" • **{num_queries} queries** projected"
    
    info_text += f" • UMAP reduction • {file_path.name} ({file_size:.2f} MB)"
    
    st.info(info_text)
else:
    st.error(f"File not found: {file_path}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Dynamically detect categorical columns for coloring
    categorical_cols = []
    if 'cluster' in df.columns:
        categorical_cols.append('cluster')
    
    # Add schema-specific grouping columns
    for col in ['video_id', 'category', 'author', 'brand', 'doc_id', 'product_id']:
        if col in df.columns:
            categorical_cols.append(col)
    
    # Add any other string columns with reasonable cardinality
    for col in df.columns:
        if col not in categorical_cols and df[col].dtype == 'object':
            # Skip columns that contain arrays or complex objects
            try:
                if df[col].nunique() < 50:  # Reasonable limit for color categories
                    categorical_cols.append(col)
            except (TypeError, ValueError):
                # Skip columns that can't be hashed (like arrays)
                continue
    
    if categorical_cols:
        color_by = st.selectbox(
            "Color by",
            options=categorical_cols,
            index=0
        )
    else:
        color_by = None
    
    point_size = st.slider("Point size", 2, 20, 8)
    
    st.header("Stats")
    st.metric("Total Points", len(df))
    if 'cluster' in df.columns:
        st.metric("Clusters", df['cluster'].nunique())
    
    # Show schema-specific stats
    for col in ['video_id', 'category', 'author', 'brand']:
        if col in df.columns:
            st.metric(f"Unique {col.replace('_', ' ').title()}s", df[col].nunique())

# Prepare color values once for all views
if color_by in df.columns:
    if df[color_by].dtype == 'object' or isinstance(df[color_by].iloc[0], str):
        # Map string categories to numbers
        unique_vals = df[color_by].unique()
        color_map = {val: i for i, val in enumerate(unique_vals)}
        color_values = df[color_by].map(color_map)
        colorbar_title = f"{color_by} (n={len(unique_vals)})"
    else:
        color_values = df[color_by]
        colorbar_title = color_by
else:
    color_values = df.index
    colorbar_title = "Index"

# Main visualization
tab1, tab2, tab3 = st.tabs(["🌐 3D View", "📍 2D View", "📊 Data"])

with tab1:
    # Create 3D visualization
    if 'z' not in df.columns:
        # Create z coordinate based on cluster
        if 'cluster' in df.columns:
            df['z'] = df['cluster'] * 0.5 + np.random.randn(len(df)) * 0.2
        else:
            df['z'] = np.random.randn(len(df))
    
    # Build rich hover text with all available information
    hover_texts = []
    for i, row in df.iterrows():
        hover_parts = []
        
        # Schema-specific primary display
        # Video frame schema
        if 'video_title' in df.columns and pd.notna(row.get('video_title')):
            hover_parts.append(f"<b>{row['video_title']}</b>")
            if 'frame_number' in df.columns:
                hover_parts.append(f"Frame: {row.get('frame_number', 'N/A')}")
            if 'timestamp' in df.columns and pd.notna(row.get('timestamp')):
                hover_parts.append(f"Time: {row['timestamp']:.2f}s")
            if 'frame_description' in df.columns and pd.notna(row.get('frame_description')):
                desc = str(row['frame_description'])
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                hover_parts.append(f"<i>{desc}</i>")
        
        # Document schema
        elif 'title' in df.columns and pd.notna(row.get('title')):
            hover_parts.append(f"<b>{row['title']}</b>")
            if 'author' in df.columns and pd.notna(row.get('author')):
                hover_parts.append(f"Author: {row['author']}")
            if 'content' in df.columns and pd.notna(row.get('content')):
                content = str(row['content'])[:150]
                hover_parts.append(f"<i>{content}...</i>")
        
        # Product schema
        elif 'product_name' in df.columns and pd.notna(row.get('product_name')):
            hover_parts.append(f"<b>{row['product_name']}</b>")
            if 'brand' in df.columns and pd.notna(row.get('brand')):
                hover_parts.append(f"Brand: {row['brand']}")
            if 'price' in df.columns and pd.notna(row.get('price')):
                hover_parts.append(f"Price: ${row['price']:.2f}")
            if 'rating' in df.columns and pd.notna(row.get('rating')):
                hover_parts.append(f"Rating: {row['rating']:.1f}⭐")
        
        # Generic text field if available
        elif 'text' in df.columns and pd.notna(row.get('text')):
            text = str(row['text'])
            if len(text) > 150:
                text = text[:147] + "..."
            hover_parts.append(text)
        
        # Add cluster info
        if 'cluster' in df.columns:
            hover_parts.append(f"Cluster: {row.get('cluster', 'N/A')}")
        
        # Add relevance score if available
        if 'relevance_score' in df.columns and pd.notna(row.get('relevance_score')) and row['relevance_score'] > 0:
            hover_parts.append(f"Relevance: {row['relevance_score']:.3f}")
        
        # Add ID info
        if 'id' in df.columns:
            hover_parts.append(f"ID: {row.get('id', i)}")
        
        hover_texts.append("<br>".join(hover_parts))
    
    # Check if there's a query point to highlight
    has_query = "is_query" in df.columns and df["is_query"].any()
    
    if has_query:
        # Separate query and document points
        doc_mask = df["is_query"] == False
        query_mask = df["is_query"] == True
        
        doc_df = df[doc_mask].copy()  # Make explicit copies
        query_df = df[query_mask].copy()
        
        # Ensure query has z-coordinate
        if 'z' not in query_df.columns or query_df['z'].isna().any():
            # Place query slightly above the mean z-level for visibility
            mean_z = doc_df['z'].mean() if 'z' in doc_df.columns and len(doc_df) > 0 else 0
            query_df['z'] = mean_z + 1.0  # Slightly elevated for visibility
        
        # Get hover texts for documents
        doc_hover_texts = [hover_texts[i] for i, is_doc in enumerate(doc_mask) if is_doc]
        doc_color_values = [color_values[i] for i, is_doc in enumerate(doc_mask) if is_doc]
        
        # Debug info
        if len(doc_df) == 0:
            st.warning("No documents found in the data!")
            fig = go.Figure()  # Empty figure if no documents
        else:
            # Ensure doc_df has required columns with proper types
            if 'x' not in doc_df.columns or 'y' not in doc_df.columns:
                st.error("Document data missing x,y coordinates!")
                fig = go.Figure()
            else:
                # Create figure with document points
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=doc_df['x'],
                        y=doc_df['y'],
                        z=doc_df['z'],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=doc_color_values,
                            colorscale='Turbo',
                            showscale=True,
                            colorbar=dict(
                                title=colorbar_title,
                                thickness=15,
                                len=0.7,
                                x=1.02
                            ),
                            opacity=0.7,  # Slightly transparent for documents
                            line=dict(width=0.5, color='white')
                        ),
                        text=doc_hover_texts,
                        hovertemplate='%{text}<extra></extra>',
                        name='Documents'
                    )
                ])
        
        # Add query point(s) with different colors for each
        if not query_df.empty:
            # Define colors for multiple queries
            query_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
            # Use only valid 3D scatter symbols
            query_symbols = ['diamond', 'square', 'circle', 'x', 'cross', 'diamond-open', 'square-open', 'circle-open']
            
            for q_idx, (idx, row) in enumerate(query_df.iterrows()):
                query_text = row.get('text', 'Query').replace('QUERY: ', '')
                # Truncate long queries for display
                display_text = query_text[:30] + '...' if len(query_text) > 30 else query_text
                
                # Build hover text with similarity info if available
                hover_parts = [f"<b>🎯 QUERY {q_idx + 1}</b>", query_text]
                
                # Add similarity scores if available
                similarity_cols = [col for col in row.index if col.startswith('query_similarity_')]
                if similarity_cols:
                    for col in similarity_cols:
                        if pd.notna(row[col]):
                            hover_parts.append(f"Self-similarity: {row[col]:.3f}")
                            break
                
                hover_text = "<br>".join(hover_parts)
                
                # Get color and symbol for this query
                color = query_colors[q_idx % len(query_colors)]
                symbol = query_symbols[q_idx % len(query_symbols)]
                
                # Map colors to valid darker versions for borders
                if color == 'red':
                    border_color = 'darkred'
                elif color == 'blue':
                    border_color = 'darkblue'
                elif color == 'green':
                    border_color = 'darkgreen'
                elif color == 'purple':
                    border_color = 'indigo'  # Use indigo instead of darkpurple
                elif color == 'orange':
                    border_color = 'darkorange'
                elif color == 'cyan':
                    border_color = 'darkcyan'
                elif color == 'magenta':
                    border_color = 'darkmagenta'
                else:  # yellow
                    border_color = 'gold'
                
                fig.add_trace(go.Scatter3d(
                    x=[row['x']],
                    y=[row['y']],
                    z=[row['z']],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=color,
                        symbol=symbol,
                        line=dict(color=border_color, width=2)
                    ),
                    text=[f'Q{q_idx + 1}'],
                    textposition="top center",
                    textfont=dict(size=12, color=color),
                    hovertext=[hover_text],
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=f'Query {q_idx + 1}: {display_text}'
                ))
                
                # Add connecting lines to top similar documents for each query
                similarity_col = f'query_similarity_{idx}'
                if similarity_col in doc_df.columns:
                    top_k = min(3, len(doc_df))  # Show top 3 for each query to avoid clutter
                    top_similar = doc_df.nlargest(top_k, similarity_col)
                    
                    for doc_idx, doc_row in top_similar.iterrows():
                        similarity = doc_row[similarity_col]
                        # Line opacity based on similarity
                        line_opacity = min(0.4, similarity * 0.5)
                        
                        # Convert color to RGB for transparency
                        if color == 'red':
                            rgba = f'rgba(255, 0, 0, {line_opacity})'
                        elif color == 'blue':
                            rgba = f'rgba(0, 0, 255, {line_opacity})'
                        elif color == 'green':
                            rgba = f'rgba(0, 255, 0, {line_opacity})'
                        elif color == 'purple':
                            rgba = f'rgba(128, 0, 128, {line_opacity})'
                        elif color == 'orange':
                            rgba = f'rgba(255, 165, 0, {line_opacity})'
                        elif color == 'cyan':
                            rgba = f'rgba(0, 255, 255, {line_opacity})'
                        elif color == 'magenta':
                            rgba = f'rgba(255, 0, 255, {line_opacity})'
                        else:  # yellow
                            rgba = f'rgba(255, 255, 0, {line_opacity})'
                        
                        fig.add_trace(go.Scatter3d(
                            x=[row['x'], doc_row['x']],
                            y=[row['y'], doc_row['y']],
                            z=[row['z'], doc_row['z']],
                            mode='lines',
                            line=dict(
                                color=rgba,
                                width=1.5 * similarity,  # Thicker lines for higher similarity
                                dash='dot'
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
    else:
        # Standard visualization without query
        fig = go.Figure(data=[go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_values,  # Use pre-computed color values
                colorscale='Turbo',  # More vibrant color scale
                showscale=True,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                opacity=0.9,
                line=dict(width=0.5, color='white')  # Add white outline for better visibility
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>'
        )])
    
    # Better 3D layout with improved aesthetics
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showticklabels=False,
                showspikes=False,
                title=''
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showticklabels=False,
                showspikes=False,
                title=''
            ),
            zaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showticklabels=False,
                showspikes=False,
                title=''
            ),
            bgcolor='#0e1117',  # Dark background matching Streamlit
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Better initial camera angle
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        title=dict(
            text='3D Embedding Space',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("🖱️ Drag to rotate • Scroll to zoom • Hover for details")

with tab2:
    # 2D scatter plot - handle queries separately if present
    if has_query:
        # Plot documents
        doc_hover_texts_2d = [hover_texts[i] for i, is_doc in enumerate(doc_mask) if is_doc]
        doc_color_values_2d = [color_values[i] for i, is_doc in enumerate(doc_mask) if is_doc]
        
        fig2 = go.Figure(data=[go.Scatter(
            x=doc_df['x'],
            y=doc_df['y'],
            mode='markers',
            marker=dict(
                size=point_size,
                color=doc_color_values_2d,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                opacity=0.7
            ),
            text=doc_hover_texts_2d,
            hovertemplate='%{text}<extra></extra>',
            name='Documents'
        )])
        
        # Add query points with different colors
        query_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        # Use valid 2D scatter symbols
        query_symbols = ['diamond', 'star', 'square', 'circle', 'x', 'cross', 'triangle-up', 'pentagon']
        
        for q_idx, (idx, row) in enumerate(query_df.iterrows()):
            query_text = row.get('text', 'Query').replace('QUERY: ', '')
            display_text = query_text[:30] + '...' if len(query_text) > 30 else query_text
            
            color = query_colors[q_idx % len(query_colors)]
            symbol = query_symbols[q_idx % len(query_symbols)]
            
            # Map colors to valid darker versions for borders (2D)
            if color == 'red':
                border_color_2d = 'darkred'
            elif color == 'blue':
                border_color_2d = 'darkblue'
            elif color == 'green':
                border_color_2d = 'darkgreen'
            elif color == 'purple':
                border_color_2d = 'indigo'
            elif color == 'orange':
                border_color_2d = 'darkorange'
            elif color == 'cyan':
                border_color_2d = 'darkcyan'
            elif color == 'magenta':
                border_color_2d = 'darkmagenta'
            else:  # yellow
                border_color_2d = 'gold'
            
            fig2.add_trace(go.Scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol,
                    line=dict(color=border_color_2d, width=2)
                ),
                text=[f'Q{q_idx + 1}'],
                textposition="top center",
                textfont=dict(size=10, color=color),
                hovertext=[f"<b>🎯 QUERY {q_idx + 1}</b><br>{query_text}"],
                hovertemplate='%{hovertext}<extra></extra>',
                name=f'Query {q_idx + 1}: {display_text}'
            ))
    else:
        # Standard visualization without queries
        fig2 = go.Figure(data=[go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_values,  # Use pre-computed color values
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                opacity=0.7
            ),
            text=hover_texts,  # Reuse the same rich hover texts from 3D view
            hovertemplate='%{text}<extra></extra>'
        )])
    
    fig2.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=600
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Data table - show most relevant columns based on schema
    display_columns = []
    
    # Schema-specific priority columns
    if 'video_title' in df.columns:  # Video frame schema
        priority_cols = ['id', 'video_id', 'video_title', 'frame_number', 'timestamp', 
                         'frame_description', 'audio_transcript', 'cluster', 'relevance_score']
    elif 'title' in df.columns and 'author' in df.columns:  # Document schema
        priority_cols = ['id', 'doc_id', 'title', 'author', 'category', 'content',
                         'created_date', 'cluster', 'relevance_score']
    elif 'product_name' in df.columns:  # Product schema
        priority_cols = ['id', 'product_id', 'product_name', 'brand', 'category',
                         'description', 'price', 'rating', 'cluster', 'relevance_score']
    else:  # Generic schema
        priority_cols = ['id', 'text', 'cluster', 'relevance_score']
        # Add any string columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in priority_cols and col not in ['embedding', 'x', 'y', 'z']:
                priority_cols.append(col)
    
    for col in priority_cols:
        if col in df.columns:
            display_columns.append(col)
    
    # Add x, y coordinates at the end
    if 'x' in df.columns and 'y' in df.columns:
        display_columns.extend(['x', 'y'])
    
    # Show the dataframe with selected columns
    if display_columns:
        st.write(f"Showing {min(100, len(df))} of {len(df)} total records")
        st.dataframe(df[display_columns].head(100), use_container_width=True)
    else:
        st.dataframe(df.head(100), use_container_width=True)