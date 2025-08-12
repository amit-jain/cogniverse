#!/usr/bin/env python3
"""
Export Vespa embeddings to Parquet format for embedding-atlas visualization
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.search.backends.vespa_backend import VespaSearchBackend
from src.utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VespaEmbeddingExporter:
    """Export embeddings from Vespa to Parquet for embedding-atlas"""
    
    def __init__(self):
        config = get_config()
        self.vespa_backend = VespaSearchBackend(
            url=config.get("vespa_url", "http://localhost"),
            port=config.get("vespa_port", 8080),
            schema=config.get("vespa_schema", "video_frame")
        )
        
    def export_embeddings(
        self,
        output_path: str,
        max_documents: Optional[int] = None,
        video_id: Optional[str] = None,
        query: Optional[str] = None,
        reduce_dims: bool = True,
        dim_reduction_method: str = "umap"
    ) -> Path:
        """
        Export embeddings from Vespa to Parquet format
        
        Args:
            output_path: Output parquet file path
            max_documents: Maximum number of documents to export
            video_id: Filter by specific video ID
            query: Optional search query to filter documents
            reduce_dims: Whether to reduce dimensions for visualization
            dim_reduction_method: Method for dimension reduction (umap, tsne, pca)
            
        Returns:
            Path to the created parquet file
        """
        logger.info("Starting Vespa embedding export...")
        
        # Build YQL query
        yql = self._build_yql_query(video_id, query, max_documents)
        
        # Execute query
        logger.info(f"Executing YQL query: {yql}")
        response = self.vespa_backend.app.query(
            yql=yql,
            ranking="unranked",
            hits=max_documents or 10000
        )
        
        # Extract data
        documents = []
        embeddings = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # Extract embedding
            embedding = self._extract_embedding(fields)
            if embedding is None:
                continue
                
            embeddings.append(embedding)
            
            # Extract metadata
            doc = {
                "id": hit["id"],
                "video_id": fields.get("video_id", ""),
                "frame_number": fields.get("frame_number", 0),
                "timestamp": fields.get("timestamp", 0.0),
                "video_title": fields.get("video_title", ""),
                "frame_description": fields.get("frame_description", ""),
                "audio_transcript": fields.get("audio_transcript", ""),
                "source_type": fields.get("source_type", "video_frame"),
                "relevance_score": fields.get("relevance_score", 0.0),
            }
            
            # Add text representation for search/display
            doc["text"] = self._create_text_representation(fields)
            
            documents.append(doc)
        
        logger.info(f"Extracted {len(documents)} documents with embeddings")
        
        if not documents:
            logger.error("No documents found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(documents)
        
        # Add original embeddings
        embeddings_array = np.array(embeddings)
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        
        # Reduce dimensions if requested
        if reduce_dims and len(embeddings) > 10:
            logger.info(f"Reducing dimensions using {dim_reduction_method}...")
            reduced_embeddings = self._reduce_dimensions(
                embeddings_array, 
                method=dim_reduction_method
            )
            
            # Add 2D coordinates
            df["x"] = reduced_embeddings[:, 0]
            df["y"] = reduced_embeddings[:, 1]
            
            # If 3D reduction
            if reduced_embeddings.shape[1] > 2:
                df["z"] = reduced_embeddings[:, 2]
        
        # Add raw embeddings as array column (for nearest neighbor search)
        df["embedding"] = list(embeddings_array)
        
        # Add clustering labels (optional)
        if len(documents) > 20:
            from sklearn.cluster import KMeans
            n_clusters = min(10, len(documents) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df["cluster"] = kmeans.fit_predict(embeddings_array)
            logger.info(f"Added {n_clusters} cluster labels")
        
        # Save to Parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow table with proper schema
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        
        logger.info(f"Exported {len(df)} documents to {output_path}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return output_path
    
    def _build_yql_query(
        self, 
        video_id: Optional[str] = None,
        query: Optional[str] = None,
        max_docs: Optional[int] = None
    ) -> str:
        """Build YQL query for Vespa"""
        
        conditions = []
        
        if video_id:
            # Handle strategy-based filtering
            if video_id.startswith("strategy:"):
                strategy = video_id.replace("strategy:", "")
                conditions.append(f'ranking_strategy contains "{strategy}" or search_profile contains "{strategy}"')
            else:
                conditions.append(f'video_id contains "{video_id}"')
        
        if query:
            conditions.append(f'userQuery("{query}")')
        
        where_clause = " and ".join(conditions) if conditions else "true"
        
        yql = f"select * from video_frame where {where_clause}"
        
        if max_docs:
            yql += f" limit {max_docs}"
            
        return yql
    
    def _extract_embedding(self, fields: Dict) -> Optional[np.ndarray]:
        """Extract embedding from document fields"""
        
        # Try different embedding field names
        embedding_fields = [
            "embedding",
            "frame_embedding", 
            "video_embedding",
            "text_embedding",
            "colpali_embedding",
            "videoprism_embedding"
        ]
        
        for field_name in embedding_fields:
            if field_name in fields:
                embedding = fields[field_name]
                
                # Handle different formats
                if isinstance(embedding, dict) and "values" in embedding:
                    return np.array(embedding["values"])
                elif isinstance(embedding, list):
                    return np.array(embedding)
                elif isinstance(embedding, str):
                    # Binary format - decode
                    try:
                        # Assuming hex string format
                        if embedding.startswith("0x"):
                            embedding = embedding[2:]
                        bytes_data = bytes.fromhex(embedding)
                        return np.frombuffer(bytes_data, dtype=np.float32)
                    except:
                        logger.warning(f"Could not decode binary embedding from {field_name}")
                        
        return None
    
    def _create_text_representation(self, fields: Dict) -> str:
        """Create text representation for display/search"""
        parts = []
        
        if fields.get("video_title"):
            parts.append(f"Video: {fields['video_title']}")
        
        if fields.get("frame_description"):
            parts.append(f"Frame: {fields['frame_description']}")
            
        if fields.get("audio_transcript"):
            parts.append(f"Audio: {fields['audio_transcript']}")
            
        if fields.get("timestamp"):
            parts.append(f"Time: {fields['timestamp']:.1f}s")
            
        return " | ".join(parts) if parts else f"Frame {fields.get('frame_number', 'N/A')}"
    
    def _reduce_dimensions(
        self, 
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization"""
        
        if method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42
            )
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        logger.info(f"Reducing from {embeddings.shape[1]} to {n_components} dimensions...")
        reduced = reducer.fit_transform(embeddings)
        
        return reduced


def main():
    parser = argparse.ArgumentParser(description="Export Vespa embeddings for visualization")
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/embeddings/vespa_embeddings.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--max-documents",
        "-n",
        type=int,
        default=5000,
        help="Maximum number of documents to export"
    )
    parser.add_argument(
        "--video-id",
        "-v",
        help="Filter by specific video ID"
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Search query to filter documents"
    )
    parser.add_argument(
        "--reduction-method",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="Dimension reduction method"
    )
    parser.add_argument(
        "--no-reduction",
        action="store_true",
        help="Skip dimension reduction"
    )
    
    args = parser.parse_args()
    
    exporter = VespaEmbeddingExporter()
    
    output_path = exporter.export_embeddings(
        output_path=args.output,
        max_documents=args.max_documents,
        video_id=args.video_id,
        query=args.query,
        reduce_dims=not args.no_reduction,
        dim_reduction_method=args.reduction_method
    )
    
    if output_path:
        print(f"\n✅ Export complete: {output_path}")
        print(f"\n📊 Visualize with embedding-atlas:")
        print(f"   embedding-atlas {output_path}")
        print(f"\n🐍 Or in Python:")
        print(f"   from embedding_atlas import EmbeddingAtlasWidget")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_parquet('{output_path}')")
        print(f"   EmbeddingAtlasWidget(df)")
    else:
        print("\n❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()