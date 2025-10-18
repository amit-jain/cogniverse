#!/usr/bin/env python3
"""
Export embeddings from any backend using the backend abstraction.
Replaces the Vespa-specific export_vespa_embeddings.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogniverse_core.interfaces.backend import SearchBackend
from src.backends.vespa.search_backend import VespaSearchBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackendEmbeddingExporter:
    """Export embeddings from any backend implementation"""
    
    def __init__(self, backend: SearchBackend):
        """
        Initialize exporter with a backend instance.
        
        Args:
            backend: Any SearchBackend implementation
        """
        self.backend = backend
    
    def export_embeddings(
        self,
        output_path: str,
        schema: str = "video_frame",
        profile: Optional[str] = None,
        max_documents: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        reduce_dims: bool = True,
        dim_reduction_method: str = "umap"
    ) -> Path:
        """
        Export embeddings from backend and save to parquet.
        
        Args:
            output_path: Output parquet file path
            schema: Schema/index to export from
            max_documents: Maximum documents to export
            filters: Optional filters for documents
            reduce_dims: Whether to compute 2D projections
            dim_reduction_method: Method for dimension reduction
            
        Returns:
            Path to output file if successful, None otherwise
        """
        logger.info(f"Starting embedding export from {schema}...")
        
        # Get documents from backend
        documents = self.backend.export_embeddings(
            schema=schema,
            max_documents=max_documents,
            filters=filters,
            include_embeddings=True
        )
        
        if not documents:
            logger.error("No documents found")
            return None
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Process documents and extract embeddings
        processed_docs = []
        embeddings = []
        
        for doc in documents:
            # Extract embedding
            embedding = self._extract_embedding(doc)
            if embedding is not None:
                embeddings.append(embedding)
                
                # Create processed document
                processed_doc = self._extract_fields(doc, schema)
                processed_docs.append(processed_doc)
        
        if not processed_docs:
            logger.error("No documents with embeddings found")
            return None
        
        logger.info(f"Extracted {len(processed_docs)} documents with embeddings")
        
        # Create DataFrame
        df = pd.DataFrame(processed_docs)
        
        # Add embeddings
        embeddings_array = np.array(embeddings)
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        
        # Add metadata about the encoder/profile used
        # This helps with query comparison later
        df["embedding_dim"] = embeddings_array.shape[1]
        df["export_schema"] = schema
        
        # Store the profile if provided - CRITICAL for query encoding
        if profile:
            df["encoder_profile"] = profile
            logger.info(f"Stored encoder profile: {profile}")
        
        # Reduce dimensions if requested
        if reduce_dims and len(embeddings) >= 3:  # Need at least 3 points
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
        
        # Add raw embeddings as array column
        df["embedding"] = list(embeddings_array)
        
        # Add clustering labels if enough documents
        if len(processed_docs) > 20:
            from sklearn.cluster import KMeans
            n_clusters = min(10, len(processed_docs) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df["cluster"] = kmeans.fit_predict(embeddings_array)
            logger.info(f"Added {n_clusters} cluster labels")
        
        # Save to Parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        
        logger.info(f"Exported {len(df)} documents to {output_path}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return output_path
    
    def _extract_embedding(self, doc: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract embedding from document"""
        
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
            if field_name in doc:
                embedding = doc[field_name]
                
                # Handle different formats
                if isinstance(embedding, dict):
                    if "values" in embedding:
                        return np.array(embedding["values"])
                    elif "blocks" in embedding:
                        # Tensor format
                        blocks = embedding["blocks"]
                        values = []
                        for block_key in sorted(blocks.keys(), key=int):
                            values.extend(blocks[block_key])
                        return np.array(values)
                elif isinstance(embedding, list):
                    return np.array(embedding)
                elif isinstance(embedding, np.ndarray):
                    return embedding
                elif isinstance(embedding, str):
                    # Binary format - decode
                    try:
                        if embedding.startswith("0x"):
                            embedding = embedding[2:]
                        bytes_data = bytes.fromhex(embedding)
                        return np.frombuffer(bytes_data, dtype=np.float32)
                    except:
                        logger.warning(f"Could not decode binary embedding from {field_name}")
                        
        return None
    
    def _extract_fields(self, doc: Dict[str, Any], schema: str) -> Dict[str, Any]:
        """Extract relevant fields based on schema"""
        
        # Common fields
        extracted = {
            "id": doc.get("id", ""),
            "source_type": schema,
            "relevance_score": doc.get("relevance_score", 0.0),
        }
        
        # Schema-specific fields
        if schema == "video_frame":
            extracted.update({
                "video_id": doc.get("video_id", ""),
                "frame_number": doc.get("frame_number", 0),
                "timestamp": doc.get("timestamp", 0.0),
                "video_title": doc.get("video_title", ""),
                "frame_description": doc.get("frame_description", ""),
                "audio_transcript": doc.get("audio_transcript", ""),
            })
        elif schema == "document":
            extracted.update({
                "doc_id": doc.get("doc_id", doc.get("document_id", "")),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "author": doc.get("author", ""),
                "category": doc.get("category", ""),
                "tags": doc.get("tags", []),
                "created_date": doc.get("created_date", ""),
            })
        elif schema == "product":
            extracted.update({
                "product_id": doc.get("product_id", ""),
                "product_name": doc.get("product_name", ""),
                "description": doc.get("description", ""),
                "category": doc.get("category", ""),
                "price": doc.get("price", 0.0),
                "brand": doc.get("brand", ""),
                "rating": doc.get("rating", 0.0),
            })
        else:
            # Generic extraction - include all non-embedding fields
            for key, value in doc.items():
                if key not in extracted and not key.endswith("embedding"):
                    extracted[key] = value
        
        # Create text representation
        extracted["text"] = self._create_text_representation(extracted)
        
        return extracted
    
    def _create_text_representation(self, fields: Dict[str, Any]) -> str:
        """Create text representation for display"""
        parts = []
        
        if "video_title" in fields:
            parts.append(f"Video: {fields['video_title']}")
            if "frame_description" in fields:
                parts.append(f"Frame: {fields['frame_description']}")
        elif "title" in fields:
            parts.append(f"Title: {fields['title']}")
            if "content" in fields:
                parts.append(f"Content: {fields['content'][:200]}")
        elif "product_name" in fields:
            parts.append(f"Product: {fields['product_name']}")
            if "description" in fields:
                parts.append(f"Description: {fields['description']}")
        
        return " | ".join(parts) if parts else ""
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization"""
        
        if method == "umap":
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42
            )
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        logger.info(f"Reducing from {embeddings.shape[1]} to {n_components} dimensions...")
        reduced = reducer.fit_transform(embeddings)
        
        return reduced


def get_backend(backend_type: str, **kwargs) -> SearchBackend:
    """
    Factory function to create backend instances.
    
    Args:
        backend_type: Type of backend ('vespa', 'elasticsearch', etc.)
        **kwargs: Backend-specific configuration
        
    Returns:
        SearchBackend instance
    """
    if backend_type.lower() == "vespa":
        # Parse URL and port
        url = kwargs.get("url", "http://localhost:8080")
        # Simple parsing - just use localhost for now
        base_url = "http://localhost"
        port = 8080
            
        schema = kwargs.get("schema", "video_frame")
        
        return VespaSearchBackend(
            vespa_url=base_url if "://" in base_url else f"http://{base_url}",
            vespa_port=port,
            schema_name=schema,
            profile=None,  # No profile needed for export
            strategy=None,  # No strategy needed for export
            query_encoder=None,  # No encoder needed for export
            enable_metrics=False,  # Disable for export
            enable_connection_pool=False  # Disable for export
        )
    # Add more backends here as needed
    # elif backend_type.lower() == "elasticsearch":
    #     return ElasticsearchBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def main():
    parser = argparse.ArgumentParser(description="Export embeddings from any backend")
    parser.add_argument(
        "--backend",
        default="vespa",
        choices=["vespa"],  # Add more as implemented
        help="Backend type to use"
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8080",
        help="Backend URL"
    )
    parser.add_argument(
        "--schema",
        default="video_frame",
        help="Schema/index to export from"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/embeddings/backend_embeddings.parquet",
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
        "--filter-key",
        help="Filter field name (e.g., video_id)"
    )
    parser.add_argument(
        "--filter-value",
        help="Filter field value"
    )
    parser.add_argument(
        "--profile",
        help="Strategy profile used for encoding (e.g., frame_based_colpali, direct_video_frame)"
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
    
    # Create backend instance
    backend = get_backend(
        args.backend,
        url=args.backend_url,
        schema=args.schema
    )
    
    # Create filters if provided
    filters = None
    if args.filter_key and args.filter_value:
        filters = {args.filter_key: args.filter_value}
    
    # Create exporter
    exporter = BackendEmbeddingExporter(backend)
    
    # Export embeddings
    output_path = exporter.export_embeddings(
        output_path=args.output,
        schema=args.schema,
        profile=args.profile,
        max_documents=args.max_documents,
        filters=filters,
        reduce_dims=not args.no_reduction,
        dim_reduction_method=args.reduction_method
    )
    
    if output_path:
        print(f"\n✅ Export complete: {output_path}")
        print("\n📊 Visualize with embedding-atlas:")
        print("   uv run streamlit run scripts/atlas_viewer.py")
        print("\n🎨 Or with custom 3D visualization:")
        print("   uv run streamlit run scripts/simple_atlas.py")
    else:
        print("\n❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
