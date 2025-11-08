#!/usr/bin/env python3
"""
Export Vespa embeddings to Parquet format for embedding-atlas visualization
"""

import argparse
import logging

# Fix protobuf issue - must be before other imports
import os
import sys
from pathlib import Path
from typing import Dict, Optional

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.common.config_utils import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VespaEmbeddingExporter:
    """Export embeddings from Vespa to Parquet for embedding-atlas"""

    def __init__(self, schema_name: str = "video_frame"):
        from cogniverse_core.config.utils import create_default_config_manager, get_config
        config_manager = create_default_config_manager()
        config = get_config(tenant_id="default", config_manager=config_manager)
        # Use simpler Vespa client directly for export
        from vespa.application import Vespa
        self.vespa = Vespa(url=f"http://localhost:{config.get('vespa_port', 8080)}")
        self.schema_name = schema_name
        
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
        
        # Use visit API to get documents with embeddings
        visit_url = f"http://localhost:8080/document/v1/video/{self.schema_name}/docid"
        
        # Build selection criteria
        selection = "true"
        if video_id:
            if video_id.startswith("strategy:"):
                strategy = video_id.replace("strategy:", "")
                selection = f'{self.schema_name}.ranking_strategy="{strategy}" or {self.schema_name}.search_profile="{strategy}"'
            else:
                selection = f'{self.schema_name}.video_id="{video_id}"'
        
        params = {
            "selection": selection,
            "wantedDocumentCount": max_documents or 10000,
            "fieldSet": f"{self.schema_name}:[document]"  # Get all document fields including tensors
        }
        
        logger.info(f"Visiting documents with selection: {selection}")
        
        import requests
        response = requests.get(visit_url, params=params)
        response_json = response.json()
        
        # Extract data
        documents = []
        embeddings = []
        
        # Parse visit response
        if "documents" not in response_json:
            logger.error(f"No documents in response: {response_json}")
            return None
            
        for doc in response_json.get("documents", []):
            fields = doc.get("fields", {})
            
            # Extract embedding
            embedding = self._extract_embedding(fields)
            if embedding is None:
                continue
                
            embeddings.append(embedding)
            
            # Extract metadata based on schema
            doc_data = self._extract_schema_fields(doc, fields)
            
            # Add text representation for search/display
            doc_data["text"] = self._create_text_representation(fields)
            
            documents.append(doc_data)
        
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
        if reduce_dims and len(embeddings) >= 3:  # Need at least 3 points for meaningful projection
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
        
        yql = f"select * from {self.schema_name} where {where_clause}"
        
        if max_docs:
            yql += f" limit {max_docs}"
            
        return yql
    
    def _extract_embedding(self, fields: Dict) -> Optional[np.ndarray]:
        """Extract embedding from document fields"""
        
        # Try different embedding field names (skip binary embeddings)
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
                
                # Handle Vespa tensor format
                if isinstance(embedding, dict):
                    if "blocks" in embedding:
                        # Tensor format from visit API
                        blocks = embedding["blocks"]
                        values = []
                        for block_key in sorted(blocks.keys(), key=int):
                            values.extend(blocks[block_key])
                        return np.array(values)
                    elif "values" in embedding:
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
    
    def _extract_schema_fields(self, doc: Dict, fields: Dict) -> Dict:
        """Extract fields based on schema type"""
        doc_data = {
            "id": doc.get("id", ""),
            "relevance_score": fields.get("relevance_score", 0.0),
        }
        
        if self.schema_name == "video_frame":
            # Video frame specific fields
            doc_data.update({
                "video_id": fields.get("video_id", ""),
                "frame_number": fields.get("frame_number", 0),
                "timestamp": fields.get("timestamp", 0.0),
                "video_title": fields.get("video_title", ""),
                "frame_description": fields.get("frame_description", ""),
                "audio_transcript": fields.get("audio_transcript", ""),
                "source_type": fields.get("source_type", "video_frame"),
            })
        elif self.schema_name == "document":
            # Document schema fields
            doc_data.update({
                "doc_id": fields.get("doc_id", fields.get("document_id", "")),
                "title": fields.get("title", ""),
                "content": fields.get("content", ""),
                "author": fields.get("author", ""),
                "category": fields.get("category", ""),
                "tags": fields.get("tags", []),
                "created_date": fields.get("created_date", ""),
            })
        elif self.schema_name == "product":
            # Product schema fields
            doc_data.update({
                "product_id": fields.get("product_id", ""),
                "product_name": fields.get("product_name", ""),
                "description": fields.get("description", ""),
                "category": fields.get("category", ""),
                "price": fields.get("price", 0.0),
                "brand": fields.get("brand", ""),
                "rating": fields.get("rating", 0.0),
            })
        else:
            # Generic extraction - include all non-tensor fields
            for key, value in fields.items():
                if not isinstance(value, dict) or "values" not in value:  # Skip tensor fields
                    doc_data[key] = value
        
        return doc_data
    
    def _create_text_representation(self, fields: Dict) -> str:
        """Create text representation for display/search"""
        parts = []
        
        if self.schema_name == "video_frame":
            if fields.get("video_title"):
                parts.append(f"Video: {fields['video_title']}")
            if fields.get("frame_description"):
                parts.append(f"Frame: {fields['frame_description']}")
            if fields.get("audio_transcript"):
                parts.append(f"Audio: {fields['audio_transcript']}")
            if fields.get("timestamp"):
                parts.append(f"Time: {fields['timestamp']:.1f}s")
            return " | ".join(parts) if parts else f"Frame {fields.get('frame_number', 'N/A')}"
            
        elif self.schema_name == "document":
            if fields.get("title"):
                parts.append(f"Title: {fields['title']}")
            if fields.get("content"):
                parts.append(fields['content'][:200])  # First 200 chars
            if fields.get("author"):
                parts.append(f"By: {fields['author']}")
            return " | ".join(parts) if parts else "Document"
            
        elif self.schema_name == "product":
            if fields.get("product_name"):
                parts.append(fields['product_name'])
            if fields.get("description"):
                parts.append(fields['description'][:200])
            if fields.get("brand"):
                parts.append(f"Brand: {fields['brand']}")
            if fields.get("price"):
                parts.append(f"${fields['price']}")
            return " | ".join(parts) if parts else "Product"
            
        else:
            # Generic - combine any text-like fields
            for key, value in fields.items():
                if isinstance(value, str) and len(value) > 0 and not key.endswith("_id"):
                    parts.append(f"{key}: {value[:100]}")
            return " | ".join(parts[:3]) if parts else "Document"
    
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
    parser.add_argument(
        "--schema",
        default="video_frame",
        help="Vespa schema to export from (default: video_frame)"
    )
    
    args = parser.parse_args()
    
    exporter = VespaEmbeddingExporter(schema_name=args.schema)
    
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
        print("\n📊 Visualize with embedding-atlas:")
        print(f"   embedding-atlas {output_path}")
        print("\n🐍 Or in Python:")
        print("   from embedding_atlas import EmbeddingAtlasWidget")
        print("   import pandas as pd")
        print(f"   df = pd.read_parquet('{output_path}')")
        print("   EmbeddingAtlasWidget(df)")
    else:
        print("\n❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
