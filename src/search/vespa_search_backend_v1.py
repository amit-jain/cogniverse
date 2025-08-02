"""Vespa search backend implementation."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from vespa.application import Vespa

from .search import SearchBackend, SearchResult
from src.core import Document, MediaType, TemporalInfo, SegmentInfo

logger = logging.getLogger(__name__)


class VespaSearchBackend(SearchBackend):
    """Vespa implementation of search backend."""
    
    def __init__(self, vespa_url: str, vespa_port: int, schema_name: str, profile: str):
        """
        Initialize Vespa search backend.
        
        Args:
            vespa_url: Vespa URL
            vespa_port: Vespa port
            schema_name: Vespa schema to search
            profile: Video processing profile
        """
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.schema_name = schema_name
        self.profile = profile
        self.app = None
        self._connect()
        
    def _connect(self):
        """Connect to Vespa."""
        try:
            self.app = Vespa(url=self.vespa_url, port=self.vespa_port)
            logger.info(f"Connected to Vespa at {self.vespa_url}:{self.vespa_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Vespa: {e}")
            raise
    
    def _embeddings_to_vespa_format(self, embeddings: np.ndarray, profile: str) -> Dict[str, Any]:
        """Convert embeddings to Vespa query format."""
        # For binary embeddings, handle differently
        if "_binary" in profile:
            # For binary tensors, values should be int8
            if embeddings.ndim == 1:
                # 1D binary embeddings (global models)
                cells = [{"address": {"v": str(i)}, "value": int(val)} 
                        for i, val in enumerate(embeddings)]
                return {"cells": cells}
            else:
                # 2D binary embeddings (patch-based models like ColPali)
                cells = []
                for patch_idx in range(embeddings.shape[0]):
                    for v_idx in range(embeddings.shape[1]):
                        cells.append({
                            "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                            "value": int(embeddings[patch_idx, v_idx])
                        })
                return {"cells": cells}
        # For global profiles, embeddings are 1D
        elif "global" in profile:
            # Convert to tensor cells format for Vespa
            cells = [{"address": {"v": str(i)}, "value": float(val)} 
                    for i, val in enumerate(embeddings)]
            return {"cells": cells}
        else:
            # For patch-based models, embeddings are 2D
            cells = []
            for patch_idx in range(embeddings.shape[0]):
                for v_idx in range(embeddings.shape[1]):
                    cells.append({
                        "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                        "value": float(embeddings[patch_idx, v_idx])
                    })
            return {"cells": cells}
    
    def _generate_binary_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate binary embeddings from float embeddings."""
        # Binarize embeddings (>0 becomes 1, <=0 becomes 0)
        binary = np.where(embeddings > 0, 1, 0).astype(np.uint8)
        
        # Pack bits into bytes
        if len(binary.shape) == 1:
            # 1D embeddings (global)
            # Pad to multiple of 8 bits
            padding = (8 - len(binary) % 8) % 8
            if padding:
                binary = np.pad(binary, (0, padding), mode='constant')
            # Pack bits
            packed = np.packbits(binary).astype(np.int8)
        else:
            # 2D embeddings (patch-based)
            # Pack each patch separately
            packed = np.packbits(binary, axis=1).astype(np.int8)
        
        return packed
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert Vespa result to Document object."""
        fields = result.get("fields", {})
        
        # Extract document ID
        doc_id = result.get("id", "").split("::")[-1]
        
        # Determine media type based on schema
        if "frame" in self.schema_name:
            media_type = MediaType.VIDEO_FRAME
        elif "global" in self.schema_name:
            media_type = MediaType.VIDEO_SEGMENT
        else:
            media_type = MediaType.VIDEO_FRAME  # Default to frame
        
        # Build temporal info if available
        temporal_info = None
        if "start_time" in fields and "end_time" in fields:
            temporal_info = TemporalInfo(
                start_time=fields["start_time"],
                end_time=fields["end_time"]
            )
        
        # Build segment info if available
        segment_info = None
        if "segment_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["segment_id"],
                total_segments=fields.get("total_segments", 1)
            )
        elif "frame_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["frame_id"],
                total_segments=1
            )
        
        # Extract metadata
        metadata = {
            "video_title": fields.get("video_title"),
            "frame_description": fields.get("frame_description"),
            "segment_description": fields.get("segment_description"),
            "audio_transcript": fields.get("audio_transcript"),
            "creation_timestamp": fields.get("creation_timestamp")
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Add source_id to metadata
        metadata["source_id"] = fields.get("video_id", doc_id.split("_")[0])
        
        # Create Document using new structure
        return Document(
            doc_id=doc_id,
            media_type=media_type,
            temporal_info=temporal_info,
            segment_info=segment_info,
            metadata=metadata
        )
    
    def search(
        self,
        query_embeddings: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for documents matching the query."""
        if not self.app:
            raise RuntimeError("Not connected to Vespa")
        
        # Determine ranking profile
        if ranking_strategy:
            # Use provided ranking strategy
            ranking_profile = ranking_strategy
        else:
            # Use default based on profile type
            if "global" in self.profile:
                ranking_profile = "float_float"
            else:
                ranking_profile = "hybrid_binary_bm25"
        
        # Build query based on ranking strategy and profile
        # For pure visual search strategies, use nearestNeighbor
        pure_visual_strategies = ["float_float", "binary_binary", "float_binary", "phased"]
        
        if ranking_profile in pure_visual_strategies:
            # For global schemas, use nearestNeighbor; for patch-based, use regular ranking
            if "global" in self.profile:
                # Determine which field and query tensor to use for nearestNeighbor
                if ranking_profile == "float_float":
                    nn_field = "embedding"
                    query_tensor_name = "qt"
                elif ranking_profile == "binary_binary":
                    nn_field = "embedding_binary"
                    query_tensor_name = "qtb"
                elif ranking_profile in ["float_binary", "phased"]:
                    nn_field = "embedding_binary"
                    query_tensor_name = "qtb"
                else:
                    nn_field = "embedding"
                    query_tensor_name = "qt"
                
                # Global embeddings - use nearestNeighbor
                query_body = {
                    "yql": f"select * from {self.schema_name} where {{targetHits: {top_k}}}nearestNeighbor({nn_field}, {query_tensor_name})",
                    "ranking.profile": ranking_profile,
                    "hits": top_k,
                    "ranking": ranking_profile
                }
            else:
                # Patch-based embeddings - use regular ranking without nearestNeighbor
                query_body = {
                    "yql": f"select * from {self.schema_name} where true",
                    "ranking.profile": ranking_profile,
                    "hits": top_k,
                    "ranking": ranking_profile
                }
        else:
            # Hybrid or text search - use userInput
            query_body = {
                "yql": f"select * from {self.schema_name} where userInput(@userQuery)",
                "userQuery": query_text,
                "ranking.profile": ranking_profile,
                "hits": top_k
            }
        
        # Add tensor embeddings based on ranking profile
        needs_binary = ranking_profile in ["binary_binary", "float_binary", "phased", "hybrid_binary_bm25", 
                                          "hybrid_binary_bm25_no_description", "default"]
        needs_float = ranking_profile in ["float_float", "float_binary", "phased", "hybrid_float_bm25"]
        
        if needs_float:
            # For global embeddings, use list format (like original code)
            if "global" in self.profile:
                query_body["input.query(qt)"] = query_embeddings.tolist()
            else:
                # For patch-based models, use dict format
                query_tensor = self._embeddings_to_vespa_format(query_embeddings, self.profile)
                query_body["input.query(qt)"] = query_tensor
        
        if needs_binary:
            # Generate binary embeddings from float embeddings
            binary_embeddings = self._generate_binary_embeddings(query_embeddings)
            if "global" in self.profile:
                query_body["input.query(qtb)"] = binary_embeddings.tolist()
            else:
                binary_tensor = self._embeddings_to_vespa_format(binary_embeddings, self.profile + "_binary")
                query_body["input.query(qtb)"] = binary_tensor
        
        # Add filters if provided
        if filters:
            if "start_date" in filters or "end_date" in filters:
                # Add date filtering to YQL
                date_filters = []
                if "start_date" in filters:
                    date_filters.append(f"creation_timestamp >= {filters['start_date']}")
                if "end_date" in filters:
                    date_filters.append(f"creation_timestamp <= {filters['end_date']}")
                
                if date_filters:
                    yql = f"select * from {self.schema_name} where userInput(@userQuery) and {' and '.join(date_filters)}"
                    query_body["yql"] = yql
        
        # Execute search
        try:
            # Add schema to query body to avoid conflicts with other schemas
            query_body["model.restrict"] = self.schema_name  # Must be string, not list!
            response = self.app.query(body=query_body)
            
            # Convert results to SearchResult objects
            results = []
            for hit in response.hits:
                doc = self._result_to_document(hit)
                score = hit.get("relevance", 0.0)
                
                # Extract highlights if available
                highlights = {}
                if "summaryfeatures" in hit:
                    highlights = hit["summaryfeatures"]
                
                results.append(SearchResult(doc, score, highlights))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        if not self.app:
            raise RuntimeError("Not connected to Vespa")
        
        try:
            # Construct full document ID
            full_id = f"id:video:{self.schema_name}::{document_id}"
            
            # Get document
            response = self.app.get_data(data_id=full_id, schema=self.schema_name)
            
            if response and response.is_successful():
                # Create fake result dict for conversion
                result = {
                    "id": full_id,
                    "fields": response.get_json()
                }
                return self._result_to_document(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None