#!/usr/bin/env python3
"""
Single Vector Document Builder - Generic builder for all single-vector approaches

This builder handles document creation for any single-vector embedding strategy:
- Chunked documents (multiple segments in one doc)
- Window documents (one doc per segment)
- Global documents (entire video as one doc)

Model-agnostic and schema-flexible.
"""

from typing import Dict, Any, List, Optional, Literal
import numpy as np
import logging
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.pipeline_steps.embedding_generator.document_builders import BaseDocumentBuilder, DocumentMetadata

logger = logging.getLogger(__name__)


class SingleVectorDocumentBuilder(BaseDocumentBuilder):
    """
    Generic document builder for all single-vector embedding approaches.
    Configurable for different storage strategies and schemas.
    """
    
    def __init__(
        self,
        schema_name: str,
        storage_mode: Literal["single_doc", "multi_doc"] = "single_doc",
        embedding_field_name: str = "embeddings",
        metadata_field_name: str = "segment_metadata"
    ):
        """
        Initialize builder with configuration.
        
        Args:
            schema_name: Vespa schema name
            storage_mode: How to store segments
                - "single_doc": All segments in one document (like TwelveLabs)
                - "multi_doc": Each segment as separate document
            embedding_field_name: Name of embedding field in schema
            metadata_field_name: Name of metadata field in schema
        """
        super().__init__(schema_name)
        self.storage_mode = storage_mode
        self.embedding_field_name = embedding_field_name
        self.metadata_field_name = metadata_field_name
        
    def build_documents(
        self,
        video_data: Dict[str, Any],
        embeddings: List[np.ndarray],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build documents from processed video data and embeddings.
        
        Args:
            video_data: Output from SingleVectorVideoProcessor
            embeddings: List of embeddings for each segment
            additional_metadata: Extra metadata (title, keywords, etc.)
            
        Returns:
            List of documents ready for Vespa
        """
        if self.storage_mode == "single_doc":
            return self._build_single_document(video_data, embeddings, additional_metadata)
        else:
            return self._build_multi_documents(video_data, embeddings, additional_metadata)
    
    def _build_single_document(
        self,
        video_data: Dict[str, Any],
        embeddings: List[np.ndarray],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build a single document containing all segments"""
        segments = video_data["segments"]
        metadata = video_data["metadata"]
        
        # Build embeddings tensor
        embeddings_dict = {}
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, np.ndarray):
                embeddings_dict[str(i)] = embedding.tolist()
            else:
                embeddings_dict[str(i)] = embedding
        
        # Build segment arrays
        start_times = []
        end_times = []
        segment_transcripts = []
        
        for i, segment in enumerate(segments):
            start_times.append(segment.start_time)
            end_times.append(segment.end_time)
            segment_transcripts.append(segment.transcript_text)
        
        # Create document
        doc_fields = {
            "video_id": metadata["video_id"],
            "duration": metadata["duration"],
            "creation_timestamp": int(time.time()),
            self.embedding_field_name: embeddings_dict,
            "start_times": start_times,
            "end_times": end_times,
            "segment_transcripts": segment_transcripts,
            "num_segments": len(segments)
        }
        
        # Add additional metadata
        if additional_metadata:
            doc_fields.update({
                "title": additional_metadata.get("title", metadata["video_id"]),
                "keywords": additional_metadata.get("keywords", ""),
                "summary": additional_metadata.get("summary", ""),
                "url": additional_metadata.get("url", "")
            })
        else:
            doc_fields.update({
                "title": metadata["video_id"],
                "keywords": "",
                "summary": "",
                "url": ""
            })
        
        # Create Vespa document
        doc = {
            "put": f"id:{self.schema_name}:{self.schema_name}::{metadata['video_id']}",
            "fields": doc_fields
        }
        
        return [doc]
    
    def _build_multi_documents(
        self,
        video_data: Dict[str, Any],
        embeddings: List[np.ndarray],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build multiple documents, one per segment"""
        segments = video_data["segments"]
        metadata = video_data["metadata"]
        documents = []
        
        for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
            # Create document for this segment
            doc_fields = {
                "video_id": metadata["video_id"],
                "segment_id": i,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "creation_timestamp": int(time.time()),
                "transcript": segment.transcript_text,
                "processing_strategy": metadata["strategy"]
            }
            
            # Add embedding
            if isinstance(embedding, np.ndarray):
                doc_fields[self.embedding_field_name] = embedding.tolist()
            else:
                doc_fields[self.embedding_field_name] = embedding
            
            # Add video-level metadata
            if additional_metadata:
                doc_fields.update({
                    "video_title": additional_metadata.get("title", metadata["video_id"]),
                    "video_keywords": additional_metadata.get("keywords", ""),
                    "video_summary": additional_metadata.get("summary", "")
                })
            
            # Create Vespa document
            doc_id = f"{metadata['video_id']}_segment_{i}"
            doc = {
                "put": f"id:{self.schema_name}:{self.schema_name}::{doc_id}",
                "fields": doc_fields
            }
            
            documents.append(doc)
        
        return documents
    
    def extract_segment_results(
        self,
        search_result: Dict[str, Any],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract individual segment matches from search results.
        
        For single_doc mode: Identifies which segments matched
        For multi_doc mode: Already segment-level results
        """
        if self.storage_mode == "multi_doc":
            # Already segment-level
            return [{
                "video_id": search_result.get("video_id"),
                "segment_id": search_result.get("segment_id"),
                "start_time": search_result.get("start_time"),
                "end_time": search_result.get("end_time"),
                "transcript": search_result.get("transcript"),
                "score": search_result.get("relevance", 0)
            }]
        
        # For single_doc, extract matching segments
        segment_results = []
        embeddings = search_result.get(self.embedding_field_name, {})
        start_times = search_result.get("start_times", [])
        end_times = search_result.get("end_times", [])
        transcripts = search_result.get("segment_transcripts", [])
        
        # If we have query embedding, compute similarities
        if query_embedding is not None and embeddings:
            for seg_id, seg_embedding in embeddings.items():
                seg_idx = int(seg_id)
                if seg_idx < len(start_times) and seg_idx < len(end_times):
                    # Compute similarity (simplified - Vespa would do this)
                    similarity = self._compute_similarity(query_embedding, seg_embedding)
                    
                    segment_results.append({
                        "video_id": search_result.get("video_id"),
                        "segment_id": seg_idx,
                        "start_time": start_times[seg_idx],
                        "end_time": end_times[seg_idx],
                        "transcript": transcripts[seg_idx] if seg_idx < len(transcripts) else "",
                        "score": similarity
                    })
        else:
            # No query embedding - return all segments
            for seg_id in embeddings.keys():
                seg_idx = int(seg_id)
                if seg_idx < len(start_times) and seg_idx < len(end_times):
                    segment_results.append({
                        "video_id": search_result.get("video_id"),
                        "segment_id": seg_idx,
                        "start_time": start_times[seg_idx],
                        "end_time": end_times[seg_idx],
                        "transcript": transcripts[seg_idx] if seg_idx < len(transcripts) else "",
                        "score": 0
                    })
        
        # Sort by score if available
        segment_results.sort(key=lambda x: x["score"], reverse=True)
        
        return segment_results
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: List[float]) -> float:
        """Simple cosine similarity computation"""
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return float(dot_product / (norm1 * norm2))
    
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Legacy method for compatibility.
        Use build_documents() for new code.
        """
        raise NotImplementedError(
            "Use build_documents() method for SingleVectorDocumentBuilder"
        )