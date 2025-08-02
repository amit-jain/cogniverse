#!/usr/bin/env python3
"""
Document Builders - Handles creation of documents for different backends and schemas
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass


@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    video_id: str
    video_title: str
    segment_idx: int
    start_time: float
    end_time: float
    creation_timestamp: int = None
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = int(time.time())


class BaseDocumentBuilder(ABC):
    """Base class for document builders"""
    
    def __init__(self, schema_name: str):
        self.schema_name = schema_name
    
    @abstractmethod
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a document for the backend"""
        pass
    
    def create_document_id(self, metadata: DocumentMetadata) -> str:
        """Create a unique document ID"""
        return f"{metadata.video_id}_segment_{metadata.segment_idx}"
    
    def create_put_id(self, doc_id: str) -> str:
        """Create the PUT ID for Vespa"""
        return f"id:video:{self.schema_name}::{doc_id}"


class VideoFrameDocumentBuilder(BaseDocumentBuilder):
    """Document builder for video_frame schema"""
    
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build document for video_frame schema"""
        
        doc_id = self.create_document_id(metadata)
        
        # Base fields
        fields = {
            "video_id": metadata.video_id,
            "video_title": metadata.video_title,
            "creation_timestamp": metadata.creation_timestamp,
            "frame_id": metadata.segment_idx,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "colpali_embedding": embeddings.get("float_embeddings", {}),
            "colpali_binary": embeddings.get("binary_embeddings", {})
        }
        
        # Add optional fields
        if additional_fields:
            if "frame_description" in additional_fields:
                fields["frame_description"] = additional_fields["frame_description"]
            if "audio_transcript" in additional_fields:
                fields["audio_transcript"] = additional_fields["audio_transcript"]
        
        return {
            "put": self.create_put_id(doc_id),
            "fields": fields
        }


class ColQwenDocumentBuilder(BaseDocumentBuilder):
    """Document builder for video_colqwen schema"""
    
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build document for video_colqwen schema"""
        
        doc_id = self.create_document_id(metadata)
        
        # ColQwen schema uses different field names
        fields = {
            "video_id": metadata.video_id,
            "video_title": metadata.video_title,
            "creation_timestamp": metadata.creation_timestamp,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "embedding": embeddings.get("float_embeddings", {}),
            "embedding_binary": embeddings.get("binary_embeddings", {})
        }
        
        # Add segment metadata
        if additional_fields:
            if "segment_id" in additional_fields:
                fields["segment_id"] = additional_fields["segment_id"]
            if "total_segments" in additional_fields:
                fields["total_segments"] = additional_fields["total_segments"]
            if "segment_duration" in additional_fields:
                fields["segment_duration"] = additional_fields["segment_duration"]
        
        return {
            "put": self.create_put_id(doc_id),
            "fields": fields
        }


class VideoPrismDocumentBuilder(BaseDocumentBuilder):
    """Document builder for VideoPrism schemas"""
    
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build document for VideoPrism schema"""
        
        doc_id = self.create_document_id(metadata)
        
        # VideoPrism schema
        fields = {
            "video_id": metadata.video_id,
            "video_title": metadata.video_title,
            "creation_timestamp": metadata.creation_timestamp,
            "frame_id": metadata.segment_idx,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "embedding": embeddings.get("float_embeddings", {}),
            "embedding_binary": embeddings.get("binary_embeddings", {})
        }
        
        return {
            "put": self.create_put_id(doc_id),
            "fields": fields
        }


class DocumentBuilderFactory:
    """Factory for creating document builders"""
    
    @staticmethod
    def create_builder(schema_name: str) -> BaseDocumentBuilder:
        """Create appropriate document builder based on schema name"""
        
        schema_lower = schema_name.lower()
        
        if "colqwen" in schema_lower:
            return ColQwenDocumentBuilder(schema_name)
        elif "videoprism" in schema_lower:
            return VideoPrismDocumentBuilder(schema_name)
        else:
            # Default to video_frame builder
            return VideoFrameDocumentBuilder(schema_name)