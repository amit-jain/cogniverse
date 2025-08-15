#!/usr/bin/env python3
"""
Document Builders - Handles creation of documents for different backends and schemas
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.strategy import StrategyConfig


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
        self.strategy_config = StrategyConfig()
        self.field_names = self._get_field_names()
    
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
    
    def _get_field_names(self) -> Dict[str, str]:
        """Get field names from unified strategy"""
        try:
            # Find a profile that uses this schema
            profiles = self.strategy_config.config.get("video_processing_profiles", {})
            for profile_name, profile in profiles.items():
                if profile.get("vespa_schema") == self.schema_name:
                    return self.strategy_config.get_embedding_fields(profile_name)
            
            # Fallback to defaults
            return {
                'float_field': 'embedding',
                'binary_field': 'embedding_binary'
            }
        except Exception:
            # Fallback to default field names
            return {
                'float_field': 'embedding',
                'binary_field': 'embedding_binary'
            }


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
            self.field_names['float_field']: embeddings.get("float_embeddings", {}),
            self.field_names['binary_field']: embeddings.get("binary_embeddings", {})
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
    """Document builder for old video_colqwen schema (deprecated - use colqwen_chunks instead)"""
    
    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: Dict[str, Any],
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build document for old video_colqwen schema (deprecated)"""
        
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
        
        # Check for single vector schemas
        if "sv_chunk" in schema_lower:
            from .single_vector_document_builder import SingleVectorDocumentBuilder
            # Determine storage mode based on schema
            storage_mode = "single_doc" if "chunks" in schema_lower or "6s" in schema_lower else "multi_doc"
            return SingleVectorDocumentBuilder(
                schema_name=schema_name,
                storage_mode=storage_mode
            )
        elif "sv_chunk" in schema_lower and "colqwen" in self.schema_name:
            from .colqwen_chunks_builder import ColQwenChunksBuilder
            return ColQwenChunksBuilder()
        elif "colqwen" in schema_lower and "sv_chunk" not in schema_lower:
            return ColQwenDocumentBuilder(schema_name)
        elif "mv_chunk" in schema_lower or "sv_global" in schema_lower:
            return VideoPrismDocumentBuilder(schema_name)
        else:
            # Default to video_frame builder
            return VideoFrameDocumentBuilder(schema_name)