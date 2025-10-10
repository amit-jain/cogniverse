#!/usr/bin/env python3
"""
Document Builders - Handles creation of documents for different backends and schemas
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from cogniverse_runtime.ingestion.strategy import StrategyConfig


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


class DocumentBuilder:
    """Document builder for all schemas

    Since all schemas now use consistent field names and structure,
    we can use a single builder for all of them.
    """

    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.strategy_config = StrategyConfig()
        self.field_names = self._get_field_names()

    def build_document(
        self,
        metadata: DocumentMetadata,
        embeddings: dict[str, Any],
        additional_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a document for any schema"""

        doc_id = self.create_document_id(metadata)

        # Base fields - all schemas use these
        fields = {
            "video_id": metadata.video_id,
            "video_title": metadata.video_title,
            "creation_timestamp": metadata.creation_timestamp,
            "segment_id": metadata.segment_idx,  # All schemas now use segment_id
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
        }

        # Add embeddings using field names from strategy
        if "float_embeddings" in embeddings:
            fields[self.field_names["float_field"]] = embeddings["float_embeddings"]
        if "binary_embeddings" in embeddings:
            fields[self.field_names["binary_field"]] = embeddings["binary_embeddings"]

        # Add optional fields
        if additional_fields:
            # Text description (only ColPali has this)
            if "segment_description" in additional_fields:
                fields["segment_description"] = additional_fields["segment_description"]

            # Audio transcript (all schemas can have this)
            if "audio_transcript" in additional_fields:
                fields["audio_transcript"] = additional_fields["audio_transcript"]

            # Additional metadata
            if "total_segments" in additional_fields:
                fields["total_segments"] = additional_fields["total_segments"]
            if "segment_duration" in additional_fields:
                fields["segment_duration"] = additional_fields["segment_duration"]

        return {"id": doc_id, "fields": fields}

    def create_document_id(self, metadata: DocumentMetadata) -> str:
        """Create a unique document ID"""
        return f"{metadata.video_id}_segment_{metadata.segment_idx}"

    def _get_field_names(self) -> dict[str, str]:
        """Get field names from unified strategy"""
        try:
            # All schemas now use the same field names
            return {"float_field": "embedding", "binary_field": "embedding_binary"}
        except Exception:
            # Fallback to default field names
            return {"float_field": "embedding", "binary_field": "embedding_binary"}


class DocumentBuilderFactory:
    """Factory for creating document builders

    This factory simply returns the DocumentBuilder for all schemas.
    """

    @staticmethod
    def create_builder(schema_name: str) -> DocumentBuilder:
        """Create document builder for any schema"""
        return DocumentBuilder(schema_name)
