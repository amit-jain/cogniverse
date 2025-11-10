#!/usr/bin/env python3
"""
Generic Document - Universal document structure for all content types.

A single Document class that can represent any piece of content (video, image, text)
with flexible metadata and embedding storage.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ContentType(Enum):
    """Types of content the pipeline can process."""

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    DATAFRAME = "dataframe"
    DOCUMENT = "document"


class ProcessingStatus(Enum):
    """Processing status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Document:
    """Generic document that can represent any type of content.

    Completely generic design - no content-specific fields, just flexible structure.
    """

    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.DOCUMENT

    # Content information
    content_path: Optional[Path] = None
    content_id: Optional[str] = None
    title: Optional[str] = None

    # Generic content data
    text_content: Optional[str] = None
    description: Optional[str] = None

    # Embeddings - flexible storage for any embedding type
    embeddings: Dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

    # Flexible metadata for any additional fields
    metadata: Dict[str, Any] = field(default_factory=dict)

    # System metadata
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self):
        """Post-initialization processing."""
        if self.content_path:
            self.content_path = Path(self.content_path)

        # Auto-detect content type from content_path if not specified
        if self.content_type == ContentType.DOCUMENT and self.content_path:
            self._auto_detect_type()

    def _auto_detect_type(self):
        """Auto-detect content type from file extension."""
        if not self.content_path:
            return

        suffix = self.content_path.suffix.lower()
        if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            self.content_type = ContentType.VIDEO
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.content_type = ContentType.IMAGE
        elif suffix in [".wav", ".mp3", ".m4a"]:
            self.content_type = ContentType.AUDIO
        elif suffix in [".txt", ".md"]:
            self.content_type = ContentType.TEXT
        elif suffix in [".csv", ".parquet", ".json"]:
            self.content_type = ContentType.DATAFRAME

    def add_embedding(
        self,
        name: str,
        embedding: Union[List, Dict, Any],
        metadata: Optional[Dict] = None,
    ):
        """Add an embedding with optional metadata."""
        self.embeddings[name] = {
            "data": embedding,
            "metadata": metadata or {},
            "created_at": int(time.time()),
        }
        self.updated_at = int(time.time())

    def get_embedding(self, name: str) -> Optional[Any]:
        """Get embedding data by name."""
        return self.embeddings.get(name, {}).get("data")

    def get_embedding_metadata(self, name: str) -> Optional[Dict]:
        """Get embedding metadata by name."""
        return self.embeddings.get(name, {}).get("metadata")

    def set_processing_status(
        self, status: ProcessingStatus, error_message: Optional[str] = None
    ):
        """Update processing status."""
        self.status = status
        self.error_message = error_message
        self.updated_at = int(time.time())

    def mark_completed(self, processing_time: Optional[float] = None):
        """Mark document as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.processing_time = processing_time
        self.updated_at = int(time.time())

    def mark_failed(self, error_message: str):
        """Mark document as failed."""
        self.status = ProcessingStatus.FAILED
        self.error_message = error_message
        self.updated_at = int(time.time())

    def add_metadata(self, key: str, value: Any):
        """Add metadata field."""
        self.metadata[key] = value
        self.updated_at = int(time.time())

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def to_backend_document(self, schema_name: str) -> Dict[str, Any]:
        """Convert to backend document format (Vespa, ElasticSearch, etc.)."""
        # Base document structure
        doc = {
            "id": self.id,
            "content_id": self.content_id or "",
            "title": self.title or "",
            "text_content": self.text_content or "",
            "description": self.description or "",
            "content_type": self.content_type.value,
            "created_at": self.created_at,
        }

        # Add embeddings
        for emb_name, emb_data in self.embeddings.items():
            doc[emb_name] = emb_data["data"]

        # Add custom metadata (this is where content-specific fields go)
        doc.update(self.metadata)

        return doc

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content_type": self.content_type.value,
            "content_path": str(self.content_path) if self.content_path else None,
            "content_id": self.content_id,
            "title": self.title,
            "text_content": self.text_content,
            "description": self.description,
            "embeddings": self.embeddings,
            "status": self.status.value,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        doc = cls(
            id=data.get("id", str(uuid.uuid4())),
            content_type=ContentType(data.get("content_type", "document")),
            content_path=(
                Path(data["content_path"]) if data.get("content_path") else None
            ),
            content_id=data.get("content_id"),
            title=data.get("title"),
            text_content=data.get("text_content"),
            description=data.get("description"),
            embeddings=data.get("embeddings", {}),
            status=ProcessingStatus(data.get("status", "pending")),
            processing_time=data.get("processing_time"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
        )
        return doc

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Document(id={self.id[:8]}, type={self.content_type.value}, "
            f"content_id={self.content_id})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Document(id='{self.id}', type={self.content_type.value}, "
            f"status={self.status.value})"
        )


class SearchResult:
    """Represents a search result with document and score."""

    def __init__(
        self,
        document: Document,
        score: float,
        highlights: Optional[Dict[str, Any]] = None,
    ):
        self.document = document
        self.score = score
        self.highlights = highlights or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "document_id": self.document.id,
            "score": self.score,
            "metadata": self.document.metadata,
            "highlights": self.highlights,
        }

        # Add source_id if present in metadata
        if "source_id" in self.document.metadata:
            result["source_id"] = self.document.metadata["source_id"]

        # Add temporal info if present in metadata
        if (
            "start_time" in self.document.metadata
            and "end_time" in self.document.metadata
        ):
            result["temporal_info"] = {
                "start_time": self.document.metadata["start_time"],
                "end_time": self.document.metadata["end_time"],
                "duration": self.document.metadata["end_time"]
                - self.document.metadata["start_time"],
            }

        return result
