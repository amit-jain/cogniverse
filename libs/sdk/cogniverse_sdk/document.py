#!/usr/bin/env python3
"""
Generic Document - Universal document structure for all content types.

A single Document class that can represent any piece of content (video, image, text)
with flexible metadata and embedding storage. ``DocumentFieldMapping`` +
``Document.to_schema_fields`` translate the generic fields into one concrete
schema's field names for feeding — schemas declare their mapping, the
serializer stays pure.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _as_epoch_seconds(value: Any, field_name: str) -> int:
    """Coerce a stored timestamp to epoch seconds, failing with context.

    Accepts int/float seconds, digit strings, and millisecond epochs
    (converted); anything else raises instead of storing a mistyped value
    the backend's long field rejects much later.
    """
    if value is None:
        return int(time.time())
    # numpy scalars first: np.float64 IS a float subclass (handled below) but
    # np.int64 is NOT an int subclass and np.bool_ is NOT a bool subclass, so a
    # created_at read from a pandas/parquet row (naturally np.int64) would fall
    # through to the raise. Coerce to the Python scalar so the checks apply
    # uniformly (and np.bool_ still gets rejected as a bool below).
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"Document.from_dict: {field_name} must be a timestamp")
    if isinstance(value, str):
        if not value.strip().isdigit():
            raise TypeError(
                f"Document.from_dict: {field_name} must be an integer "
                f"timestamp, got {value!r}"
            )
        value = int(value.strip())
    if isinstance(value, float):
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(
            f"Document.from_dict: {field_name} must be an integer timestamp, "
            f"got {type(value).__name__}"
        )
    if value >= 1_000_000_000_000:  # milliseconds epoch
        value //= 1000
    return value


@dataclass
class DocumentFieldMapping:
    """Write-side translation of generic Document fields to one schema's
    field names.

    Declared per schema (a ``document_mapping`` block in the schema JSON).
    A generic field left unmapped (None) is OMITTED from the feed — schemas
    only receive fields they declare, which is what makes a generic
    Document feedable at all.
    """

    id: Optional[str] = None
    title: Optional[str] = None
    text_content: Optional[str] = None
    description: Optional[str] = None
    content_type: Optional[str] = None
    content_id: Optional[str] = None
    content_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # "epoch" (int seconds), "epoch_ms" (int milliseconds), or "iso" (UTC string)
    created_at_format: str = "epoch"
    embeddings: Dict[str, str] = field(default_factory=dict)
    # Rename specific metadata keys to schema field names on the way out
    # (e.g. {"segment_index": "segment_id"}) — for values a Document carries in
    # metadata rather than a core field. The renamed value wins over a raw
    # metadata passthrough of the same source key.
    metadata_fields: Dict[str, str] = field(default_factory=dict)
    include_metadata: bool = True

    _FORMATS = ("epoch", "epoch_ms", "iso")

    def __post_init__(self):
        if self.created_at_format not in self._FORMATS:
            raise ValueError(
                f"DocumentFieldMapping: created_at_format must be one of "
                f"{self._FORMATS}, got {self.created_at_format!r}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentFieldMapping":
        if not isinstance(data, dict):
            raise TypeError(
                f"document_mapping must be a dict, got {type(data).__name__}"
            )
        known = {
            "id",
            "title",
            "text_content",
            "description",
            "content_type",
            "content_id",
            "content_path",
            "created_at",
            "updated_at",
            "created_at_format",
            "embeddings",
            "metadata_fields",
            "include_metadata",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"document_mapping has unknown keys: {sorted(unknown)}")
        metadata_fields = data.get("metadata_fields")
        if metadata_fields is not None:
            if not isinstance(metadata_fields, dict):
                raise ValueError(
                    "document_mapping.metadata_fields must be a dict of "
                    f"{{metadata_key: schema_field}}, got "
                    f"{type(metadata_fields).__name__}"
                )
            for src, target in metadata_fields.items():
                if not isinstance(src, str) or not isinstance(target, str):
                    raise ValueError(
                        "document_mapping.metadata_fields must map str to str, "
                        f"got {src!r} -> {target!r}"
                    )
        # embeddings must be a dict of {embedding_name: schema_field}; a
        # list-typed block would otherwise pass load and crash later at
        # ``mapping.embeddings.items()`` during the feed.
        embeddings = data.get("embeddings")
        if embeddings is not None:
            if not isinstance(embeddings, dict):
                raise ValueError(
                    "document_mapping.embeddings must be a dict of "
                    f"{{embedding_name: schema_field}}, got "
                    f"{type(embeddings).__name__}"
                )
            for name, target in embeddings.items():
                if not isinstance(name, str) or not isinstance(target, str):
                    raise ValueError(
                        "document_mapping.embeddings must map str to str, got "
                        f"{name!r} -> {target!r}"
                    )
        return cls(**data)


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

    def to_schema_fields(self, mapping: DocumentFieldMapping) -> Dict[str, Any]:
        """Serialize into one schema's field names per *mapping*.

        Metadata keys pass through verbatim (they are schema-specific by
        contract); mapped core fields overwrite a colliding metadata key so
        the Document's own values win deterministically. Embedding values
        unwrap both the wrapped ``add_embedding`` shape and raw vectors.
        Fields whose generic value is None are omitted, as are generic
        fields the mapping does not name.
        """
        fields_out: Dict[str, Any] = {}
        if mapping.include_metadata:
            fields_out.update(self.metadata)

        # Explicit metadata-key -> schema-field renames (e.g. segment_index ->
        # segment_id). These win over a raw passthrough of the same source key
        # and let a schema whose values live in metadata be fed without a blanket
        # passthrough (set include_metadata false to feed ONLY the declared set).
        for src_key, schema_field in mapping.metadata_fields.items():
            if src_key in self.metadata:
                fields_out[schema_field] = self.metadata[src_key]

        core: Dict[str, Any] = {}
        if mapping.id:
            core[mapping.id] = self.id
        if mapping.title and self.title is not None:
            core[mapping.title] = self.title
        if mapping.text_content and self.text_content is not None:
            core[mapping.text_content] = self.text_content
        if mapping.description and self.description is not None:
            core[mapping.description] = self.description
        if mapping.content_type:
            # content_type is normally a ContentType enum, but a caller may set
            # it to a raw string ("video"); coerce through the enum so a bad
            # value raises a clear ValueError instead of an opaque AttributeError.
            ct = self.content_type
            if not isinstance(ct, ContentType):
                ct = ContentType(ct)
            core[mapping.content_type] = ct.value
        if mapping.content_id and self.content_id is not None:
            core[mapping.content_id] = self.content_id
        if mapping.content_path and self.content_path is not None:
            core[mapping.content_path] = str(self.content_path)

        def _stamp(value: int) -> Any:
            # created_at is an epoch in SECONDS; land it as the schema field
            # wants: an ISO string, milliseconds, or seconds — always an int for
            # the numeric forms (a float must not reach a Vespa long field).
            if mapping.created_at_format == "iso":
                return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
            if mapping.created_at_format == "epoch_ms":
                return int(value) * 1000
            return int(value)

        if mapping.created_at:
            core[mapping.created_at] = _stamp(self.created_at)
        if mapping.updated_at:
            core[mapping.updated_at] = _stamp(self.updated_at)
        fields_out.update(core)

        for emb_name, schema_field in mapping.embeddings.items():
            if emb_name not in self.embeddings:
                continue
            emb_data = self.embeddings[emb_name]
            if isinstance(emb_data, dict) and "data" in emb_data:
                fields_out[schema_field] = emb_data["data"]
            else:
                fields_out[schema_field] = emb_data

        return fields_out

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
        """Create Document from dictionary.

        A corrupt payload raises with the offending field named — a
        silently mistyped document (string timestamp, scalar embeddings
        map) detonates far from here otherwise, inside whatever backend
        consumes it.
        """
        embeddings = data.get("embeddings") or {}
        if not isinstance(embeddings, dict):
            raise TypeError(
                f"Document.from_dict: embeddings must be a dict, "
                f"got {type(embeddings).__name__}"
            )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Document.from_dict: metadata must be a dict, "
                f"got {type(metadata).__name__}"
            )
        try:
            content_type = ContentType(data.get("content_type", "document"))
        except ValueError:
            raise ValueError(
                f"Document.from_dict: unknown content_type {data.get('content_type')!r}"
            ) from None
        try:
            status = ProcessingStatus(data.get("status", "pending"))
        except ValueError:
            raise ValueError(
                f"Document.from_dict: unknown status {data.get('status')!r}"
            ) from None

        doc = cls(
            id=data.get("id", str(uuid.uuid4())),
            content_type=content_type,
            content_path=(
                Path(data["content_path"]) if data.get("content_path") else None
            ),
            content_id=data.get("content_id"),
            title=data.get("title"),
            text_content=data.get("text_content"),
            description=data.get("description"),
            embeddings=embeddings,
            status=status,
            processing_time=data.get("processing_time"),
            error_message=data.get("error_message"),
            metadata=metadata,
            created_at=_as_epoch_seconds(data.get("created_at"), "created_at"),
            updated_at=_as_epoch_seconds(data.get("updated_at"), "updated_at"),
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

        # Add temporal info if present in metadata; duration only when both
        # bounds are numeric — string timecodes previously crashed the
        # whole serialization on the subtraction.
        start = self.document.metadata.get("start_time")
        end = self.document.metadata.get("end_time")
        if start is not None and end is not None:
            temporal: Dict[str, Any] = {"start_time": start, "end_time": end}
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                temporal["duration"] = end - start
            result["temporal_info"] = temporal

        return result
