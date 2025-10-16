"""
Abstract base classes for backend implementations.

This module defines the interfaces that all backends must implement
to integrate with the Cogniverse system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from cogniverse_core.common.document import Document


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


class IngestionBackend(ABC):
    """Abstract base class for ingestion backends."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend with configuration.

        Args:
            config: Backend-specific configuration
        """
        pass

    @abstractmethod
    def ingest_documents(
        self, documents: List[Document], schema_name: str
    ) -> Dict[str, Any]:
        """
        Ingest a batch of documents into the backend.

        Args:
            documents: List of Document objects to ingest
            schema_name: Schema to ingest documents into

        Returns:
            Ingestion results including success count, errors, etc.
        """
        pass

    @abstractmethod
    def ingest_stream(self, documents: Iterator[Document]) -> Iterator[Dict[str, Any]]:
        """
        Stream documents for ingestion (for large datasets).

        Args:
            documents: Iterator of Document objects

        Yields:
            Ingestion results for each batch
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update an existing document.

        Args:
            document_id: ID of document to update
            document: Updated Document object

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the backend.

        Args:
            document_id: ID of document to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the backend schema.

        Returns:
            Schema information including fields, types, etc.
        """
        pass

    @abstractmethod
    def validate_schema(self, schema_name: str) -> bool:
        """
        Validate that a schema exists and is properly configured.

        Args:
            schema_name: Name of schema to validate

        Returns:
            True if schema is valid, False otherwise
        """
        pass


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the search backend with configuration.

        Args:
            config: Backend-specific configuration
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a search query.

        Args:
            query_embeddings: Optional query embeddings for vector search
            query_text: Optional text query for keyword search
            top_k: Number of results to return
            filters: Optional filters to apply
            ranking_strategy: Optional ranking strategy to use

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: ID of document to retrieve

        Returns:
            Document object or None if not found
        """
        pass

    @abstractmethod
    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of Document objects (None for not found)
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search backend statistics.

        Returns:
            Statistics including document count, index size, etc.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the search backend is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass


class Backend(IngestionBackend, SearchBackend):
    """
    Base class for backends that support both ingestion and search.

    Many backends (like Vespa, Elasticsearch) support both operations,
    so this provides a convenient base class.
    """

    def __init__(self, name: str):
        """
        Initialize backend.

        Args:
            name: Backend name for identification
        """
        self.name = name
        self._initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize both ingestion and search capabilities."""
        if self._initialized:
            return

        self._initialize_backend(config)
        self._initialized = True

    @abstractmethod
    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        """
        Backend-specific initialization.

        Args:
            config: Backend configuration
        """
        pass

    # ============================================================================
    # Schema Management Operations
    # ============================================================================

    @abstractmethod
    def deploy_schema(
        self, schema_name: str, tenant_id: Optional[str] = None, **kwargs
    ) -> bool:
        """
        Deploy or ensure schema exists for tenant.

        Args:
            schema_name: Base schema name to deploy
            tenant_id: Tenant identifier (for multi-tenant backends)
            **kwargs: Additional schema deployment options

        Returns:
            True if successful, False otherwise

        Note:
            For multi-tenant backends, this creates tenant-specific schema
            (e.g., video_colpali_{tenant_id}). For single-tenant backends,
            this ensures the base schema exists.
        """
        pass

    @abstractmethod
    def delete_schema(
        self, schema_name: str, tenant_id: Optional[str] = None
    ) -> List[str]:
        """
        Delete tenant schema(s).

        Args:
            schema_name: Base schema name to delete
            tenant_id: Tenant identifier

        Returns:
            List of deleted schema names

        Note:
            For multi-tenant backends, deletes all schemas for tenant.
            For single-tenant backends, this is typically a no-op.
        """
        pass

    @abstractmethod
    def schema_exists(
        self, schema_name: str, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Check if schema exists.

        Args:
            schema_name: Base schema name
            tenant_id: Tenant identifier

        Returns:
            True if schema exists, False otherwise
        """
        pass

    @abstractmethod
    def get_tenant_schema_name(
        self, tenant_id: str, base_schema_name: str
    ) -> str:
        """
        Get tenant-specific schema name.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            Tenant-specific schema name (e.g., "video_colpali_acme")

        Note:
            For non-tenant-aware backends, returns base_schema_name unchanged.
        """
        pass

    # ============================================================================
    # Metadata Document Operations (for tenant management, etc.)
    # ============================================================================

    @abstractmethod
    def create_metadata_document(
        self, schema: str, doc_id: str, fields: Dict[str, Any]
    ) -> bool:
        """
        Create or update metadata document.

        Args:
            schema: Schema name (e.g., "organization_metadata", "tenant_metadata")
            doc_id: Document ID
            fields: Document fields as dict

        Returns:
            True if successful, False otherwise

        Note:
            Used for storing organization/tenant metadata, not video content.
            Different from ingest_documents() which handles video documents.
        """
        pass

    @abstractmethod
    def get_metadata_document(
        self, schema: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata document by ID.

        Args:
            schema: Schema name
            doc_id: Document ID

        Returns:
            Document fields as dict, or None if not found
        """
        pass

    @abstractmethod
    def query_metadata_documents(
        self,
        schema: str,
        query: Optional[str] = None,
        yql: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query metadata documents.

        Args:
            schema: Schema name to query
            query: Text query (backend-specific syntax)
            yql: Direct query language (e.g., Vespa YQL)
            **kwargs: Additional query options (hits, filters, etc.)

        Returns:
            List of matching documents as dicts
        """
        pass

    @abstractmethod
    def delete_metadata_document(self, schema: str, doc_id: str) -> bool:
        """
        Delete metadata document.

        Args:
            schema: Schema name
            doc_id: Document ID

        Returns:
            True if successful, False otherwise
        """
        pass
