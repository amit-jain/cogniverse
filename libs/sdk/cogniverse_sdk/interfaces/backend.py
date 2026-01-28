"""
Abstract base classes for backend implementations.

This module defines the interfaces that all backends must implement
to integrate with the Cogniverse system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from cogniverse_sdk.document import Document


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

    @abstractmethod
    def get_embedding_requirements(self, schema_name: str) -> Dict[str, Any]:
        """
        Get embedding requirements for a specific schema.

        This method allows the backend to specify what types of embeddings
        it needs for ingestion, based on its internal schema configuration
        (e.g., rank-profiles in Vespa, index config in other backends).

        Args:
            schema_name: Name of schema to get requirements for

        Returns:
            Dict containing:
                - needs_float: bool - whether float embeddings are needed
                - needs_binary: bool - whether binary embeddings are needed
                - float_field: str - name of float embedding field
                - binary_field: str - name of binary embedding field

        Note:
            This is backend-specific metadata that should NOT be exposed
            to application code. Only used internally by ingestion pipeline.
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

        This method is called once when the backend is first created. Implementations
        should perform backend-specific setup including connection initialization,
        client creation, and optionally deploying metadata/system schemas.

        Args:
            config: Backend configuration

        Metadata Schema Deployment:
            Some backends may choose to deploy metadata schemas (e.g.,
            organization_metadata, tenant_metadata) automatically during
            initialization. This is backend-specific.

            CRITICAL: For backends that require ALL schemas in a single deployment
            (e.g., Vespa's ApplicationPackage model), metadata schema deployment MUST:

            1. Query SchemaRegistry for all existing tenant schemas
            2. Include existing schemas in the deployment alongside metadata schemas
            3. Deploy all schemas together to prevent schema-removal errors

            Example (Vespa pattern):
                def _initialize_backend(self, config):
                    # ... connection setup ...

                    # Deploy metadata schemas (schema-aware)
                    if self.schema_registry:
                        self.schema_manager._schema_registry = self.schema_registry

                    # This method queries SchemaRegistry internally:
                    # - Gets all existing tenant schemas
                    # - Deploys metadata + existing schemas together
                    self.schema_manager.upload_metadata_schemas(app_name)

            Backends with incremental schema deployment (e.g., Elasticsearch, MongoDB)
            do NOT need this pattern - they can deploy metadata schemas independently.

        Schema Coordination:
            The SchemaRegistry is injected before _initialize_backend() is called,
            making it available via self.schema_registry. Use it to:
            - Query existing schemas before deployment
            - Ensure all backend instances coordinate schema state
            - Prevent schema-removal errors in multi-backend scenarios

        Multi-Backend Coordination:
            When multiple backend instances are created (e.g., ingestion backend,
            then search backend), each initialization triggers metadata deployment.
            Without querying SchemaRegistry, the second backend would only deploy
            metadata schemas, causing the backend to believe tenant schemas were
            intentionally removed.

            Example scenario:
                1. Ingestion backend deploys tenant schema "video_colpali_tenant_a"
                2. Search backend created → _initialize_backend() called
                3. Metadata deployment WITHOUT SchemaRegistry query:
                   - Deploys ONLY [org_metadata, tenant_metadata]
                   - Backend sees video_colpali_tenant_a missing
                   - Interprets as schema removal → ERROR
                4. Metadata deployment WITH SchemaRegistry query:
                   - Queries registry:
                     [org_metadata, tenant_metadata, video_colpali_tenant_a]
                   - Deploys all 3 schemas together
                   - No schema removal detected → SUCCESS

        Implementation Requirements:
            - Initialize backend connections and clients
            - Set up internal state and caching
            - Optionally deploy metadata/system schemas (backend-specific)
            - For all-schemas-at-once backends: query SchemaRegistry before
              metadata deployment
            - Ensure idempotency: safe to call initialize() multiple times
        """
        pass

    # ============================================================================
    # Schema Management Operations
    # ============================================================================

    @abstractmethod
    def deploy_schemas(self, schema_definitions: List[Dict[str, Any]]) -> bool:
        """
        Deploy multiple schemas together (required for multi-tenant backends).

        This is the low-level deployment interface called by SchemaRegistry.
        For backends like Vespa that require ALL schemas to be deployed together,
        this method receives the complete list of schemas to deploy.

        Args:
            schema_definitions: List of schema definition dicts, each containing:
                - name: Full schema name (e.g., "video_colpali_acme")
                - definition: Schema structure (fields, rank profiles, etc.)
                - tenant_id: Tenant identifier
                - base_schema_name: Original base schema name

        Returns:
            True if successful, False otherwise

        Note:
            This is called by SchemaRegistry, which collects all existing schemas
            and adds the new one before calling this method. This ensures that
            backends requiring full schema redeployment (like Vespa) work correctly.

        Example:
            schemas = [
                {
                    "name": "video_colpali_tenant_a",
                    "definition": {...},
                    "tenant_id": "tenant_a",
                    "base_schema_name": "video_colpali"
                },
                {
                    "name": "video_colpali_tenant_b",
                    "definition": {...},
                    "tenant_id": "tenant_b",
                    "base_schema_name": "video_colpali"
                }
            ]
            backend.deploy_schemas(schemas)
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
    def schema_exists(self, schema_name: str, tenant_id: Optional[str] = None) -> bool:
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
    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
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
        **kwargs,
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
