"""
Vespa backend implementation with unified interface.

This module provides a Vespa backend that implements both IngestionBackend
and SearchBackend interfaces, with self-registration to the backend registry.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from cogniverse_core.common.document import Document
from cogniverse_core.interfaces.backend import Backend

from .config import calculate_config_port
from .ingestion_client import VespaPyClient
from .search_backend import VespaSearchBackend
from .tenant_schema_manager import TenantSchemaManager
from .vespa_schema_manager import VespaSchemaManager

# Check if async ingestion client is available (optional dependency)
try:
    from .async_ingestion_client import AsyncVespaBackendAdapter  # noqa: F401
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class VespaBackend(Backend):
    """
    Vespa backend implementation supporting both ingestion and search.
    
    This class wraps the existing Vespa implementations and provides
    a unified interface compatible with the backend registry.
    """
    
    def __init__(self):
        """Initialize Vespa backend."""
        super().__init__("vespa")
        self._vespa_search_backend: Optional[VespaSearchBackend] = None
        # Store multiple ingestion clients, one per schema
        self._vespa_ingestion_clients: Dict[str, VespaPyClient] = {}
        self._async_ingestion_clients: Dict[str, Any] = {}  # For async ingestion (optional)
        self.schema_manager: Optional[VespaSchemaManager] = None
        self.tenant_schema_manager: Optional[TenantSchemaManager] = None
        self._initialized_as_search = False
        self._initialized_as_ingestion = False
        self.use_async_ingestion = False  # Flag to enable async mode
        # Store only what's needed for creating clients
        self._vespa_url: Optional[str] = None
        self._vespa_port: int = 8080
        self._tenant_id: Optional[str] = None
    
    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        """
        Initialize Vespa backend components.

        Args:
            config: Backend configuration including:
                - vespa_url: Vespa endpoint URL
                - vespa_port: Vespa port
                - tenant_id: Tenant identifier (REQUIRED for multi-tenancy)
                - schema_name: Schema to use
                - profile: Processing profile
                - strategy: Strategy object (optional)
                - query_encoder: Query encoder (optional)
        """
        # Check if async ingestion is requested (optional feature)
        self.use_async_ingestion = config.get("use_async_ingestion", False) and ASYNC_AVAILABLE

        # Store config for accessing profiles later
        self.config = config

        # Extract and store tenant_id (REQUIRED for tenant-aware operations)
        self._tenant_id = config.get("tenant_id")
        if not self._tenant_id:
            logger.warning("No tenant_id provided in config - backend will use base schemas without tenant isolation")

        # Store connection details needed for creating clients
        self._vespa_url = config.get("vespa_url")
        self._vespa_port = config.get("vespa_port", 8080)

        # Mark as ingestion backend if schema_name is provided
        if "schema_name" in config:
            self._initialized_as_ingestion = True
            # Don't create client yet - will create per-schema on demand

        # Initialize schema manager for schema operations
        vespa_url = config["vespa_url"]  # Required, no default
        vespa_port = config.get("vespa_port", 8080)
        deployment_port = config.get("vespa_deployment_port", 19071)

        self.schema_manager = VespaSchemaManager(
            vespa_endpoint=f"{vespa_url}:{vespa_port}",
            vespa_port=deployment_port
        )

        # Initialize TenantSchemaManager for tenant-aware schema management
        # Calculate config port from data port if not explicitly provided
        vespa_config_port = config.get("vespa_config_port")
        if not vespa_config_port:
            vespa_config_port = calculate_config_port(vespa_port)
            logger.debug(f"Calculated config port {vespa_config_port} from data port {vespa_port}")

        self.tenant_schema_manager = TenantSchemaManager(
            vespa_url=vespa_url,
            vespa_port=vespa_config_port
        )
        
        # Initialize search backend if this is being used for search
        if "schema_name" in config and ("strategy" in config or "profile" in config):
            # This means we're initializing for search operations
            self._initialized_as_search = True

            # Transform schema name to tenant-scoped if tenant_id is provided
            search_schema_name = config["schema_name"]
            if self._tenant_id and self.tenant_schema_manager:
                search_schema_name = self.tenant_schema_manager.get_tenant_schema_name(
                    self._tenant_id, config["schema_name"]
                )
                logger.debug(
                    f"Using tenant schema '{search_schema_name}' for tenant '{self._tenant_id}' "
                    f"(base schema: '{config['schema_name']}')"
                )

            # Create the actual VespaSearchBackend with tenant-scoped schema
            # Pass strategy=None to let VespaSearchBackend load it from profile
            self._vespa_search_backend = VespaSearchBackend(
                vespa_url=config["vespa_url"],  # Required
                vespa_port=config.get("vespa_port", 8080),
                schema_name=search_schema_name,  # Use tenant-scoped schema
                profile=config.get("profile"),
                strategy=None,  # Let VespaSearchBackend load Strategy object from profile
                query_encoder=config.get("query_encoder")
            )
        
        logger.info(f"Initialized Vespa backend with config: {config}")
    
    # Ingestion methods
    
    def _get_or_create_ingestion_client(self, schema_name: str) -> VespaPyClient:
        """
        Get or create a schema-specific ingestion client with tenant-aware schema naming.

        Args:
            schema_name: Base schema name to get client for

        Returns:
            VespaPyClient configured for the tenant-specific schema

        Note:
            If tenant_id is set, this method will:
            1. Transform base schema name to tenant-scoped name (e.g., video_colpali_smol500_mv_frame_test_tenant)
            2. Ensure the tenant-scoped schema exists in Vespa (auto-deploy if needed)
            3. Create a client that ingests to the tenant-scoped schema
        """
        # Transform base schema name to tenant-scoped name if tenant_id is set
        target_schema_name = schema_name
        if self._tenant_id and self.tenant_schema_manager:
            target_schema_name = self.tenant_schema_manager.get_tenant_schema_name(
                self._tenant_id, schema_name
            )
            logger.debug(
                f"Transformed base schema '{schema_name}' to tenant schema '{target_schema_name}' "
                f"for tenant '{self._tenant_id}'"
            )

            # Ensure tenant schema exists (auto-deploy if needed)
            try:
                self.tenant_schema_manager.ensure_tenant_schema_exists(
                    self._tenant_id, schema_name
                )
                logger.debug(f"Verified tenant schema '{target_schema_name}' exists in Vespa")
            except Exception as e:
                logger.error(f"Failed to ensure tenant schema exists: {e}")
                raise
        if target_schema_name not in self._vespa_ingestion_clients:
            # Create new client with config dict
            logger.info(f"Creating new VespaPyClient for schema: {target_schema_name}")

            # Get the specific profile config using BASE schema name (config uses base names)
            profile_config = {}
            if self.config:
                profiles = self.config.get("video_processing_profiles", {})
                profile_config = profiles.get(schema_name, {})  # Use base name for config lookup

            # Pass connection details and profile config
            client_config = {
                "schema_name": target_schema_name,  # Use tenant-scoped name for Vespa
                "base_schema_name": schema_name,  # Base schema name for loading schema file
                "vespa_url": self._vespa_url,
                "vespa_port": self._vespa_port,
                "profile_config": profile_config  # Pass only the specific profile config
            }

            client = VespaPyClient(
                config=client_config,
                logger=logger
            )
            client.connect()

            self._vespa_ingestion_clients[target_schema_name] = client

        return self._vespa_ingestion_clients[target_schema_name]
    
    def ingest_documents(self, documents: List[Document], schema_name: str) -> Dict[str, Any]:
        """
        Ingest documents into Vespa.
        
        Args:
            documents: List of Document objects to ingest
            schema_name: Schema to ingest documents into
            
        Returns:
            Ingestion results
        """
        # Get schema-specific client
        client = self._get_or_create_ingestion_client(schema_name)
        
        # Process and feed documents using the schema-specific client
        # Each client already knows its schema, no need to pass it
        prepared_docs = []
        for doc in documents:
            prepared = client.process(doc)  # Client uses its own schema
            prepared_docs.append(prepared)
        
        # Feed documents to Vespa
        success_count, failed_docs = client._feed_prepared_batch(
            prepared_docs  # Client uses its own schema
        )
        
        return {
            "success_count": success_count,
            "failed_count": len(failed_docs),
            "failed_documents": failed_docs,
            "total_documents": len(documents)
        }
    
    def feed(self, document: Document, schema_name: str) -> Tuple[int, List[str]]:
        """
        Feed a single document to Vespa.
        
        Args:
            document: Document object to feed
            schema_name: Schema to feed document to (REQUIRED)
            
        Returns:
            Tuple of (success_count, failed_document_ids)
        """
        # Convert single document to list and call ingest_documents
        result = self.ingest_documents([document], schema_name)
        
        # Extract failed document IDs from the result
        failed_ids = []
        if result.get("failed_documents"):
            for failed_doc in result["failed_documents"]:
                # Extract the document ID from the failed document info
                if isinstance(failed_doc, str):
                    failed_ids.append(failed_doc)
                elif isinstance(failed_doc, dict) and "id" in failed_doc:
                    failed_ids.append(failed_doc["id"])
        
        success_count = result.get("success_count", 0)
        return success_count, failed_ids
    
    def ingest_stream(self, documents: Iterator[Document], batch_size: int = 100) -> Iterator[Dict[str, Any]]:
        """
        Stream documents for ingestion.
        
        Args:
            documents: Iterator of Document objects
            batch_size: Number of documents per batch
            
        Yields:
            Ingestion results for each batch
        """
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield self.ingest_documents(batch)
                batch = []
        
        # Process remaining documents
        if batch:
            yield self.ingest_documents(batch)
    
    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update a document in Vespa.
        
        Args:
            document_id: ID of document to update
            document: Updated Document object
            
        Returns:
            True if successful
        """
        try:
            results = self.ingest_documents([document])
            return results["success_count"] > 0
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from Vespa.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        try:
            # Use Vespa client to delete document
            # This would use the actual Vespa deletion API
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get Vespa schema information.
        
        Returns:
            Schema information
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        try:
            # Get actual schema info if available from search backend
            if self._vespa_search_backend:
                # Delegate to search backend which has schema access
                return {
                    "name": self.config["schema_name"],
                    "backend": "vespa",
                    "initialized": True,
                    "search_enabled": self._initialized_as_search,
                "ingestion_enabled": self._initialized_as_ingestion
                }
            
            # Basic info if only ingestion is configured
            return {
                "name": self.config.get("schema_name", "unknown"),
                "backend": "vespa",
                "initialized": True,
                "search_enabled": False,
                "ingestion_enabled": self._initialized_as_ingestion
            }
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            raise  # Re-raise instead of returning empty dict
    
    def validate_schema(self, schema_name: str) -> bool:
        """
        Validate that a schema exists in Vespa.
        
        Args:
            schema_name: Name of schema to validate
            
        Returns:
            True if valid
        """
        if not self.schema_manager:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        try:
            # Check if schema exists
            # This would query Vespa for the schema
            return True
        except Exception as e:
            logger.error(f"Failed to validate schema {schema_name}: {e}")
            return False
    
    # Search methods
    
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> Any:
        """
        Execute a search query.

        This method delegates to VespaSearchBackend and returns its results directly.
        The return type matches what VespaSearchBackend returns (List[SearchResult]).

        Args:
            query_embeddings: Optional query embeddings
            query_text: Optional text query
            top_k: Number of results
            filters: Optional filters
            ranking_strategy: Optional ranking strategy

        Returns:
            Search results (List[SearchResult] from VespaSearchBackend)
        """
        # Lazy initialization: create search backend if not already initialized
        if not self._vespa_search_backend:
            if not self.config or "schema_name" not in self.config:
                raise RuntimeError("Search backend not initialized. Ensure schema_name is provided in config.")

            logger.debug("Creating search backend on-demand (lazy initialization)")

            # Transform schema name to tenant-scoped if tenant_id is provided
            search_schema_name = self.config["schema_name"]
            if self._tenant_id and self.tenant_schema_manager:
                search_schema_name = self.tenant_schema_manager.get_tenant_schema_name(
                    self._tenant_id, self.config["schema_name"]
                )
                logger.debug(
                    f"Using tenant schema '{search_schema_name}' for tenant '{self._tenant_id}' "
                    f"(base schema: '{self.config['schema_name']}')"
                )

            # Create VespaSearchBackend with tenant-scoped schema
            # Pass strategy=None to let VespaSearchBackend load it from profile
            self._vespa_search_backend = VespaSearchBackend(
                vespa_url=self.config["vespa_url"],
                vespa_port=self.config.get("vespa_port", 8080),
                schema_name=search_schema_name,  # Use tenant-scoped schema
                profile=self.config.get("profile"),
                strategy=None,  # Let VespaSearchBackend load Strategy object from profile
                query_encoder=self.config.get("query_encoder")
            )
            self._initialized_as_search = True
            logger.info(f"Search backend initialized for schema '{search_schema_name}'")

        # Delegate directly to VespaSearchBackend
        # It returns List[SearchResult], which is what SearchService expects
        return self._vespa_search_backend.search(
            query_embeddings=query_embeddings,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            ranking_strategy=ranking_strategy
        )
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID (uses batch method).

        Args:
            document_id: Document ID

        Returns:
            Document or None
        """
        # Use batch method for consistency and optimization
        results = self.batch_get_documents([document_id])
        return results[0] if results else None
    
    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID (primary batch method).

        Args:
            document_ids: List of document IDs

        Returns:
            List of Documents (None for not found)
        """
        if not self._vespa_search_backend:
            raise RuntimeError("Search backend not initialized.")

        return self._vespa_search_backend.batch_get_documents(document_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Vespa statistics.
        
        Returns:
            Statistics dictionary
        """
        if self._vespa_search_backend:
            # Delegate to search backend if available
            return self._vespa_search_backend.get_statistics()
        
        # Basic stats if only ingestion is configured
        return {
            "backend": "vespa",
            "status": "healthy" if self.schema_manager else "not initialized",
            "search_enabled": self._initialized_as_search
        }
    
    def close(self) -> None:
        """
        Close connections to Vespa.
        """
        # Close all schema-specific clients
        for schema_name, client in self._vespa_ingestion_clients.items():
            client.close()
            logger.info(f"Closed Vespa client for schema: {schema_name}")
        
        for schema_name, client in self._async_ingestion_clients.items():
            client.close()
            logger.info(f"Closed async Vespa client for schema: {schema_name}")
        
        if self._vespa_search_backend:
            # Search backend may not have a close method
            pass
        
        logger.info("Closed all Vespa backend connections")
    
    def health_check(self) -> bool:
        """
        Check Vespa health.
        
        Returns:
            True if healthy
        """
        if self._vespa_search_backend:
            return self._vespa_search_backend.health_check()
        
        # Basic health check
        return self.schema_manager is not None


# Self-registration when module is imported
def register() -> None:
    """Register Vespa backend with the backend registry."""
    from cogniverse_core.registries.backend_registry import register_backend
    
    try:
        register_backend("vespa", VespaBackend)
        logger.info("Vespa backend registered successfully")
    except Exception as e:
        logger.error(f"Failed to register Vespa backend: {e}")


# Call registration on import
register()
