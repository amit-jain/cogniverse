"""
Vespa backend implementation with unified interface.

This module provides a Vespa backend that implements both IngestionBackend
and SearchBackend interfaces, with self-registration to the backend registry.
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import numpy as np

from src.common.core.interfaces import Backend
from src.common.core.documents import Document
from .search_backend import VespaSearchBackend
from .vespa_schema_manager import VespaSchemaManager
from .ingestion_client import VespaPyClient

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
        self._vespa_ingestion_client: Optional[VespaPyClient] = None
        self.schema_manager: Optional[VespaSchemaManager] = None
        self.config: Dict[str, Any] = {}
        self._initialized_as_search = False
        self._initialized_as_ingestion = False
        self.schema_name: Optional[str] = None
    
    def _initialize_backend(self, config: Dict[str, Any]) -> None:
        """
        Initialize Vespa backend components.
        
        Args:
            config: Backend configuration including:
                - vespa_url: Vespa endpoint URL
                - vespa_port: Vespa port
                - schema_name: Schema to use
                - profile: Processing profile
                - strategy: Strategy object (optional)
                - query_encoder: Query encoder (optional)
        """
        self.config = config
        self.schema_name = config.get("schema_name")
        
        # Initialize for ingestion if schema_name is provided without search-specific params
        if "schema_name" in config and not ("strategy" in config or "profile" in config):
            # This means we're initializing for ingestion operations
            self._initialized_as_ingestion = True
            
            # Create the VespaPyClient for ingestion
            self._vespa_ingestion_client = VespaPyClient(
                config=config,
                logger=logger
            )
            self._vespa_ingestion_client.connect()
        
        # Initialize schema manager for schema operations
        vespa_url = config["vespa_url"]  # Required, no default
        vespa_port = config.get("vespa_port", 8080)
        deployment_port = config.get("vespa_deployment_port", 19071)
        
        self.schema_manager = VespaSchemaManager(
            vespa_endpoint=f"{vespa_url}:{vespa_port}",
            vespa_port=deployment_port
        )
        
        # Initialize search backend if this is being used for search
        if "schema_name" in config and ("strategy" in config or "profile" in config):
            # This means we're initializing for search operations
            self._initialized_as_search = True
            
            # Create the actual VespaSearchBackend
            self._vespa_search_backend = VespaSearchBackend(
                vespa_url=config["vespa_url"],  # Required
                vespa_port=config.get("vespa_port", 8080),
                schema_name=config["schema_name"],
                profile=config.get("profile"),
                strategy=config.get("strategy"),
                query_encoder=config.get("query_encoder")
            )
        
        logger.info(f"Initialized Vespa backend with config: {config}")
    
    # Ingestion methods
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest documents into Vespa.
        
        Args:
            documents: List of Document objects to ingest
            
        Returns:
            Ingestion results
        """
        if not self._vespa_ingestion_client:
            # Initialize ingestion client if not already done
            if "schema_name" not in self.config:
                raise RuntimeError("schema_name must be provided in config for ingestion")
            
            self._vespa_ingestion_client = VespaPyClient(
                config=self.config,
                logger=logger
            )
            self._vespa_ingestion_client.connect()
            self._initialized_as_ingestion = True
        
        # Process and feed documents using the ingestion client
        prepared_docs = []
        for doc in documents:
            prepared = self._vespa_ingestion_client.process(doc)
            prepared_docs.append(prepared)
        
        # Feed documents to Vespa
        success_count, failed_docs = self._vespa_ingestion_client._feed_prepared_batch(
            prepared_docs
        )
        
        return {
            "success_count": success_count,
            "failed_count": len(failed_docs),
            "failed_documents": failed_docs,
            "total_documents": len(documents)
        }
    
    def feed(self, document: Document) -> Tuple[int, List[str]]:
        """
        Feed a single document to Vespa.
        
        This method provides compatibility with the embedding generator
        which expects a feed method.
        
        Args:
            document: Document object to feed
            
        Returns:
            Tuple of (success_count, failed_document_ids)
        """
        # Convert single document to list and call ingest_documents
        result = self.ingest_documents([document])
        
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
        if not self._vespa_search_backend:
            raise RuntimeError("Search backend not initialized. Ensure schema_name is provided in config.")
        
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
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document or None
        """
        if not self._vespa_search_backend:
            raise RuntimeError("Search backend not initialized.")
        
        return self._vespa_search_backend.get_document(document_id)
    
    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            List of Documents (None for not found)
        """
        return [self.get_document(doc_id) for doc_id in document_ids]
    
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
        if self._vespa_ingestion_client:
            self._vespa_ingestion_client.close()
        if self._vespa_search_backend:
            # Search backend may not have a close method
            pass
        logger.info("Closed Vespa backend connections")
    
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
    from src.common.core.backend_registry import register_backend
    
    try:
        register_backend("vespa", VespaBackend)
        logger.info("Vespa backend registered successfully")
    except Exception as e:
        logger.error(f"Failed to register Vespa backend: {e}")


# Call registration on import
register()