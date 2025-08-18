"""
Abstract base classes for backend implementations.

This module defines the interfaces that all backends must implement
to integrate with the Cogniverse system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import numpy as np

from .documents import Document


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
    def ingest_documents(self, documents: List[Document], schema_name: str) -> Dict[str, Any]:
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
        ranking_strategy: Optional[str] = None
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