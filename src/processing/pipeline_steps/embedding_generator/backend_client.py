#!/usr/bin/env python3
"""
Backend Client Interface - Abstract base for different search backends.

This module defines the interface that all backend clients must implement.
Backend clients are responsible for:

1. Converting Documents to backend-specific formats
2. Managing connections to the backend service
3. Feeding documents for storage/indexing
4. Providing backend-specific embedding processors

The key abstraction is the `process()` method that converts a universal
Document to the backend's required format, handling all format-specific
details like hex encoding for Vespa's bfloat16 tensors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from src.core import Document


class BackendClient(ABC):
    """
    Abstract base class for backend clients.
    
    This class defines the interface for backend storage systems (Vespa,
    Elasticsearch, etc.). Each backend handles its own document format
    conversion and connection management.
    
    The client follows a two-phase approach:
    1. process(): Convert Document to backend-specific format
    2. feed(): Send processed documents to backend
    
    This separation allows for batch processing and error handling.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        schema_name: str,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.schema_name = schema_name
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the backend"""
        pass
    
    @abstractmethod
    def process(self, doc: Document) -> Dict[str, Any]:
        """
        Convert universal Document to backend-specific format.
        
        This is where backend-specific transformations happen:
        - Embedding format conversion (e.g., numpy to hex for Vespa)
        - Field mapping (Document fields to schema fields)
        - Metadata extraction and formatting
        
        Args:
            doc: Universal Document with raw embeddings and metadata
            
        Returns:
            Dict containing backend-specific document structure
            
        Example for Vespa:
            {
                "put": "id:namespace:schema::doc_id",
                "fields": {
                    "embedding": {0: "hex_string", ...},
                    "video_id": "video123",
                    ...
                }
            }
        """
        pass
    
    def feed(
        self,
        documents: Union[Document, List[Document]],
        batch_size: int = 100
    ) -> Tuple[int, List[str]]:
        """
        Feed documents to the backend.
        
        This method accepts both single documents and lists, making it
        flexible for different use cases. It handles:
        - Document conversion via process()
        - Batch processing for efficiency
        - Error tracking and reporting
        
        Args:
            documents: Single Document or list of Documents
            batch_size: Batch size for processing (backend-specific limits)
            
        Returns:
            Tuple of (success_count, failed_doc_ids)
            - success_count: Number of successfully fed documents
            - failed_doc_ids: List of document IDs that failed
            
        Example:
            success, failed = client.feed(doc)
            if success == 0:
                logger.error(f"Failed to feed document: {failed[0]}")
        """
        # Ensure we have a list
        if isinstance(documents, Document):
            documents = [documents]
        
        # Convert all documents
        prepared_docs = [self.process(doc) for doc in documents]
        
        # Feed using batch method
        return self._feed_prepared_batch(prepared_docs, batch_size)
    
    @abstractmethod
    def _feed_prepared_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, List[str]]:
        """Feed prepared documents in batches"""
        pass
    
    @abstractmethod
    def check_document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the backend"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the backend"""
        pass
    
    @abstractmethod
    def close(self):
        """Close the connection"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()