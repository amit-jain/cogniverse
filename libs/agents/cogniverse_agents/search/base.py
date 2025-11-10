"""Base search interface for different backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from cogniverse_sdk.document import Document


class SearchResult:
    """Represents a search result with document and score."""
    
    def __init__(self, document: Document, score: float, highlights: Optional[Dict[str, Any]] = None):
        self.document = document
        self.score = score
        self.highlights = highlights or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "document_id": self.document.id,
            "score": self.score,
            "metadata": self.document.metadata,
            "highlights": self.highlights
        }
        
        # Add source_id if present in metadata
        if "source_id" in self.document.metadata:
            result["source_id"] = self.document.metadata["source_id"]
        
        # Add temporal info if present in metadata
        if "start_time" in self.document.metadata and "end_time" in self.document.metadata:
            result["temporal_info"] = {
                "start_time": self.document.metadata["start_time"],
                "end_time": self.document.metadata["end_time"],
                "duration": self.document.metadata["end_time"] - self.document.metadata["start_time"]
            }
        
        return result


class SearchBackend(ABC):
    """Abstract base class for search backends."""
    
    @abstractmethod
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query_embeddings: Optional query embeddings from encoder (generated on-demand if None)
            query_text: Original query text
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            ranking_strategy: Optional ranking strategy override
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        pass
    
    @abstractmethod
    def export_embeddings(
        self,
        schema: str = "video_frame",
        max_documents: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Export documents with embeddings from the backend.
        
        Args:
            schema: Schema/index to export from
            max_documents: Maximum number of documents to export
            filters: Optional filters (e.g., video_id, date range)
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of document dictionaries with embeddings and metadata
        """
        pass
