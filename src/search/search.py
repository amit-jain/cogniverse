"""Base search interface for different backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from src.core import Document


class SearchResult:
    """Represents a search result with document and score."""
    
    def __init__(self, document: Document, score: float, highlights: Optional[Dict[str, Any]] = None):
        self.document = document
        self.score = score
        self.highlights = highlights or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "document_id": self.document.doc_id,
            "score": self.score,
            "metadata": self.document.metadata,
            "highlights": self.highlights
        }
        
        # Add source_id if present in metadata
        if "source_id" in self.document.metadata:
            result["source_id"] = self.document.metadata["source_id"]
        
        # Add temporal info if present
        if self.document.temporal_info:
            result["temporal_info"] = {
                "start_time": self.document.temporal_info.start_time,
                "end_time": self.document.temporal_info.end_time,
                "duration": self.document.temporal_info.duration
            }
        
        return result


class SearchBackend(ABC):
    """Abstract base class for search backends."""
    
    @abstractmethod
    def search(
        self,
        query_embeddings: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query_embeddings: Query embeddings from encoder
            query_text: Original query text
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        pass