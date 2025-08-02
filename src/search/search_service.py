"""Unified search service that coordinates query encoding and backend search."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.models import get_or_load_model
from src.agents.query_encoders import QueryEncoderFactory
from .search import SearchBackend, SearchResult
from .vespa_search_backend import VespaSearchBackend

logger = logging.getLogger(__name__)


class SearchService:
    """Unified search service for video retrieval."""
    
    def __init__(self, config: Dict[str, Any], profile: str):
        """
        Initialize search service.
        
        Args:
            config: Configuration dictionary
            profile: Video processing profile to use
        """
        self.config = config
        self.profile = profile
        
        # Initialize query encoder
        self._init_query_encoder()
        
        # Initialize search backend
        self._init_search_backend()
    
    def _init_query_encoder(self):
        """Initialize query encoder based on profile."""
        profiles = self.config.get("video_processing_profiles", {})
        if self.profile not in profiles:
            raise ValueError(f"Unknown profile: {self.profile}")
        
        profile_config = profiles[self.profile]
        model_name = profile_config.get("embedding_model")
        
        # Create query encoder
        self.query_encoder = QueryEncoderFactory.create_encoder(self.profile, model_name)
        logger.info(f"Initialized query encoder for profile: {self.profile}")
    
    def _init_search_backend(self):
        """Initialize search backend."""
        # For now, only Vespa is supported
        backend_type = self.config.get("search_backend", "vespa")
        
        if backend_type == "vespa":
            vespa_url = self.config.get("vespa_url", "http://localhost")
            vespa_port = self.config.get("vespa_port", 8080)
            
            # Get schema from profile
            profiles = self.config.get("video_processing_profiles", {})
            schema_name = profiles[self.profile].get("vespa_schema", "video_frame")
            
            self.search_backend = VespaSearchBackend(
                vespa_url=vespa_url,
                vespa_port=vespa_port,
                schema_name=schema_name,
                profile=self.profile
            )
        else:
            raise ValueError(f"Unsupported search backend: {backend_type}")
        
        logger.info(f"Initialized {backend_type} search backend")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for videos matching the query.
        
        Args:
            query: Text query
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            ranking_strategy: Optional ranking strategy override
            
        Returns:
            List of SearchResult objects
        """
        # Encode query
        logger.info(f"Encoding query: '{query}'")
        query_embeddings = self.query_encoder.encode(query)
        logger.info(f"Query embeddings shape: {query_embeddings.shape}")
        
        # Search
        logger.info(f"Searching with backend...")
        if ranking_strategy:
            logger.info(f"Using ranking strategy: {ranking_strategy}")
        
        results = self.search_backend.search(
            query_embeddings=query_embeddings,
            query_text=query,
            top_k=top_k,
            filters=filters,
            ranking_strategy=ranking_strategy
        )
        
        return results
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document as dictionary or None if not found
        """
        doc = self.search_backend.get_document(document_id)
        if doc:
            return {
                "document_id": doc.document_id,
                "source_id": doc.source_id,
                "media_type": doc.media_type.value,
                "metadata": doc.metadata,
                "temporal_info": {
                    "start_time": doc.temporal_info.start_time,
                    "end_time": doc.temporal_info.end_time
                } if doc.temporal_info else None
            }
        return None