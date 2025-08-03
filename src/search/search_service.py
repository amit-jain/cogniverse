"""Unified search service that coordinates query encoding and backend search."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.models import get_or_load_model
from src.agents.query_encoders import QueryEncoderFactory
from src.core.strategy_registry import get_registry
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
        
        # Get strategy from registry
        self.registry = get_registry()
        self.strategy = self.registry.get_strategy(profile)
        
        # Initialize query encoder first
        self._init_query_encoder()
        
        # Initialize search backend with strategy and query encoder
        self._init_search_backend()
    
    def _init_query_encoder(self):
        """Initialize query encoder based on strategy."""
        model_name = self.strategy.model_name
        logger.info(f"Profile {self.profile} has embedding_model: {model_name}")
        
        # Create query encoder using strategy information
        logger.info(f"Creating query encoder for profile: {self.profile} with model: {model_name}")
        self.query_encoder = QueryEncoderFactory.create_encoder(self.profile, model_name)
        logger.info(f"Initialized query encoder type: {type(self.query_encoder).__name__} for profile: {self.profile}")
    
    def _init_search_backend(self):
        """Initialize search backend with strategy."""
        # For now, only Vespa is supported
        backend_type = self.config.get("search_backend", "vespa")
        
        if backend_type == "vespa":
            vespa_url = self.config.get("vespa_url", "http://localhost")
            vespa_port = self.config.get("vespa_port", 8080)
            
            # Use schema from strategy
            schema_name = self.strategy.schema_name
            
            self.search_backend = VespaSearchBackend(
                vespa_url=vespa_url,
                vespa_port=vespa_port,
                schema_name=schema_name,
                profile=self.profile,
                strategy=self.strategy,  # Pass the strategy object
                query_encoder=self.query_encoder
            )
        else:
            raise ValueError(f"Unsupported search backend: {backend_type}")
        
        logger.info(f"Initialized {backend_type} search backend with schema: {schema_name}")
    
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
        # Let backend decide if embeddings are needed based on ranking strategy
        logger.info(f"Searching with backend...")
        query_embeddings = None
        
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