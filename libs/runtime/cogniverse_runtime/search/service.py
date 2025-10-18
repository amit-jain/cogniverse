"""Unified search service that coordinates query encoding and backend search."""

import logging
from typing import Any, Dict, List, Optional

from cogniverse_agents.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import get_backend_registry
from cogniverse_core.registries.registry import get_registry

from .base import SearchResult

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
        
        # Initialize new telemetry system
        from cogniverse_core.telemetry.manager import get_telemetry_manager
        get_telemetry_manager()  # Initialize singleton
        
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
        """Initialize search backend with strategy using backend registry."""
        backend_type = self.config.get("search_backend", "vespa")
        
        # Get backend from registry
        backend_registry = get_backend_registry()
        
        # Prepare backend configuration
        backend_config = {
            "vespa_url": self.config["vespa_url"],  # Required
            "vespa_port": self.config.get("vespa_port", 8080),
            "schema_name": self.strategy.schema_name,
            "profile": self.profile,
            "strategy": self.strategy,
            "query_encoder": self.query_encoder
        }
        
        # Get backend instance from registry
        self.search_backend = backend_registry.get_search_backend(
            backend_type, 
            backend_config
        )
        logger.info(f"Initialized {backend_type} search backend with schema: {self.strategy.schema_name}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for videos matching the query.
        
        Args:
            query: Text query
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            ranking_strategy: Optional ranking strategy override
            tenant_id: Optional tenant identifier for multi-tenant telemetry
            
        Returns:
            List of SearchResult objects
        """
        # Use new multi-tenant telemetry system
        from cogniverse_core.telemetry.context import (
            add_embedding_details_to_span,
            add_search_results_to_span,
            backend_search_span,
            encode_span,
            search_span,
        )
        
        # Default tenant if not provided (for backwards compatibility)
        effective_tenant_id = tenant_id or "default"
        
        logger.info(f"Searching with backend for tenant: {effective_tenant_id}")
        
        with search_span(
            tenant_id=effective_tenant_id,
            query=query,
            top_k=top_k,
            ranking_strategy=ranking_strategy or "default",
            profile=self.profile,
            backend=self.config.get("search_backend", "vespa")
        ) as search_span_ctx:
            
            if ranking_strategy:
                logger.info(f"Using ranking strategy: {ranking_strategy}")
            
            # Generate embeddings with telemetry
            query_embeddings = None
            if self.query_encoder:
                encoder_type = type(self.query_encoder).__name__.lower().replace("queryencoder", "")
                logger.info(f"Generating embeddings with {type(self.query_encoder).__name__}")
                
                with encode_span(
                    tenant_id=effective_tenant_id,
                    encoder_type=encoder_type,
                    query_length=len(query),
                    query=query
                ) as encode_span_ctx:
                    query_embeddings = self.query_encoder.encode(query)
                    # Add embedding details to span
                    add_embedding_details_to_span(encode_span_ctx, query_embeddings)
            
            # Add embeddings info to search span
            if query_embeddings is not None:
                search_span_ctx.set_attribute("has_embeddings", True)
                search_span_ctx.set_attribute("embedding_shape", str(query_embeddings.shape))
            else:
                search_span_ctx.set_attribute("has_embeddings", False)
            
            # Call backend with embeddings and telemetry
            with backend_search_span(
                tenant_id=effective_tenant_id,
                backend_type="vespa",
                schema_name=self.strategy.schema_name,
                ranking_strategy=ranking_strategy or "default",
                top_k=top_k,
                has_embeddings=query_embeddings is not None,
                query_text=query
            ) as backend_span_ctx:
                # Add embeddings info to backend span
                if query_embeddings is not None:
                    backend_span_ctx.set_attribute("embedding_shape", str(query_embeddings.shape))
                
                results = self.search_backend.search(
                    query_embeddings=query_embeddings,
                    query_text=query,
                    top_k=top_k,
                    filters=filters,
                    ranking_strategy=ranking_strategy
                )
                
                # Add result details to backend span
                add_search_results_to_span(backend_span_ctx, results)
            
            # Add result details to search span 
            add_search_results_to_span(search_span_ctx, results)
            
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
                "document_id": doc.id,  # Use new Document structure
                "source_id": doc.content_id,  # Use new Document structure
                "content_type": doc.content_type.value,  # Use new Document structure
                "metadata": doc.metadata,
                "temporal_info": {
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time")
                } if doc.metadata.get("start_time") is not None else None
            }
        return None
