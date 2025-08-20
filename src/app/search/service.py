"""Unified search service that coordinates query encoding and backend search."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os

from src.common.models import get_or_load_model
from src.app.agents.query_encoders import QueryEncoderFactory
from src.common.core.registry import get_registry
from src.common.core.backend_registry import get_backend_registry
from .base import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

# Initialize Phoenix instrumentation if available
PHOENIX_ENABLED = False
tracer = None

def _init_phoenix_instrumentation():
    """Initialize Phoenix instrumentation - called lazily to avoid circular imports"""
    global PHOENIX_ENABLED, tracer
    
    if PHOENIX_ENABLED:  # Already initialized
        return
        
    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
        # Phoenix OTEL registration will handle the tracer provider setup
        
        # Check if tracer provider is already set (e.g., by experiments)
        current_provider = trace.get_tracer_provider()
        
        # Check if we have a real tracer provider (not the default no-op one or proxy)
        provider_type = str(type(current_provider))
        if (current_provider is not None and 
            provider_type != "<class 'opentelemetry.trace._DefaultTracerProvider'>" and
            provider_type != "<class 'opentelemetry.trace.ProxyTracerProvider'>" and
            hasattr(current_provider, 'force_flush')):
            # Reuse existing real tracer provider
            tracer_provider = current_provider
            logger.info(f"Reusing existing tracer provider: {provider_type}")
        else:
            # Use Phoenix's register function to create tracer provider with project name
            from phoenix.otel import register
            
            tracer_provider = register(
                project_name="cogniverse-video-search",  # sets project name for spans
                batch=True,  # uses batch span processor
                auto_instrument=False  # we handle instrumentation manually
            )
            logger.info(f"Created Phoenix tracer provider for project: cogniverse-video-search")
        
        # Get tracer
        tracer = trace.get_tracer(__name__)
        
        # Initialize Cogniverse instrumentation
        from src.app.instrumentation.phoenix import CogniverseInstrumentor
        instrumentor = CogniverseInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        
        logger.info("Phoenix instrumentation initialized")
        PHOENIX_ENABLED = True
    except Exception as e:
        import traceback
        logger.warning(f"Phoenix instrumentation not available: {e}")
        traceback.print_exc()
        PHOENIX_ENABLED = False
        tracer = None


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
        
        # Initialize Phoenix instrumentation on first use
        _init_phoenix_instrumentation()
        
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
        logger.info(f"Searching with backend...")
        
        if ranking_strategy:
            logger.info(f"Using ranking strategy: {ranking_strategy}")
        
        # Generate embeddings at SearchService level (creates encoder span as child of search_service.search)
        query_embeddings = None
        if self.query_encoder:
            logger.info(f"Generating embeddings with {type(self.query_encoder).__name__}")
            query_embeddings = self.query_encoder.encode(query)
        
        # Call backend with embeddings (creates search.execute span as sibling of encoder span)
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