"""
Production-ready Vespa search backend with advanced features.

Features:
- Connection pooling with health checks
- Retry logic with exponential backoff
- Circuit breaker pattern
- Comprehensive metrics collection
- Request correlation and tracing
- Strategy-based search with no hardcoded logic
"""

import json
import time
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime
import logging

import numpy as np
from vespa.application import Vespa

from cogniverse_core.interfaces.backend import SearchBackend, SearchResult
from cogniverse_core.common.document import Document, ContentType, ProcessingStatus
from cogniverse_core.common.models.videoprism_text_encoder import (
    VideoPrismTextEncoder, create_text_encoder
)
from cogniverse_runtime.ingestion.strategy import Strategy
from cogniverse_core.registries.registry import get_registry
from cogniverse_core.common.utils.retry import retry_with_backoff, RetryConfig
from cogniverse_core.common.utils.output_manager import OutputManager
from cogniverse_agents.query.encoders import QueryEncoderFactory, QueryEncoder

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool"""
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    health_check_interval: float = 60.0


@dataclass
class SearchMetrics:
    """Comprehensive search metrics"""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    total_latency_ms: float = 0
    search_latencies: List[float] = field(default_factory=list)
    strategy_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_search(
        self,
        success: bool,
        latency_ms: float,
        strategy: str,
        error: Optional[Exception] = None
    ):
        """Record search metrics"""
        self.total_searches += 1
        self.total_latency_ms += latency_ms
        self.search_latencies.append(latency_ms)
        
        if success:
            self.successful_searches += 1
        else:
            self.failed_searches += 1
            if error:
                self.error_types[type(error).__name__] += 1
                
        self.strategy_usage[strategy] += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_searches == 0:
            return 100.0
        return (self.successful_searches / self.total_searches) * 100
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if not self.search_latencies:
            return 0.0
        return sum(self.search_latencies) / len(self.search_latencies)
    
    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency"""
        if not self.search_latencies:
            return 0.0
        sorted_latencies = sorted(self.search_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class VespaConnection:
    """Managed Vespa connection with health checking"""
    
    def __init__(self, url: str, connection_id: str):
        self.url = url
        self.connection_id = connection_id
        self.vespa = Vespa(url=url)
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_healthy = True
        self._lock = threading.Lock()
        
    def query(self, *args, **kwargs):
        """Execute query and update last used time"""
        with self._lock:
            self.last_used = time.time()
        return self.vespa.query(*args, **kwargs)
    
    def health_check(self) -> bool:
        """Check connection health"""
        try:
            # Simple health check query
            result = self.vespa.query(
                yql="select * from sources * where true limit 1"
            )
            self.is_healthy = result is not None
            return self.is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {self.connection_id}: {e}")
            self.is_healthy = False
            return False
    
    @property
    def idle_time(self) -> float:
        """Time since last use in seconds"""
        return time.time() - self.last_used


class ConnectionPool:
    """Thread-safe connection pool with health monitoring"""
    
    def __init__(self, url: str, config: ConnectionPoolConfig):
        self.url = url
        self.config = config
        self._connections: List[VespaConnection] = []
        self._available: List[VespaConnection] = []
        self._lock = threading.Lock()
        self._stop_health_check = threading.Event()
        
        # Initialize minimum connections
        self._initialize_connections()
        
        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
    
    def _initialize_connections(self):
        """Create initial connections"""
        for i in range(self.config.min_connections):
            conn = VespaConnection(
                self.url,
                f"conn-{uuid.uuid4().hex[:8]}"
            )
            self._connections.append(conn)
            self._available.append(conn)
            logger.info(f"Created connection {conn.connection_id}")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        start_time = time.time()
        
        try:
            # Try to get available connection
            with self._lock:
                if self._available:
                    conn = self._available.pop()
                elif len(self._connections) < self.config.max_connections:
                    # Create new connection if under limit
                    conn = VespaConnection(
                        self.url,
                        f"conn-{uuid.uuid4().hex[:8]}"
                    )
                    self._connections.append(conn)
                    logger.info(f"Created new connection {conn.connection_id}")
            
            # Wait for connection if none available
            while conn is None and (time.time() - start_time) < self.config.connection_timeout:
                time.sleep(0.1)
                with self._lock:
                    if self._available:
                        conn = self._available.pop()
            
            if conn is None:
                raise TimeoutError("No connections available")
                
            yield conn
            
        finally:
            # Return connection to pool
            if conn is not None:
                with self._lock:
                    self._available.append(conn)
    
    def _health_check_loop(self):
        """Periodic health check for all connections"""
        while not self._stop_health_check.is_set():
            try:
                unhealthy = []
                
                with self._lock:
                    for conn in self._connections:
                        if not conn.health_check():
                            unhealthy.append(conn)
                        elif conn.idle_time > self.config.idle_timeout:
                            # Remove idle connections above minimum
                            if len(self._connections) > self.config.min_connections:
                                unhealthy.append(conn)
                
                # Remove unhealthy connections
                for conn in unhealthy:
                    self._remove_connection(conn)
                    
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
            self._stop_health_check.wait(self.config.health_check_interval)
    
    def _remove_connection(self, conn: VespaConnection):
        """Remove a connection from the pool"""
        with self._lock:
            if conn in self._connections:
                self._connections.remove(conn)
            if conn in self._available:
                self._available.remove(conn)
        logger.info(f"Removed connection {conn.connection_id}")
    
    def close(self):
        """Close all connections and stop health checks"""
        self._stop_health_check.set()
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        
        with self._lock:
            self._connections.clear()
            self._available.clear()


class VespaSearchBackend(SearchBackend):
    """Production-ready Vespa search backend"""

    def __init__(
        self,
        vespa_url: str = None,
        vespa_port: int = None,
        schema_name: str = None,
        profile: str = None,  # Make profile optional
        strategy: Optional[Strategy] = None,
        query_encoder: Optional[Any] = None,
        enable_metrics: bool = True,
        enable_connection_pool: bool = True,
        pool_config: Optional[ConnectionPoolConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize Vespa search backend
        
        Args:
            vespa_url: Vespa URL
            vespa_port: Vespa port
            schema_name: Name of the Vespa schema
            profile: Video processing profile
            strategy: Optional Strategy object (will be loaded if not provided)
            query_encoder: Optional query encoder instance
            enable_metrics: Whether to collect metrics
            enable_connection_pool: Whether to use connection pooling
            pool_config: Connection pool configuration
            retry_config: Retry configuration
        """
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.schema_name = schema_name
        self.profile = profile
        self.query_encoder = query_encoder
        
        # Get strategy from registry if profile provided
        if strategy is None and profile is not None:
            registry = get_registry()
            self.strategy = registry.get_strategy(profile)
        else:
            self.strategy = strategy  # Can be None for export-only usage
        
        # Combine URL and port
        full_url = f"{vespa_url}:{vespa_port}"
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # Setup connection pool
        if enable_connection_pool:
            self.pool = ConnectionPool(
                full_url,
                pool_config or ConnectionPoolConfig()
            )
        else:
            self.pool = None
            self.vespa = Vespa(url=vespa_url, port=vespa_port)
        
        # Setup retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            exceptions=(Exception,)
        )
        
        # Initialize metrics
        if enable_metrics:
            self.metrics = SearchMetrics()
        else:
            self.metrics = None
        
        if self.strategy:
            logger.info(
                f"VespaSearchBackend initialized for schema '{schema_name}' "
                f"with pool={enable_connection_pool}, metrics={enable_metrics}, "
                f"strategy={self.strategy.processing_type}/{self.strategy.segmentation}"
            )
        else:
            logger.info(
                f"VespaSearchBackend initialized for schema '{schema_name}' "
                f"with pool={enable_connection_pool}, metrics={enable_metrics}, "
                f"strategy=None (export-only mode)"
            )

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize search backend (already done in __init__).
        This method exists for SearchBackend interface compatibility.
        """
        # VespaSearchBackend uses __init__ for initialization
        # This method is a no-op since init is already done
        pass

    def batch_get_documents(self, document_ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID using batch search query (primary batch method).

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of Document objects (None for not found), in the same order as document_ids
        """
        if not document_ids:
            return []

        # Use search query with document ID filter for efficient batch retrieval
        # Build YQL with documentid() function for precise ID matching
        doc_id_conditions = " OR ".join([f'documentid = "id:video:{self.schema_name}::{doc_id}"' for doc_id in document_ids])
        yql = f"select * from {self.schema_name} where {doc_id_conditions}"

        query_params = {
            "yql": yql,
            "hits": len(document_ids),
            "timeout": "10s"
        }

        try:
            # Execute batch query
            if self.pool:
                with self.pool.get_connection() as conn:
                    response = conn.query(body=query_params)
            else:
                response = self.vespa.query(body=query_params)

            # Build results dictionary for fast lookup
            results_dict = {}
            if response and hasattr(response, 'hits'):
                for hit in response.hits:
                    fields = hit.get("fields", {})
                    # Extract document ID from Vespa document id format
                    full_doc_id = hit.get("id", "")
                    doc_id = full_doc_id.split("::")[-1]

                    doc = Document(
                        id=doc_id,
                        content_type=ContentType.VIDEO,
                        text_content=fields.get("content", ""),
                        status=ProcessingStatus.COMPLETED
                    )

                    # Add all fields as metadata
                    for key, value in fields.items():
                        if value is not None:
                            doc.add_metadata(key, value)

                    results_dict[doc_id] = doc

            # Return results in the same order as input document_ids
            return [results_dict.get(doc_id) for doc_id in document_ids]

        except Exception as e:
            logger.error(f"Batch document retrieval failed: {e}")
            # Fallback to individual retrieval if batch fails
            return self._fallback_individual_get(document_ids)

    def _fallback_individual_get(self, document_ids: List[str]) -> List[Optional[Document]]:
        """Fallback method for individual document retrieval when batch query fails."""
        results = []
        for doc_id in document_ids:
            try:
                if self.pool:
                    with self.pool.get_connection() as conn:
                        response = conn.vespa.get_data(
                            schema=self.schema_name,
                            data_id=doc_id,
                            namespace="video"
                        )
                else:
                    response = self.vespa.get_data(
                        schema=self.schema_name,
                        data_id=doc_id,
                        namespace="video"
                    )

                if response and response.status_code == 200:
                    data = response.json()
                    fields = data.get("fields", {})

                    doc = Document(
                        id=doc_id,
                        content_type=ContentType.VIDEO,
                        text_content=fields.get("content", ""),
                        status=ProcessingStatus.COMPLETED
                    )

                    for key, value in fields.items():
                        if value is not None:
                            doc.add_metadata(key, value)

                    results.append(doc)
                else:
                    results.append(None)

            except Exception as e:
                logger.error(f"Failed to retrieve document {doc_id}: {e}")
                results.append(None)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search backend statistics.

        Returns:
            Statistics including document count, metrics, etc.
        """
        return self.get_metrics()

    def _embeddings_to_vespa_format(self, embeddings: np.ndarray, profile: str) -> Dict[str, Any]:
        """Convert embeddings to Vespa query format."""
        # For binary embeddings, handle differently
        if "_binary" in profile:
            # For binary tensors, values should be int8
            if embeddings.ndim == 1:
                # 1D binary embeddings (global models)
                cells = [{"address": {"v": str(i)}, "value": int(val)} 
                        for i, val in enumerate(embeddings)]
                return {"cells": cells}
            else:
                # 2D binary embeddings (patch-based models like ColPali)
                cells = []
                for patch_idx in range(embeddings.shape[0]):
                    for v_idx in range(embeddings.shape[1]):
                        cells.append({
                            "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                            "value": int(embeddings[patch_idx, v_idx])
                        })
                return {"cells": cells}
        # For global profiles, embeddings are 1D
        elif "global" in profile:
            # Convert to tensor cells format for Vespa
            cells = [{"address": {"v": str(i)}, "value": float(val)} 
                    for i, val in enumerate(embeddings)]
            return {"cells": cells}
        else:
            # For patch-based models, embeddings are 2D
            cells = []
            for patch_idx in range(embeddings.shape[0]):
                for v_idx in range(embeddings.shape[1]):
                    cells.append({
                        "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                        "value": float(embeddings[patch_idx, v_idx])
                    })
            return {"cells": cells}
    
    def _generate_binary_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate binary embeddings from float embeddings."""
        # Binarize embeddings (>0 becomes 1, <=0 becomes 0)
        binary = np.where(embeddings > 0, 1, 0).astype(np.uint8)
        
        # Pack bits into bytes
        if len(binary.shape) == 1:
            # 1D embeddings (global)
            # Pad to multiple of 8 bits
            padding = (8 - len(binary) % 8) % 8
            if padding:
                binary = np.pad(binary, (0, padding), mode='constant')
            # Pack bits
            packed = np.packbits(binary).astype(np.int8)
        else:
            # 2D embeddings (patch-based)
            # Pack each patch separately
            packed = np.packbits(binary, axis=1).astype(np.int8)
        
        return packed
    
    @retry_with_backoff
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents. Requires strategy to be configured.
        """
        if self.strategy is None:
            raise RuntimeError(
                "Search not available: VespaSearchBackend was initialized without a strategy. "
                "This instance is configured for export-only operations."
            )
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
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Log search request
            logger.info(
                f"[{correlation_id}] Search request: query='{query_text}', "
                f"limit={top_k}, profile={ranking_strategy}"
            )
            
            # Determine ranking profile
            if ranking_strategy:
                ranking_profile = ranking_strategy
            else:
                ranking_profile = self.strategy.default_ranking
            
            # Validate and get ranking strategy config
            if ranking_profile not in self.strategy.ranking_strategies:
                logger.warning(
                    f"[{correlation_id}] Unknown ranking profile '{ranking_profile}', "
                    f"using default: {self.strategy.default_ranking}"
                )
                ranking_profile = self.strategy.default_ranking
            
            rank_config = self.strategy.ranking_strategies[ranking_profile]
            
            # Check if strategy requires embeddings
            requires_embeddings = (
                rank_config.get("needs_float_embeddings", False) or 
                rank_config.get("needs_binary_embeddings", False)
            )
            
            # Generate embeddings on-demand if needed and not provided
            if requires_embeddings and query_embeddings is None:
                if self.query_encoder:
                    logger.info(f"[{correlation_id}] Generating embeddings on-demand for strategy '{ranking_profile}'")
                    logger.info(f"[{correlation_id}] Query encoder type: {type(self.query_encoder).__name__}")
                    logger.info(f"[{correlation_id}] Query text: '{query_text}'")
                    query_embeddings = self.query_encoder.encode(query_text)
                    logger.info(f"[{correlation_id}] Generated embeddings shape: {query_embeddings.shape}")
                    logger.info(f"[{correlation_id}] Embeddings dtype: {query_embeddings.dtype}")
                    logger.info(f"[{correlation_id}] Embeddings min/max: {query_embeddings.min():.4f}/{query_embeddings.max():.4f}")
                    logger.info(f"[{correlation_id}] First 5 values: {query_embeddings.flatten()[:5]}")
                else:
                    logger.warning(
                        f"[{correlation_id}] Strategy '{ranking_profile}' requires embeddings but no encoder available"
                    )
            
            # Build query based on strategy
            query_params = self._build_query(query_text, query_embeddings, rank_config, ranking_profile, top_k, correlation_id)
            
            # Execute search
            logger.info(f"[{correlation_id}] Executing query with ranking={query_params.get('ranking')}, keys={list(query_params.keys())}")
            
            # Debug qtb type right before sending
            if "input.query(qtb)" in query_params:
                qtb_val = query_params["input.query(qtb)"]
                logger.info(f"[{correlation_id}] RIGHT BEFORE QUERY - qtb type: {type(qtb_val)}, is_list: {isinstance(qtb_val, list)}")
                if isinstance(qtb_val, list) and len(qtb_val) > 0:
                    logger.info(f"[{correlation_id}] qtb[0] type: {type(qtb_val[0])}, value: {qtb_val[0]}")
            
            if self.pool:
                with self.pool.get_connection() as conn:
                    response = conn.query(body=query_params)
            else:
                response = self.vespa.query(body=query_params)
            
            # Process results
            results = self._process_results(response, correlation_id)
            
            # Record metrics
            if self.metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_search(True, latency_ms, ranking_profile or "default")
            
            logger.info(
                f"[{correlation_id}] Search completed: {len(results)} results "
                f"in {(time.time() - start_time)*1000:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            # Record failure metrics
            if self.metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_search(False, latency_ms, ranking_profile or "default", e)
            
            logger.error(f"[{correlation_id}] Search failed: {e}")
            raise
    
    def _build_query(
        self,
        query_text: str,
        query_embeddings: Optional[np.ndarray],
        rank_config: Dict[str, Any],
        ranking_profile: str,
        limit: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Build Vespa query based on ranking strategy - NO HARDCODING!"""
        
        # Initialize query params
        query_params = {
            "ranking.profile": ranking_profile,
            "hits": limit,
            "ranking": ranking_profile,
        }
        
        # Build YQL based on strategy configuration
        if rank_config.get("use_nearestneighbor"):
            # Use nearestNeighbor for visual search
            nn_field = rank_config.get("nearestneighbor_field", "embedding")
            nn_tensor = rank_config.get("nearestneighbor_tensor", "qt")
            
            if rank_config.get("needs_text_query") and query_text:
                # Hybrid search with nearestNeighbor
                query_params["yql"] = (
                    f"select * from {self.schema_name} where "
                    f"userInput(@userQuery) OR "
                    f"({{targetHits: {limit}}}nearestNeighbor({nn_field}, {nn_tensor}))"
                )
                query_params["userQuery"] = query_text
            else:
                # Pure visual search with nearestNeighbor
                query_params["yql"] = (
                    f"select * from {self.schema_name} where "
                    f"{{targetHits: {limit}}}nearestNeighbor({nn_field}, {nn_tensor})"
                )
        elif rank_config.get("needs_text_query"):
            # Text or hybrid search without nearestNeighbor
            query_params["yql"] = f"select * from {self.schema_name} where userInput(@userQuery)"
            query_params["userQuery"] = query_text
        else:
            # Regular ranking without nearestNeighbor (patch-based models)
            query_params["yql"] = f"select * from {self.schema_name} where true"
        
        # Add tensor embeddings based on what the strategy says it needs
        # NO MORE HARDCODED CHECKS!
        if query_embeddings is not None:
            # Check what tensors this ranking strategy needs from its inputs
            inputs_needed = rank_config.get("inputs", {})
            
            logger.info(f"[{correlation_id}] Strategy '{ranking_profile}' needs inputs: {list(inputs_needed.keys())}")
            
            for input_name, input_type in inputs_needed.items():
                # The input names are like 'qt', 'qtb', etc. We need to construct the full Vespa query parameter name
                vespa_param_name = f"input.query({input_name})"
                
                if input_name == "qt" and "float" in input_type:
                    # Float embeddings needed
                    logger.info(f"[{correlation_id}] Adding float embeddings for {vespa_param_name}")
                    # For multi-vector models, convert to dict format as per Vespa docs
                    if query_embeddings.ndim == 2:
                        query_params[vespa_param_name] = {index: vector.tolist() for index, vector in enumerate(query_embeddings)}
                    else:
                        query_params[vespa_param_name] = query_embeddings.tolist()
                    
                elif input_name == "qtb" and "int8" in input_type:
                    # Binary embeddings needed
                    logger.info(f"[{correlation_id}] Adding binary embeddings for {vespa_param_name}")
                    binary_embeddings = self._generate_binary_embeddings(query_embeddings)
                    # For multi-vector models, convert to dict format as per Vespa docs
                    if binary_embeddings.ndim == 2:
                        query_params[vespa_param_name] = {index: vector.tolist() for index, vector in enumerate(binary_embeddings)}
                    else:
                        query_params[vespa_param_name] = binary_embeddings.tolist()
                    
                elif input_name == "q":
                    # Generic query tensor (used by some schemas)
                    logger.info(f"[{correlation_id}] Adding generic embeddings for {vespa_param_name}")
                    query_params[vespa_param_name] = query_embeddings.tolist()
                    
                else:
                    # Unknown input name - this shouldn't happen with proper configuration
                    logger.warning(
                        f"[{correlation_id}] Unknown input name '{input_name}' with type '{input_type}' "
                        f"in ranking strategy '{ranking_profile}'. This input will be skipped. "
                        f"Known input names: 'qt' (float), 'qtb' (binary), 'q' (generic)."
                    )
        
        # Add schema to query body to avoid conflicts with other schemas
        query_params["model.restrict"] = self.schema_name  # Must be string, not list!
        
        # Log the query parameters for debugging
        logger.info(f"[{correlation_id}] Query params keys: {list(query_params.keys())}")
        if "input.query(qt)" in query_params:
            qt_value = query_params["input.query(qt)"]
            logger.info(f"[{correlation_id}] input.query(qt) type: {type(qt_value)}, is list: {isinstance(qt_value, list)}")
            if isinstance(qt_value, list):
                logger.info(f"[{correlation_id}] input.query(qt) length: {len(qt_value)}, first 5 values: {qt_value[:5] if qt_value else 'empty'}...")
        if "input.query(qtb)" in query_params:
            qtb_value = query_params["input.query(qtb)"]
            logger.info(f"[{correlation_id}] input.query(qtb) type: {type(qtb_value)}, is list: {isinstance(qtb_value, list)}")
            if isinstance(qtb_value, list):
                logger.info(f"[{correlation_id}] input.query(qtb) length: {len(qtb_value)}, first 5 values: {qtb_value[:5] if qtb_value else 'empty'}...")
        
        return query_params
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert Vespa result to Document object."""
        fields = result.get("fields", {})
        
        # Extract document ID
        doc_id = result.get("id", "").split("::")[-1]
        
        # Create Document using new generic structure
        document = Document(
            id=doc_id,
            content_type=ContentType.VIDEO,
            status=ProcessingStatus.COMPLETED
        )
        
        # Add all fields as metadata
        for key, value in fields.items():
            if value is not None:
                document.add_metadata(key, value)
        
        # Add source_id to metadata
        document.add_metadata("source_id", fields.get("video_id", doc_id.split("_")[0]))
        
        return document
    
    def _process_results(
        self,
        response: Any,
        correlation_id: str
    ) -> List[SearchResult]:
        """Process Vespa response into SearchResult objects"""
        results = []
        
        if not response or not hasattr(response, 'hits'):
            logger.warning(f"[{correlation_id}] Empty response from Vespa")
            return results
        
        for hit in response.hits:
            try:
                doc = self._result_to_document(hit)
                score = hit.get("relevance", 0.0)
                
                # Extract highlights if available
                highlights = {}
                if "summaryfeatures" in hit:
                    highlights = hit["summaryfeatures"]
                
                results.append(SearchResult(doc, score, highlights))
                
            except Exception as e:
                logger.error(
                    f"[{correlation_id}] Failed to process hit: {e}, "
                    f"hit data: {hit}"
                )
        
        return results
    
    def _extract_metadata(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document fields"""
        metadata = {}
        
        # Standard metadata fields
        for key in ['video_id', 'start_time', 'end_time', 'frame_number', 
                    'description', 'transcription']:
            if key in fields:
                metadata[key] = fields[key]
        
        # Handle temporal info
        if 'start_time' in fields and 'end_time' in fields:
            metadata['duration'] = fields['end_time'] - fields['start_time']
        
        return metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            "search_metrics": {},
            "encoder_metrics": {},
            "pool_metrics": {}
        }
        
        # Search metrics
        if self.metrics:
            metrics["search_metrics"] = {
                "total_searches": self.metrics.total_searches,
                "success_rate": self.metrics.success_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "strategy_usage": dict(self.metrics.strategy_usage),
                "error_types": dict(self.metrics.error_types)
            }
        
        
        # Connection pool metrics
        if self.pool:
            with self.pool._lock:
                metrics["pool_metrics"] = {
                    "total_connections": len(self.pool._connections),
                    "available_connections": len(self.pool._available),
                    "healthy_connections": sum(
                        1 for c in self.pool._connections if c.is_healthy
                    )
                }
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "schema": self.schema_name,
            "components": {}
        }
        
        # Check Vespa connectivity
        try:
            if self.pool:
                with self.pool.get_connection() as conn:
                    conn.health_check()
            else:
                self.vespa.query(yql="select * from sources * where true limit 1")
            health["components"]["vespa"] = "healthy"
        except Exception as e:
            health["status"] = "degraded"
            health["components"]["vespa"] = f"unhealthy: {e}"
        
        
        # Add metrics
        health["metrics"] = self.get_metrics()
        
        return health
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID (uses batch method).

        Args:
            document_id: Document ID to retrieve

        Returns:
            Document if found, None otherwise
        """
        # Use batch method for consistency and optimization
        results = self.batch_get_documents([document_id])
        return results[0] if results else None
    
    def export_embeddings(
        self,
        schema: str = "video_frame",
        max_documents: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Export documents with embeddings from Vespa.
        
        Args:
            schema: Schema to export from (overrides default)
            max_documents: Maximum number of documents to export
            filters: Optional filters (e.g., {'video_id': 'xyz'})
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of document dictionaries with embeddings and metadata
        """
        documents = []
        
        # Build YQL query
        yql = f"select * from {schema or self.schema_name} where true"
        
        # Add filters if provided
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"{key} contains '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
            if conditions:
                yql = f"select * from {schema or self.schema_name} where {' and '.join(conditions)}"
        
        # Add limit
        if max_documents:
            yql += f" limit {max_documents}"
        
        # Use visit API for bulk export
        namespace = "video"  # Default namespace for video schemas
        visit_url = f"{self.vespa_url}:{self.vespa_port}/document/v1/{namespace}/{schema or self.schema_name}/docid"
        
        try:
            import requests
            response = requests.get(
                visit_url,
                params={
                    "selection": "true" if not filters else None,
                    "continuation": None,
                    "wantedDocumentCount": max_documents or 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for doc in data.get("documents", []):
                    fields = doc.get("fields", {})
                    doc_data = {
                        "id": doc.get("id", ""),
                        **fields
                    }
                    
                    # Extract embeddings if present and requested
                    if include_embeddings:
                        for emb_field in ["embedding", "frame_embedding", "video_embedding", 
                                         "text_embedding", "colpali_embedding"]:
                            if emb_field in fields:
                                doc_data[emb_field] = fields[emb_field]
                    
                    documents.append(doc_data)
                
                # Handle continuation token for large datasets
                continuation = data.get("continuation")
                while continuation and len(documents) < (max_documents or float('inf')):
                    response = requests.get(
                        visit_url,
                        params={
                            "continuation": continuation,
                            "wantedDocumentCount": min(1000, (max_documents or 1000) - len(documents))
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for doc in data.get("documents", []):
                            fields = doc.get("fields", {})
                            doc_data = {
                                "id": doc.get("id", ""),
                                **fields
                            }
                            
                            if include_embeddings:
                                for emb_field in ["embedding", "frame_embedding", "video_embedding",
                                                 "text_embedding", "colpali_embedding"]:
                                    if emb_field in fields:
                                        doc_data[emb_field] = fields[emb_field]
                            
                            documents.append(doc_data)
                        
                        continuation = data.get("continuation")
                        if not data.get("documents"):
                            break
                    else:
                        break
                        
        except Exception as e:
            logger.error(f"Failed to export embeddings: {e}")
            raise
        
        return documents[:max_documents] if max_documents else documents
    
    def close(self):
        """Clean up resources"""
        if self.pool:
            self.pool.close()
        
        
        logger.info("VespaSearchBackend closed")


# Factory function for creating search backend
def create_vespa_search_backend(
    schema_name: str,
    vespa_url: str = "http://localhost:8080",
    **kwargs
) -> VespaSearchBackend:
    """
    Create a production-ready Vespa search backend
    
    Args:
        schema_name: Vespa schema name
        vespa_url: Vespa URL
        **kwargs: Additional configuration
        
    Returns:
        VespaSearchBackend instance
    """
    return VespaSearchBackend(
        vespa_url=vespa_url,
        schema_name=schema_name,
        **kwargs
    )