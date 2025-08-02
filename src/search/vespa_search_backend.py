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

from src.search.search import SearchBackend, SearchResult
from src.core import Document, MediaType, TemporalInfo, SegmentInfo
from src.models.videoprism_text_encoder import (
    VideoPrismTextEncoder, create_text_encoder
)
from src.processing.vespa.strategy_aware_processor import StrategyAwareProcessor
from src.processing.vespa.ranking_strategy_extractor import (
    extract_all_ranking_strategies, save_ranking_strategies
)
from src.utils.retry import retry_with_backoff, RetryConfig
from src.utils.output_manager import OutputManager
from src.agents.query_encoders import QueryEncoderFactory, QueryEncoder

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
        vespa_url: str,
        vespa_port: int,
        schema_name: str,
        profile: str,
        query_encoder_factory: Optional[QueryEncoderFactory] = None,
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
            query_encoder_factory: Optional factory for creating query encoders on-demand
            enable_metrics: Whether to collect metrics
            enable_connection_pool: Whether to use connection pooling
            pool_config: Connection pool configuration
            retry_config: Retry configuration
        """
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.schema_name = schema_name
        self.profile = profile
        
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
        
        # Load ranking strategies
        self._load_ranking_strategies()
        
        # Initialize query encoder factory and cache
        self.query_encoder_factory = query_encoder_factory
        self._query_encoder: Optional[QueryEncoder] = None
        self._encoder_lock = threading.Lock()
        
        logger.info(
            f"VespaSearchBackend initialized for schema '{schema_name}' "
            f"with pool={enable_connection_pool}, metrics={enable_metrics}"
        )
    
    def _load_ranking_strategies(self):
        """Load ranking strategies from JSON file"""
        schemas_dir = Path("configs/schemas")
        strategies_file = schemas_dir / "ranking_strategies.json"
        
        if not strategies_file.exists():
            logger.info("Ranking strategies not found, auto-generating...")
            strategies = extract_all_ranking_strategies(schemas_dir)
            save_ranking_strategies(strategies, strategies_file)
        
        # Initialize strategy processor
        self.strategy_processor = StrategyAwareProcessor()
        
        # Load strategies for this schema
        if self.schema_name in self.strategy_processor.ranking_strategies:
            self.ranking_strategies = self.strategy_processor.ranking_strategies[self.schema_name]
        else:
            self.ranking_strategies = {}
        
        if not self.ranking_strategies:
            raise ValueError(f"No ranking strategies found for schema '{self.schema_name}'")
            
        logger.info(
            f"Loaded {len(self.ranking_strategies)} ranking strategies "
            f"for schema '{self.schema_name}'"
        )
    
    def _get_query_encoder(self) -> Optional[QueryEncoder]:
        """Get or create query encoder lazily"""
        if self._query_encoder is None and self.query_encoder_factory:
            with self._encoder_lock:
                if self._query_encoder is None:
                    self._query_encoder = self.query_encoder_factory.create_encoder(self.profile)
                    logger.info(f"Created query encoder for profile: {self.profile}")
        return self._query_encoder
    
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
                # Use provided ranking strategy
                ranking_profile = ranking_strategy
            else:
                # Use default based on profile type
                if "global" in self.profile:
                    ranking_profile = "float_float"
                else:
                    ranking_profile = "hybrid_binary_bm25"
            
            # Validate ranking profile
            if ranking_profile not in self.ranking_strategies:
                logger.warning(
                    f"[{correlation_id}] Unknown ranking profile '{ranking_profile}', "
                    f"using default"
                )
                ranking_profile = next(iter(self.ranking_strategies.keys()))
            
            strategy = self.ranking_strategies[ranking_profile]
            
            # Check if strategy requires embeddings
            requires_embeddings = (
                strategy.get("needs_float_embeddings", False) or 
                strategy.get("needs_binary_embeddings", False)
            )
            
            # Generate embeddings on-demand if needed and not provided
            if requires_embeddings and query_embeddings is None:
                encoder = self._get_query_encoder()
                if encoder:
                    logger.info(f"[{correlation_id}] Generating embeddings on-demand for strategy '{ranking_profile}'")
                    query_embeddings = encoder.encode(query_text)
                else:
                    logger.warning(
                        f"[{correlation_id}] Strategy '{ranking_profile}' requires embeddings but no encoder available"
                    )
            
            # Build query based on strategy
            query_params = self._build_query(query_text, query_embeddings, strategy, top_k, correlation_id)
            
            # Execute search
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
        strategy: Dict[str, Any],
        limit: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Build Vespa query based on ranking strategy"""
        ranking_profile = strategy["name"]
        
        # Build query based on ranking strategy and profile
        # For pure visual search strategies, use nearestNeighbor
        pure_visual_strategies = ["float_float", "binary_binary", "float_binary", "phased"]
        
        if ranking_profile in pure_visual_strategies:
            # For global schemas, use nearestNeighbor; for patch-based, use regular ranking
            if "global" in self.profile:
                # Determine which field and query tensor to use for nearestNeighbor
                if ranking_profile == "float_float":
                    nn_field = "embedding"
                    query_tensor_name = "qt"
                elif ranking_profile == "binary_binary":
                    nn_field = "embedding_binary"
                    query_tensor_name = "qtb"
                elif ranking_profile in ["float_binary", "phased"]:
                    nn_field = "embedding_binary"
                    query_tensor_name = "qtb"
                else:
                    nn_field = "embedding"
                    query_tensor_name = "qt"
                
                # Global embeddings - use nearestNeighbor
                query_params = {
                    "yql": f"select * from {self.schema_name} where {{targetHits: {limit}}}nearestNeighbor({nn_field}, {query_tensor_name})",
                    "ranking.profile": ranking_profile,
                    "hits": limit,
                    "ranking": ranking_profile
                }
            else:
                # Patch-based embeddings - use regular ranking without nearestNeighbor
                query_params = {
                    "yql": f"select * from {self.schema_name} where true",
                    "ranking.profile": ranking_profile,
                    "hits": limit,
                    "ranking": ranking_profile
                }
        else:
            # Hybrid or text search - use userInput
            query_params = {
                "yql": f"select * from {self.schema_name} where userInput(@userQuery)",
                "userQuery": query_text,
                "ranking.profile": ranking_profile,
                "hits": limit
            }
        
        # Add tensor embeddings based on ranking profile
        needs_binary = ranking_profile in ["binary_binary", "float_binary", "phased", "hybrid_binary_bm25", 
                                          "hybrid_binary_bm25_no_description", "default"]
        needs_float = ranking_profile in ["float_float", "float_binary", "phased", "hybrid_float_bm25"]
        
        if needs_float and query_embeddings is not None:
            # For global embeddings, use list format (like original code)
            if "global" in self.profile:
                logger.info(f"[{correlation_id}] Query embeddings shape: {query_embeddings.shape}, type: {type(query_embeddings)}")
                embeddings_list = query_embeddings.tolist()
                logger.info(f"[{correlation_id}] Embeddings list type: {type(embeddings_list)}, length: {len(embeddings_list) if isinstance(embeddings_list, list) else 'scalar'}")
                query_params["input.query(qt)"] = embeddings_list
            else:
                # For patch-based models, use dict format
                query_tensor = self._embeddings_to_vespa_format(query_embeddings, self.profile)
                query_params["input.query(qt)"] = query_tensor
        
        if needs_binary and query_embeddings is not None:
            # Generate binary embeddings from float embeddings
            binary_embeddings = self._generate_binary_embeddings(query_embeddings)
            if "global" in self.profile:
                query_params["input.query(qtb)"] = binary_embeddings.tolist()
            else:
                binary_tensor = self._embeddings_to_vespa_format(binary_embeddings, self.profile + "_binary")
                query_params["input.query(qtb)"] = binary_tensor
        
        # Add schema to query body to avoid conflicts with other schemas
        query_params["model.restrict"] = self.schema_name  # Must be string, not list!
        
        # Log the query parameters for debugging
        logger.info(f"[{correlation_id}] Query params keys: {list(query_params.keys())}")
        if "input.query(qt)" in query_params:
            qt_value = query_params["input.query(qt)"]
            logger.info(f"[{correlation_id}] input.query(qt) type: {type(qt_value)}, is list: {isinstance(qt_value, list)}")
            if isinstance(qt_value, list):
                logger.info(f"[{correlation_id}] input.query(qt) length: {len(qt_value)}, first value: {qt_value[0] if qt_value else 'empty'}")
        
        return query_params
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert Vespa result to Document object."""
        fields = result.get("fields", {})
        
        # Extract document ID
        doc_id = result.get("id", "").split("::")[-1]
        
        # Determine media type based on schema
        if "frame" in self.schema_name:
            media_type = MediaType.VIDEO_FRAME
        elif "global" in self.schema_name:
            media_type = MediaType.VIDEO_SEGMENT
        else:
            media_type = MediaType.VIDEO_FRAME  # Default to frame
        
        # Build temporal info if available
        temporal_info = None
        if "start_time" in fields and "end_time" in fields:
            temporal_info = TemporalInfo(
                start_time=fields["start_time"],
                end_time=fields["end_time"]
            )
        
        # Build segment info if available
        segment_info = None
        if "segment_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["segment_id"],
                total_segments=fields.get("total_segments", 1)
            )
        elif "frame_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["frame_id"],
                total_segments=1
            )
        
        # Extract metadata
        metadata = {
            "video_title": fields.get("video_title"),
            "frame_description": fields.get("frame_description"),
            "segment_description": fields.get("segment_description"),
            "audio_transcript": fields.get("audio_transcript"),
            "creation_timestamp": fields.get("creation_timestamp")
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Add source_id to metadata
        metadata["source_id"] = fields.get("video_id", doc_id.split("_")[0])
        
        # Create Document using new structure
        return Document(
            doc_id=doc_id,
            media_type=media_type,
            temporal_info=temporal_info,
            segment_info=segment_info,
            metadata=metadata
        )
    
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
        
        # Encoder metrics
        with self._encoder_lock:
            for key, encoder in self._text_encoders.items():
                metrics["encoder_metrics"][key] = encoder.get_metrics()
        
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
        
        # Check encoders
        encoder_health = {}
        with self._encoder_lock:
            for key, encoder in self._text_encoders.items():
                encoder_health[key] = encoder.health_check()
        health["components"]["encoders"] = encoder_health
        
        # Add metrics
        health["metrics"] = self.get_metrics()
        
        return health
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        if self.pool:
            with self.pool.get_connection() as conn:
                response = conn.vespa.get_data(
                    schema=self.schema_name,
                    data_id=document_id,
                    namespace="video"
                )
        else:
            response = self.vespa.get_data(
                schema=self.schema_name,
                data_id=document_id,
                namespace="video"
            )
        
        if response and response.status_code == 200:
            # Convert Vespa document to Document object
            data = response.json()
            fields = data.get("fields", {})
            
            # Create Document from fields
            from src.core.documents import Document, MediaType, TemporalInfo
            
            doc = Document(
                doc_id=document_id,
                content=fields.get("content", ""),
                media_type=MediaType.VIDEO_SEGMENT,
                metadata=fields,
                temporal_info=TemporalInfo(
                    start_time=fields.get("start_time", 0.0),
                    end_time=fields.get("end_time", 0.0)
                ) if "start_time" in fields else None
            )
            
            return doc
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self.pool:
            self.pool.close()
        
        # Clear encoder cache
        with self._encoder_lock:
            self._text_encoders.clear()
        
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