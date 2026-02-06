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

import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from vespa.application import Vespa

from cogniverse_core.common.utils.output_manager import OutputManager
from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff
from cogniverse_sdk.document import (
    ContentType,
    Document,
    ProcessingStatus,
    SearchResult,
)
from cogniverse_sdk.interfaces.backend import SearchBackend

logger = logging.getLogger(__name__)

# Module-level cache for ranking strategies extracted from schemas
# This is populated once and shared across all VespaSearchBackend instances
_RANKING_STRATEGIES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CACHE_LOCK = threading.Lock()


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
        error: Optional[Exception] = None,
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
            result = self.vespa.query(yql="select * from sources * where true limit 1")
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
            target=self._health_check_loop, daemon=True
        )
        self._health_check_thread.start()

    def _initialize_connections(self):
        """Create initial connections"""
        for i in range(self.config.min_connections):
            conn = VespaConnection(self.url, f"conn-{uuid.uuid4().hex[:8]}")
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
                    conn = VespaConnection(self.url, f"conn-{uuid.uuid4().hex[:8]}")
                    self._connections.append(conn)
                    logger.info(f"Created new connection {conn.connection_id}")

            # Wait for connection if none available
            while (
                conn is None
                and (time.time() - start_time) < self.config.connection_timeout
            ):
                time.sleep(0.1)  # Poll for connection availability
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
        schema_name: str = None,  # DEPRECATED - schema determined at query time
        profile: str = None,  # DEPRECATED - profile determined at query time
        query_encoder: Optional[Any] = None,
        enable_metrics: bool = True,
        enable_connection_pool: bool = True,
        pool_config: Optional[ConnectionPoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        config: Optional[Dict[str, Any]] = None,  # NEW: Accept config dict
        config_manager=None,  # ConfigManager instance for dependency injection
        schema_loader=None,  # SchemaLoader instance for dependency injection
    ):
        """
        Initialize Vespa search backend with ALL profiles.
        Schema is determined at query time, not initialization time.

        Args:
            vespa_url: Vespa URL (or use config)
            vespa_port: Vespa port (or use config)
            schema_name: DEPRECATED - schema determined at query time from search() parameter
            profile: DEPRECATED - profile determined at query time from search() parameter
            query_encoder: Optional query encoder instance
            enable_metrics: Whether to collect metrics
            enable_connection_pool: Whether to use connection pooling
            pool_config: Connection pool configuration
            retry_config: Retry configuration
            config: Backend configuration dict (NEW - preferred way)
        """
        # Store config_manager and schema_loader
        self._config_manager = config_manager
        self._schema_loader = schema_loader

        # If config provided, extract from it (new approach)
        if config is not None:
            self.backend_url = config.get("url", "http://localhost")
            self.vespa_port = config.get("port", 8080)
            # tenant_id is REQUIRED - no fallback allowed
            self.tenant_id = config.get("tenant_id")
            if not self.tenant_id:
                logger.warning(
                    "VespaSearchBackend initialized WITHOUT tenant_id - search will fail"
                )
            # Store ALL profiles - schema determined at query time
            self.profiles = config.get("profiles", {})
            self.default_profiles = config.get("default_profiles", {})
            # No schema-specific initialization
            self.schema_name = None  # Will be set per-query
            self.profile = None  # Will be set per-query
            self.query_encoder = query_encoder or config.get("query_encoder")
        else:
            # Legacy initialization with individual parameters
            self.backend_url = vespa_url
            self.vespa_port = vespa_port
            self.schema_name = schema_name
            self.profile = profile
            self.query_encoder = query_encoder
            self.profiles = {}
            self.default_profiles = {}

        # Combine URL and port
        full_url = f"{self.backend_url}:{self.vespa_port}"

        # Initialize output manager
        self.output_manager = OutputManager()

        # Setup connection pool
        if enable_connection_pool:
            self.pool = ConnectionPool(full_url, pool_config or ConnectionPoolConfig())
        else:
            self.pool = None
            self.vespa = Vespa(url=self.backend_url, port=self.vespa_port)

        # Setup retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3, initial_delay=0.5, exceptions=(Exception,)
        )

        # Initialize metrics
        if enable_metrics:
            self.metrics = SearchMetrics()
        else:
            self.metrics = None

        logger.info(
            f"VespaSearchBackend.__init__: schema_name='{schema_name}' (query-time mode), "
            f"pool={enable_connection_pool}, metrics={enable_metrics}"
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize search backend from config.
        Called by backend registry after instantiation.

        Args:
            config: Configuration with keys url, port, schema_name, profile, tenant_id, config_manager
        """
        # Extract config values
        self.backend_url = config.get("url", "http://localhost")
        self.vespa_port = config.get("port", 8080)
        base_schema_name = config.get("schema_name")
        tenant_id = config.get("tenant_id")
        self.profile = config.get("profile")
        self.query_encoder = None
        self._config_manager = config.get("config_manager")

        # Transform schema name to tenant-scoped format if tenant_id provided
        if tenant_id and base_schema_name:
            # Generate tenant-specific schema name
            self.schema_name = f"{base_schema_name}_{tenant_id}"
            logger.info(
                f"Transformed schema name: {base_schema_name} → {self.schema_name} (tenant: {tenant_id})"
            )
        else:
            self.schema_name = base_schema_name

        # Combine URL and port
        full_url = f"{self.backend_url}:{self.vespa_port}"

        # Initialize output manager
        self.output_manager = OutputManager()

        # Setup connection pool
        self.pool = ConnectionPool(full_url, ConnectionPoolConfig())

        # Setup retry config
        self.retry_config = RetryConfig(
            max_attempts=3, initial_delay=0.5, exceptions=(Exception,)
        )

        # Initialize metrics
        self.metrics = SearchMetrics()

        logger.info(
            f"VespaSearchBackend initialized for schema '{self.schema_name}' "
            f"with pool=True, metrics=True (query-time mode)"
        )

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

        # Use schema's id field for matching (convention: all schemas should have id field)
        # TODO: Implement proper schema field mapping/registry system for text/id field conventions
        doc_id_conditions = " OR ".join(
            [f'id contains "{doc_id}"' for doc_id in document_ids]
        )
        yql = f"select * from {self.schema_name} where {doc_id_conditions}"

        query_params = {"yql": yql, "hits": len(document_ids), "timeout": "10s"}

        try:
            # Execute batch query
            if self.pool:
                with self.pool.get_connection() as conn:
                    response = conn.query(body=query_params)
            else:
                response = self.vespa.query(body=query_params)

            # Build results dictionary for fast lookup
            results_dict = {}
            if response and hasattr(response, "hits"):
                for hit in response.hits:
                    fields = hit.get("fields", {})
                    # Extract document ID from Vespa document id format
                    full_doc_id = hit.get("id", "")
                    doc_id = full_doc_id.split("::")[-1]

                    doc = Document(
                        id=doc_id,
                        content_type=ContentType.VIDEO,
                        text_content=fields.get("content", ""),
                        status=ProcessingStatus.COMPLETED,
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

    def _fallback_individual_get(
        self, document_ids: List[str]
    ) -> List[Optional[Document]]:
        """Fallback method for individual document retrieval when batch query fails."""
        results = []
        for doc_id in document_ids:
            try:
                if self.pool:
                    with self.pool.get_connection() as conn:
                        response = conn.vespa.get_data(
                            schema=self.schema_name, data_id=doc_id, namespace="video"
                        )
                else:
                    response = self.vespa.get_data(
                        schema=self.schema_name, data_id=doc_id, namespace="video"
                    )

                if response and response.status_code == 200:
                    data = response.json()
                    fields = data.get("fields", {})

                    doc = Document(
                        id=doc_id,
                        content_type=ContentType.VIDEO,
                        text_content=fields.get("content", ""),
                        status=ProcessingStatus.COMPLETED,
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

    def _embeddings_to_vespa_format(
        self, embeddings: np.ndarray, profile: str
    ) -> Dict[str, Any]:
        """Convert embeddings to Vespa query format."""
        # For binary embeddings, handle differently
        if "_binary" in profile:
            # For binary tensors, values should be int8
            if embeddings.ndim == 1:
                # 1D binary embeddings (global models)
                cells = [
                    {"address": {"v": str(i)}, "value": int(val)}
                    for i, val in enumerate(embeddings)
                ]
                return {"cells": cells}
            else:
                # 2D binary embeddings (patch-based models like ColPali)
                cells = []
                for patch_idx in range(embeddings.shape[0]):
                    for v_idx in range(embeddings.shape[1]):
                        cells.append(
                            {
                                "address": {
                                    "querytoken": str(patch_idx),
                                    "v": str(v_idx),
                                },
                                "value": int(embeddings[patch_idx, v_idx]),
                            }
                        )
                return {"cells": cells}
        # For single-vector profiles, embeddings are 1D
        elif "_sv_" in profile.lower():
            # Convert to tensor cells format for Vespa
            cells = [
                {"address": {"v": str(i)}, "value": float(val)}
                for i, val in enumerate(embeddings)
            ]
            return {"cells": cells}
        else:
            # For patch-based models, embeddings are 2D
            cells = []
            for patch_idx in range(embeddings.shape[0]):
                for v_idx in range(embeddings.shape[1]):
                    cells.append(
                        {
                            "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                            "value": float(embeddings[patch_idx, v_idx]),
                        }
                    )
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
                binary = np.pad(binary, (0, padding), mode="constant")
            # Pack bits
            packed = np.packbits(binary).astype(np.int8)
        else:
            # 2D embeddings (patch-based)
            # Pack each patch separately
            packed = np.packbits(binary, axis=1).astype(np.int8)

        return packed

    @retry_with_backoff
    def search(self, query_dict: Dict[str, Any]) -> List[SearchResult]:
        """
        Search for documents using query dict format.

        Args:
            query_dict: Dictionary with keys:
                - query: Text query string (required)
                - type: Content type (e.g., "video") (required)
                - profile: Profile name (optional)
                - strategy: Strategy name (optional)
                - top_k: Number of results (optional, defaults to 10)
                - filters: Optional filters dict
                - query_embeddings: Pre-computed embeddings (optional)

        Returns:
            List of SearchResult objects

        Profile Resolution Logic:
            1. If profile in query_dict → use it
            2. Else if only 1 profile for type → auto-select
            3. Else if default_profiles[type] exists → use it
            4. Else → raise ValueError

        Strategy Resolution: Same logic as profile
        """
        correlation_id = str(uuid.uuid4())

        # Extract parameters from query_dict
        query_text = query_dict.get("query")
        if not query_text:
            raise ValueError("query_dict must contain 'query' key with text query")

        content_type = query_dict.get("type")
        if not content_type:
            raise ValueError("query_dict must contain 'type' key (e.g., 'video')")

        top_k = query_dict.get("top_k", 10)
        filters = query_dict.get("filters", {})  # Get filters from query_dict
        query_embeddings = query_dict.get("query_embeddings")

        # Phase 1: Profile Resolution
        requested_profile = query_dict.get("profile")

        if requested_profile:
            # 1. Use explicitly requested profile
            if requested_profile not in self.profiles:
                raise ValueError(
                    f"Requested profile '{requested_profile}' not found. "
                    f"Available profiles: {list(self.profiles.keys())}"
                )
            profile_name = requested_profile
            logger.info(f"[{correlation_id}] Using requested profile: {profile_name}")
        else:
            # 2. Auto-select based on type
            # Get all profiles for this type
            type_profiles = {
                name: config
                for name, config in self.profiles.items()
                if config.get("type") == content_type
            }

            if len(type_profiles) == 1:
                # Only one profile for this type - auto-select
                profile_name = list(type_profiles.keys())[0]
                logger.info(
                    f"[{correlation_id}] Auto-selected single profile for type '{content_type}': {profile_name}"
                )
            elif len(type_profiles) > 1:
                # 3. Check default_profiles
                default_config = self.default_profiles.get(content_type, {})
                profile_name = default_config.get("profile")

                if not profile_name:
                    raise ValueError(
                        f"Multiple profiles available for type '{content_type}' but no default configured. "
                        f"Available profiles: {list(type_profiles.keys())}. "
                        f"Either specify 'profile' in query_dict or configure backend.default_profiles.{content_type}.profile"
                    )

                if profile_name not in type_profiles:
                    raise ValueError(
                        f"Default profile '{profile_name}' for type '{content_type}' not found in available profiles: "
                        f"{list(type_profiles.keys())}"
                    )

                logger.info(
                    f"[{correlation_id}] Using default profile for type '{content_type}': {profile_name}"
                )
            else:
                # 4. No profiles for this type
                raise ValueError(
                    f"No profiles found for type '{content_type}'. "
                    f"Available types: {set(p.get('type') for p in self.profiles.values())}"
                )

        # Get profile config
        profile_config = self.profiles[profile_name]

        # Determine schema_name from profile (base name)
        base_schema_name = profile_config.get("schema_name", profile_name)

        # Apply tenant scoping - tenant_id is REQUIRED
        if not self.tenant_id:
            raise ValueError(
                f"tenant_id is required for search operations. "
                f"Profile '{profile_name}' cannot be used without tenant isolation."
            )

        # Generate tenant-specific schema name
        schema_name = f"{base_schema_name}_{self.tenant_id}"
        logger.info(
            f"[{correlation_id}] Applied tenant scoping: {base_schema_name} → {schema_name}"
        )

        # Load ranking strategies from internal cache (not from Strategy object)
        global _RANKING_STRATEGIES_CACHE
        with _CACHE_LOCK:
            if _RANKING_STRATEGIES_CACHE is None:
                _RANKING_STRATEGIES_CACHE = self._load_ranking_strategies()

        available_strategies = _RANKING_STRATEGIES_CACHE.get(base_schema_name, {})
        if not available_strategies:
            raise ValueError(
                f"No ranking strategies found for schema '{base_schema_name}'. "
                f"Available schemas: {list(_RANKING_STRATEGIES_CACHE.keys())}"
            )

        # Phase 2: Strategy Resolution
        requested_strategy = query_dict.get("strategy")

        if requested_strategy:
            # 1. Use explicitly requested strategy
            if requested_strategy not in available_strategies:
                raise ValueError(
                    f"Requested strategy '{requested_strategy}' not found in profile '{profile_name}'. "
                    f"Available strategies: {list(available_strategies.keys())}"
                )
            strategy_name = requested_strategy
            logger.info(f"[{correlation_id}] Using requested strategy: {strategy_name}")
        else:
            # 2. Auto-select based on profile
            if len(available_strategies) == 1:
                # Only one strategy - auto-select
                strategy_name = list(available_strategies.keys())[0]
                logger.info(
                    f"[{correlation_id}] Auto-selected single strategy: {strategy_name}"
                )
            elif len(available_strategies) > 1:
                # 3. Check default_profiles for strategy
                default_config = self.default_profiles.get(content_type, {})
                strategy_name = default_config.get("strategy")

                if not strategy_name:
                    # Fall back to profile's default_ranking
                    strategy_name = profile_config.get("default_ranking")

                if not strategy_name or strategy_name not in available_strategies:
                    raise ValueError(
                        f"Multiple strategies available for profile '{profile_name}' but no default configured. "
                        f"Available strategies: {list(available_strategies.keys())}. "
                        f"Either specify 'strategy' in query_dict or configure backend.default_profiles.{content_type}.strategy"
                    )

                logger.info(
                    f"[{correlation_id}] Using default strategy: {strategy_name}"
                )
            else:
                raise ValueError(f"No strategies found in profile '{profile_name}'")

        # Update instance state for this query
        self.schema_name = schema_name
        self.profile = profile_name
        # No longer need strategy object - we use internal ranking strategies

        logger.info(
            f"[{correlation_id}] Query resolved: type={content_type}, profile={profile_name}, "
            f"schema={schema_name}, strategy={strategy_name}"
        )

        # Continue with search execution
        start_time = time.time()

        try:
            # Log search request
            logger.info(
                f"[{correlation_id}] Search request: query='{query_text}', "
                f"limit={top_k}, profile={profile_name}, strategy={strategy_name}"
            )

            # Get ranking strategy config from available_strategies
            rank_config = available_strategies[strategy_name]

            # Check if strategy requires embeddings
            requires_embeddings = rank_config.get(
                "needs_float_embeddings", False
            ) or rank_config.get("needs_binary_embeddings", False)

            # Generate embeddings on-demand if needed and not provided
            if requires_embeddings and query_embeddings is None:
                if self.query_encoder:
                    logger.info(
                        f"[{correlation_id}] Generating embeddings on-demand for strategy '{strategy_name}'"
                    )
                    logger.info(
                        f"[{correlation_id}] Query encoder type: {type(self.query_encoder).__name__}"
                    )
                    logger.info(f"[{correlation_id}] Query text: '{query_text}'")
                    query_embeddings = self.query_encoder.encode(query_text)
                    logger.info(
                        f"[{correlation_id}] Generated embeddings shape: {query_embeddings.shape}"
                    )
                    logger.info(
                        f"[{correlation_id}] Embeddings dtype: {query_embeddings.dtype}"
                    )
                    logger.info(
                        f"[{correlation_id}] Embeddings min/max: {query_embeddings.min():.4f}/{query_embeddings.max():.4f}"
                    )
                    logger.info(
                        f"[{correlation_id}] First 5 values: {query_embeddings.flatten()[:5]}"
                    )
                else:
                    logger.warning(
                        f"[{correlation_id}] Strategy '{strategy_name}' requires embeddings but no encoder available"
                    )

            # Build query based on strategy
            # Use tenant-scoped schema_name for Vespa query (schemas are actually deployed per-tenant)
            query_params = self._build_query(
                query_text,
                query_embeddings,
                rank_config,
                strategy_name,
                schema_name,
                top_k,
                filters,
                correlation_id,
            )

            # Execute search
            logger.info(
                f"[{correlation_id}] Executing query: yql='{query_params.get('yql')}', "
                f"ranking={query_params.get('ranking')}"
            )
            logger.debug(
                f"[{correlation_id}] Query has embeddings: {'input.query(q)' in query_params or 'input.query(qt)' in query_params}"
            )

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
                self.metrics.record_search(True, latency_ms, strategy_name)

            logger.info(
                f"[{correlation_id}] Search completed: {len(results)} results "
                f"in {(time.time() - start_time)*1000:.2f}ms"
            )

            return results

        except Exception as e:
            # Record failure metrics
            if self.metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_search(
                    False, latency_ms, strategy_name or "default", e
                )

            logger.error(f"[{correlation_id}] Search failed: {e}")
            raise

    def _build_filter_conditions(self, filters: Dict[str, Any]) -> str:
        """Build Vespa YQL filter conditions from filters dict.

        Args:
            filters: Dict of field_name -> value pairs

        Returns:
            Filter string like 'user_id contains "test" AND agent_id contains "agent1"'
            Empty string if no filters
        """
        if not filters:
            return ""

        conditions = []
        for field_name, value in filters.items():
            if isinstance(value, str):
                # String attributes use 'contains' for matching
                # This works for string attributes in Vespa
                conditions.append(f'{field_name} contains "{value}"')
            elif isinstance(value, (int, float)):
                # Numeric values use equality
                conditions.append(f"{field_name} = {value}")
            elif isinstance(value, bool):
                # Boolean values
                conditions.append(f"{field_name} = {str(value).lower()}")
            else:
                # For other types, convert to string and use contains
                conditions.append(f'{field_name} contains "{str(value)}"')

        return " AND ".join(conditions)

    def _build_query(
        self,
        query_text: str,
        query_embeddings: Optional[np.ndarray],
        rank_config: Dict[str, Any],
        ranking_profile: str,
        schema_name: str,
        limit: int,
        filters: Dict[str, Any],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Build Vespa query based on ranking strategy - NO HARDCODING!"""

        # Log schema name being used
        logger.info(
            f"[{correlation_id}] Building query with schema_name='{schema_name}'"
        )
        logger.info(f"[{correlation_id}] Ranking profile: '{ranking_profile}'")

        # Initialize query params
        query_params = {
            "hits": limit,
            "ranking": ranking_profile,
        }

        # Build filter conditions
        filter_conditions = self._build_filter_conditions(filters)
        if filter_conditions:
            logger.info(f"[{correlation_id}] Applying filters: {filter_conditions}")

        # Build YQL based on strategy configuration
        if rank_config.get("use_nearestneighbor"):
            # Use nearestNeighbor for visual search
            nn_field = rank_config.get("nearestneighbor_field", "embedding")
            nn_tensor = rank_config.get("nearestneighbor_tensor", "qt")

            if rank_config.get("needs_text_query") and query_text:
                # Hybrid search with nearestNeighbor
                base_where = (
                    f"userInput(@userQuery) OR "
                    f"({{targetHits: {limit}}}nearestNeighbor({nn_field}, {nn_tensor}))"
                )
                if filter_conditions:
                    query_params["yql"] = (
                        f"select * from {schema_name} where ({base_where}) AND {filter_conditions}"
                    )
                else:
                    query_params["yql"] = (
                        f"select * from {schema_name} where {base_where}"
                    )
                query_params["userQuery"] = query_text
            else:
                # Pure semantic/visual search with nearestNeighbor
                base_where = (
                    f"{{targetHits: {limit}}}nearestNeighbor({nn_field}, {nn_tensor})"
                )
                if filter_conditions:
                    query_params["yql"] = (
                        f"select * from {schema_name} where {base_where} AND {filter_conditions}"
                    )
                else:
                    query_params["yql"] = (
                        f"select * from {schema_name} where {base_where}"
                    )
        elif rank_config.get("needs_text_query"):
            # Text or hybrid search without nearestNeighbor
            if filter_conditions:
                query_params["yql"] = (
                    f"select * from {schema_name} where userInput(@userQuery) AND {filter_conditions}"
                )
            else:
                query_params["yql"] = (
                    f"select * from {schema_name} where userInput(@userQuery)"
                )
            query_params["userQuery"] = query_text
        else:
            # Regular ranking without nearestNeighbor (patch-based models)
            if filter_conditions:
                query_params["yql"] = (
                    f"select * from {schema_name} where {filter_conditions}"
                )
            else:
                query_params["yql"] = f"select * from {schema_name} where true"

        # Add tensor embeddings based on what the strategy says it needs
        # NO MORE HARDCODED CHECKS!
        if query_embeddings is not None:
            # Check what tensors this ranking strategy needs from its inputs
            inputs_needed = rank_config.get("inputs", {})

            logger.error(
                f"[{correlation_id}] Strategy '{ranking_profile}' needs inputs: {list(inputs_needed.keys())}"
            )
            logger.error(
                f"[{correlation_id}] Input query_embeddings shape: {query_embeddings.shape}, dtype: {query_embeddings.dtype}"
            )
            if query_embeddings.ndim == 2:
                logger.error(
                    f"[{correlation_id}] Input query_embeddings (2D) first vector (first 5): {query_embeddings[0][:5].tolist()}"
                )
            else:
                logger.error(
                    f"[{correlation_id}] Input query_embeddings (1D) first 5 values: {query_embeddings[:5].tolist()}"
                )

            for input_name, input_type in inputs_needed.items():
                # The input names are like 'qt', 'qtb', etc. We need to construct the full Vespa query parameter name
                vespa_param_name = f"input.query({input_name})"

                if input_name == "qt" and "float" in input_type:
                    # Float embeddings needed
                    logger.error(
                        f"[{correlation_id}] Adding float embeddings for {vespa_param_name}"
                    )
                    # For multi-vector models, convert to dict format as per Vespa docs
                    if query_embeddings.ndim == 2:
                        query_params[vespa_param_name] = {
                            index: vector.tolist()
                            for index, vector in enumerate(query_embeddings)
                        }
                    else:
                        query_params[vespa_param_name] = query_embeddings.tolist()

                elif input_name == "qtb" and "int8" in input_type:
                    # Binary embeddings needed
                    logger.error(
                        f"[{correlation_id}] Adding binary embeddings for {vespa_param_name}"
                    )
                    binary_embeddings = self._generate_binary_embeddings(
                        query_embeddings
                    )
                    logger.error(
                        f"[{correlation_id}] Binary embeddings shape: {binary_embeddings.shape}, dtype: {binary_embeddings.dtype}"
                    )
                    # For multi-vector models, convert to dict format as per Vespa docs
                    if binary_embeddings.ndim == 2:
                        query_params[vespa_param_name] = {
                            index: vector.tolist()
                            for index, vector in enumerate(binary_embeddings)
                        }
                        logger.error(
                            f"[{correlation_id}] Binary embeddings (2D) first vector (first 5): {binary_embeddings[0][:5].tolist()}"
                        )
                    else:
                        query_params[vespa_param_name] = binary_embeddings.tolist()
                        logger.error(
                            f"[{correlation_id}] Binary embeddings (1D) first 5 values: {binary_embeddings[:5].tolist()}"
                        )

                elif input_name == "q":
                    # Generic query tensor (used by some schemas)
                    logger.error(
                        f"[{correlation_id}] Adding generic embeddings for {vespa_param_name}"
                    )
                    query_params[vespa_param_name] = query_embeddings.tolist()

                else:
                    # Unknown input name - this shouldn't happen with proper configuration
                    logger.warning(
                        f"[{correlation_id}] Unknown input name '{input_name}' with type '{input_type}' "
                        f"in ranking strategy '{ranking_profile}'. This input will be skipped. "
                        f"Known input names: 'qt' (float), 'qtb' (binary), 'q' (generic)."
                    )

        # Add schema to query body to avoid conflicts with other schemas
        query_params["model.restrict"] = schema_name  # Must be string, not list!
        logger.info(
            f"[{correlation_id}] Set model.restrict='{query_params['model.restrict']}'"
        )

        # Log the query parameters for debugging
        logger.info(
            f"[{correlation_id}] Query params keys: {list(query_params.keys())}"
        )
        if "input.query(qt)" in query_params:
            qt_value = query_params["input.query(qt)"]
            logger.info(
                f"[{correlation_id}] input.query(qt) type: {type(qt_value)}, is list: {isinstance(qt_value, list)}"
            )
            if isinstance(qt_value, list):
                logger.info(
                    f"[{correlation_id}] input.query(qt) length: {len(qt_value)}, first 5 values: {qt_value[:5] if qt_value else 'empty'}..."
                )
        if "input.query(qtb)" in query_params:
            qtb_value = query_params["input.query(qtb)"]
            logger.info(
                f"[{correlation_id}] input.query(qtb) type: {type(qtb_value)}, is list: {isinstance(qtb_value, list)}"
            )
            if isinstance(qtb_value, list):
                logger.info(
                    f"[{correlation_id}] input.query(qtb) length: {len(qtb_value)}, first 5 values: {qtb_value[:5] if qtb_value else 'empty'}..."
                )

        return query_params

    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert Vespa result to Document object."""
        fields = result.get("fields", {})

        # Extract document ID
        doc_id = result.get("id", "").split("::")[-1]

        # Create Document using new generic structure
        document = Document(
            id=doc_id, content_type=ContentType.VIDEO, status=ProcessingStatus.COMPLETED
        )

        # Map Vespa text fields to Document.text_content
        # Try common text field names in order of priority
        for text_field_name in ["text", "transcription", "text_content", "content"]:
            if text_field_name in fields and fields[text_field_name]:
                document.text_content = fields[text_field_name]
                break

        # Add all fields as metadata
        for key, value in fields.items():
            if value is not None:
                document.add_metadata(key, value)

        # Add source_id to metadata
        document.add_metadata("source_id", fields.get("video_id", doc_id.split("_")[0]))

        return document

    def _process_results(
        self, response: Any, correlation_id: str
    ) -> List[SearchResult]:
        """Process Vespa response into SearchResult objects"""
        results = []

        if not response or not hasattr(response, "hits"):
            logger.warning(f"[{correlation_id}] Empty response from Vespa")
            return results

        logger.debug(
            f"[{correlation_id}] Processing {len(response.hits)} hits from Vespa"
        )
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
        for key in [
            "video_id",
            "start_time",
            "end_time",
            "frame_number",
            "description",
            "transcription",
        ]:
            if key in fields:
                metadata[key] = fields[key]

        # Handle temporal info
        if "start_time" in fields and "end_time" in fields:
            metadata["duration"] = fields["end_time"] - fields["start_time"]

        return metadata

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {"search_metrics": {}, "encoder_metrics": {}, "pool_metrics": {}}

        # Search metrics
        if self.metrics:
            metrics["search_metrics"] = {
                "total_searches": self.metrics.total_searches,
                "success_rate": self.metrics.success_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "strategy_usage": dict(self.metrics.strategy_usage),
                "error_types": dict(self.metrics.error_types),
            }

        # Connection pool metrics
        if self.pool:
            with self.pool._lock:
                metrics["pool_metrics"] = {
                    "total_connections": len(self.pool._connections),
                    "available_connections": len(self.pool._available),
                    "healthy_connections": sum(
                        1 for c in self.pool._connections if c.is_healthy
                    ),
                }

        return metrics

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "schema": self.schema_name,
            "components": {},
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
        include_embeddings: bool = True,
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
        visit_url = f"{self.backend_url}:{self.vespa_port}/document/v1/{namespace}/{schema or self.schema_name}/docid"

        try:
            import requests

            response = requests.get(
                visit_url,
                params={
                    "selection": "true" if not filters else None,
                    "continuation": None,
                    "wantedDocumentCount": max_documents or 1000,
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()

                for doc in data.get("documents", []):
                    fields = doc.get("fields", {})
                    doc_data = {"id": doc.get("id", ""), **fields}

                    # Extract embeddings if present and requested
                    if include_embeddings:
                        for emb_field in [
                            "embedding",
                            "frame_embedding",
                            "video_embedding",
                            "text_embedding",
                            "colpali_embedding",
                        ]:
                            if emb_field in fields:
                                doc_data[emb_field] = fields[emb_field]

                    documents.append(doc_data)

                # Handle continuation token for large datasets
                continuation = data.get("continuation")
                while continuation and len(documents) < (max_documents or float("inf")):
                    response = requests.get(
                        visit_url,
                        params={
                            "continuation": continuation,
                            "wantedDocumentCount": min(
                                1000, (max_documents or 1000) - len(documents)
                            ),
                        },
                        timeout=30,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for doc in data.get("documents", []):
                            fields = doc.get("fields", {})
                            doc_data = {"id": doc.get("id", ""), **fields}

                            if include_embeddings:
                                for emb_field in [
                                    "embedding",
                                    "frame_embedding",
                                    "video_embedding",
                                    "text_embedding",
                                    "colpali_embedding",
                                ]:
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

    def get_embedding_requirements(self, schema_name: str) -> Dict[str, Any]:
        """
        Get embedding requirements for a specific schema.

        This method allows the backend to specify what types of embeddings
        it needs for ingestion, based on its internal schema configuration
        (e.g., rank-profiles in Vespa).

        Args:
            schema_name: Name of schema to get requirements for (base name, not tenant-scoped)

        Returns:
            Dict containing:
                - needs_float: bool - whether float embeddings are needed
                - needs_binary: bool - whether binary embeddings are needed
                - float_field: str - name of float embedding field
                - binary_field: str - name of binary embedding field

        Note:
            This is backend-specific metadata that should NOT be exposed
            to application code. Only used internally by ingestion pipeline.
        """
        global _RANKING_STRATEGIES_CACHE

        # Load ranking strategies from schemas if not cached
        with _CACHE_LOCK:
            if _RANKING_STRATEGIES_CACHE is None:
                _RANKING_STRATEGIES_CACHE = self._load_ranking_strategies()

        # Get strategies for this schema
        schema_strategies = _RANKING_STRATEGIES_CACHE.get(schema_name, {})

        if not schema_strategies:
            logger.warning(
                f"No ranking strategies found for schema '{schema_name}'. "
                f"Available schemas: {list(_RANKING_STRATEGIES_CACHE.keys())}"
            )
            # Return defaults - assume float embeddings needed
            return {
                "needs_float": True,
                "needs_binary": False,
                "float_field": "embedding",
                "binary_field": "embedding_binary",
            }

        # Analyze what embeddings are needed across all strategies
        needs_float = False
        needs_binary = False
        float_fields = set()
        binary_fields = set()

        for strategy_name, strategy_info in schema_strategies.items():
            if strategy_info.get("needs_float_embeddings", False):
                needs_float = True
                field = strategy_info.get("embedding_field", "")
                if field and "binary" not in field:
                    float_fields.add(field)

            if strategy_info.get("needs_binary_embeddings", False):
                needs_binary = True
                field = strategy_info.get("embedding_field", "")
                if field and "binary" in field:
                    binary_fields.add(field)

        return {
            "needs_float": needs_float,
            "needs_binary": needs_binary,
            "float_field": next(iter(float_fields), "embedding"),
            "binary_field": next(iter(binary_fields), "embedding_binary"),
        }

    def _load_ranking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ranking strategies from Vespa schema files using injected schema_loader.

        This is an internal method that extracts ranking profile configurations
        from schema JSON files in the schemas directory.

        Returns:
            Dict mapping schema names to their ranking strategies (serialized to dicts)
        """
        from cogniverse_vespa.ranking_strategy_extractor import (
            extract_all_ranking_strategies,
        )

        # Use schema_loader to get schemas directory
        if self._schema_loader is None:
            raise ValueError("schema_loader is required for loading ranking strategies")

        # Get schemas directory from schema_loader
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        if not isinstance(self._schema_loader, FilesystemSchemaLoader):
            logger.error(f"Unsupported schema_loader type: {type(self._schema_loader)}")
            return {}

        schemas_dir = self._schema_loader.base_path

        if not schemas_dir.exists():
            logger.error(f"Schemas directory not found: {schemas_dir}")
            return {}

        logger.info(f"Loading ranking strategies from schemas directory: {schemas_dir}")

        try:
            # Extract strategies (returns Dict[str, Dict[str, RankingStrategyInfo]])
            strategies_raw = extract_all_ranking_strategies(schemas_dir)

            # Convert RankingStrategyInfo dataclasses to dicts for easier access
            strategies = {}
            for schema_name, schema_strategies in strategies_raw.items():
                strategies[schema_name] = {}
                for strategy_name, strategy_info in schema_strategies.items():
                    # Convert dataclass to dict
                    strategies[schema_name][strategy_name] = {
                        "name": strategy_info.name,
                        "strategy_type": strategy_info.strategy_type.value,
                        "needs_float_embeddings": strategy_info.needs_float_embeddings,
                        "needs_binary_embeddings": strategy_info.needs_binary_embeddings,
                        "needs_text_query": strategy_info.needs_text_query,
                        "use_nearestneighbor": strategy_info.use_nearestneighbor,
                        "nearestneighbor_field": strategy_info.nearestneighbor_field,
                        "nearestneighbor_tensor": strategy_info.nearestneighbor_tensor,
                        "embedding_field": strategy_info.embedding_field,
                        "query_tensor_name": strategy_info.query_tensor_name,
                        "timeout": strategy_info.timeout,
                        "description": strategy_info.description,
                        "inputs": strategy_info.inputs,
                        "query_tensors_needed": strategy_info.query_tensors_needed,
                        "schema_name": strategy_info.schema_name,
                    }

            logger.info(f"Loaded ranking strategies for {len(strategies)} schemas")
            return strategies
        except Exception as e:
            logger.error(f"Failed to load ranking strategies: {e}")
            return {}


# Factory function for creating search backend
def create_vespa_search_backend(
    schema_name: str, vespa_url: str = "http://localhost:8080", **kwargs
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
    return VespaSearchBackend(vespa_url=vespa_url, schema_name=schema_name, **kwargs)
