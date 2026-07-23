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
import math
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import requests
from vespa.exceptions import VespaError

from cogniverse_core.common.utils.output_manager import OutputManager
from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff
from cogniverse_sdk.document import (
    ContentType,
    Document,
    ProcessingStatus,
    SearchResult,
)
from cogniverse_sdk.interfaces.backend import SearchBackend
from cogniverse_vespa._vespa_factory import apply_failfast_timeouts, make_vespa_app
from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.embedding_processor import _is_single_vector_schema

logger = logging.getLogger(__name__)

# Transient search failures: retried and counted by the circuit breaker.
# VespaError is a bare Exception (not a requests.RequestException) — pyvespa
# raises it for 4xx/5xx error bodies, and _process_results raises it for
# HTTP-200 soft timeouts (root.errors); excluding it left the most common
# Vespa error shape unretried and invisible to the breaker.
_TRANSIENT_SEARCH_ERRORS = (
    requests.RequestException,
    ConnectionError,
    TimeoutError,
    VespaError,
)


def _format_query_vector_param(arr: np.ndarray, schema_name: str):
    """Format a query embedding for a Vespa query-tensor input.

    Single-vector schemas bind a dense ``tensor(v[dim])``; multi-vector schemas
    bind a ``{token: vector}`` mapping. A ``(1, dim)`` array is a single vector
    either way, so flatten it for the dense case and only dict-ify a genuine
    multi-row input.
    """
    if _is_single_vector_schema(schema_name):
        if arr.ndim == 2:
            if arr.shape[0] == 0:
                raise ValueError(
                    f"Single-vector schema '{schema_name}' received an empty "
                    "query embedding (no vectors)."
                )
            if arr.shape[0] > 1:
                raise ValueError(
                    f"Single-vector schema '{schema_name}' received "
                    f"{arr.shape[0]} query vectors but binds exactly one. "
                    "Refusing to silently drop rows."
                )
            arr = arr[0]
        return arr.tolist()
    if arr.ndim == 2:
        return {str(i): v.tolist() for i, v in enumerate(arr)}
    return arr.tolist()


def _coerce_numpy_scalar(value: object) -> object:
    """Convert a numpy scalar to its native Python equivalent.

    ``np.float64`` is a ``float`` subclass, so ``repr`` on it emits
    ``np.float64(0.5)`` — malformed YQL. ``np.int64``/``np.bool_`` are not
    ``int``/``bool`` subclasses, so they miss the numeric branch entirely and
    fall through to ``contains "5"``. ``.item()`` yields a plain Python scalar
    that serializes correctly. Non-numpy values pass through unchanged.
    """
    return value.item() if isinstance(value, np.generic) else value


_SAFE_FILTER_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _validate_filter_field_name(field_name: object) -> str:
    """Reject filter keys that aren't plain Vespa field identifiers.

    Filter values are escaped via yql_quote/_yql_scalar, but the field NAME is
    interpolated raw into the YQL where-clause. A caller-supplied key carrying
    YQL syntax (spaces, parens, quotes, operators) breaks out of the
    AND-grouped filter and injects arbitrary predicates.
    """
    if not isinstance(field_name, str) or not _SAFE_FILTER_FIELD.match(field_name):
        raise ValueError(
            f"Invalid filter field name {field_name!r}: filter keys must be "
            "Vespa field identifiers ([A-Za-z_][A-Za-z0-9_]*)"
        )
    return field_name


def _yql_scalar(value: object, field: str) -> str:
    """Serialize a scalar filter value into a YQL-safe literal.

    Used for both equality and range bounds. Numerics must be finite — NaN
    and Inf produce malformed YQL (``field >= nan``) that Vespa rejects.
    Strings are quoted via ``yql_quote`` so an ISO timestamp like
    ``"2024-01-01"`` (a string that looks numeric) survives the parser.
    """
    value = _coerce_numpy_scalar(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"Non-finite numeric filter value {value!r} for field {field!r}"
            )
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return yql_quote(value)
    return yql_quote(str(value))


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


# Sliding window size for the p95 latency percentile (bounds memory on a
# process-lifetime-cached backend).
_LATENCY_WINDOW = 1000


@dataclass
class SearchMetrics:
    """Comprehensive search metrics"""

    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    total_latency_ms: float = 0
    # Bounded window for the p95 percentile. VespaSearchBackend instances are
    # cached for the process lifetime, so an unbounded list leaked one float
    # per query forever. The lifetime average uses total_latency_ms /
    # total_searches (below), not this window.
    search_latencies: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_LATENCY_WINDOW)
    )
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
        """Lifetime average latency (from running totals, not the p95 window)."""
        if self.total_searches == 0:
            return 0.0
        return self.total_latency_ms / self.total_searches

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency"""
        if not self.search_latencies:
            return 0.0
        sorted_latencies = sorted(self.search_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class VespaConnection:
    """Managed Vespa connection with health checking.

    Queries run over a persistent ``VespaSync`` HTTP client so TCP
    connections are reused across searches — ``Vespa.query()`` itself
    builds and tears down a fresh client per call, which made the pool
    reuse app objects but never sockets. The pool hands a connection to
    one searcher at a time, so the client sees no concurrent use;
    ``health_check`` (which runs from the pool's background thread,
    possibly while the connection is checked out) deliberately stays on
    the per-call client path.
    """

    def __init__(self, url: str, connection_id: str):
        self.connection_id = connection_id
        self.vespa = make_vespa_app(url=url)
        self._sync = self.vespa.syncio(connections=4)
        self._sync._open_http_client()
        # Same fail-fast clamp as the document/config sessions — without it
        # the user-facing search path kept pyvespa's 120s x 11-attempt
        # defaults, so a hung Vespa blocked each query for minutes.
        apply_failfast_timeouts(self._sync)
        self.last_used = time.time()
        self.is_healthy = True
        self._lock = threading.Lock()

    def query(self, *args, **kwargs):
        """Execute query and update last used time"""
        with self._lock:
            self.last_used = time.time()
        return self._sync.query(*args, **kwargs)

    def close(self) -> None:
        """Release the persistent HTTP client."""
        try:
            self._sync._close_http_client()
        except Exception as exc:
            logger.debug(f"Closing {self.connection_id} client failed: {exc}")

    def health_check(self) -> bool:
        """Check connection health"""
        try:
            # Probe through the fail-fast-clamped session, not the raw
            # ``self.vespa`` (pyvespa's 120s x 11-attempt default) — a hung
            # Vespa would otherwise block the reaper's health sweep for minutes,
            # delaying detection and healing of the very connection it checks.
            result = self._sync.query(yql="select * from sources * where true limit 1")
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
        # Signalled whenever a connection returns to the pool, so waiters
        # wake immediately instead of polling on a sleep loop.
        self._returned = threading.Condition(self._lock)
        self._stop_health_check = threading.Event()

        self._initialize_connections()

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
        deadline = time.monotonic() + self.config.connection_timeout

        try:
            with self._returned:
                while conn is None:
                    if self._available:
                        conn = self._available.pop()
                    elif len(self._connections) < self.config.max_connections:
                        # Create new connection if under limit
                        conn = VespaConnection(self.url, f"conn-{uuid.uuid4().hex[:8]}")
                        self._connections.append(conn)
                        logger.info(f"Created new connection {conn.connection_id}")
                    else:
                        # Block until a connection returns — the previous
                        # 100ms sleep-poll added up to a tick of latency per
                        # starved query and burned CPU while waiting.
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not self._returned.wait(remaining):
                            if not self._available:
                                raise TimeoutError("No connections available")

            yield conn

        finally:
            # Return connection to pool
            if conn is not None:
                with self._returned:
                    self._available.append(conn)
                    self._returned.notify()

    def _health_check_loop(self):
        """Periodic health check for all connections"""
        while not self._stop_health_check.is_set():
            try:
                unhealthy = []

                # Snapshot under the lock, probe outside it — health checks
                # are network calls, and holding the pool lock through the
                # sweep blocked every get_connection for its duration.
                with self._lock:
                    snapshot = list(self._connections)

                for conn in snapshot:
                    if not conn.health_check():
                        unhealthy.append(conn)
                    elif conn.idle_time > self.config.idle_timeout:
                        # Remove idle connections above minimum
                        with self._lock:
                            above_min = (
                                len(self._connections) > self.config.min_connections
                            )
                        if above_min:
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
        conn.close()
        logger.info(f"Removed connection {conn.connection_id}")

    def close(self):
        """Close all connections and stop health checks"""
        self._stop_health_check.set()
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)

        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            self._available.clear()


class _DenseQueryEncoder:
    """Adapts a SemanticEmbedder to the ``.encode(text) -> ndarray`` shape the
    search path calls for on-demand query encoding of dense profiles."""

    def __init__(self, embedder):
        self._embedder = embedder

    def encode(self, query_text: str):
        return self._embedder.encode(query_text, is_query=True)


class VespaSearchBackend(SearchBackend):
    """Production-ready Vespa search backend"""

    def __init__(
        self,
        backend_url: str = None,
        backend_port: int = None,
        schema_name: str = None,
        query_encoder: Optional[Any] = None,
        enable_metrics: bool = True,
        enable_connection_pool: bool = True,
        pool_config: Optional[ConnectionPoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        config: Optional[Dict[str, Any]] = None,
        config_manager=None,
        schema_loader=None,
    ):
        """
        Initialize Vespa search backend.

        When config is provided, backend_url/backend_port/schema_name are ignored.
        The profile is resolved at query time from search() parameters.

        Args:
            backend_url: Backend URL (used when config is None)
            backend_port: Backend port (used when config is None)
            schema_name: Schema name (used when config is None; set per-query otherwise)
            query_encoder: Optional query encoder instance
            enable_metrics: Whether to collect metrics
            enable_connection_pool: Whether to use connection pooling
            pool_config: Connection pool configuration
            retry_config: Retry configuration
            config: Backend configuration dict (preferred; takes precedence)
            config_manager: ConfigManager for dependency injection
            schema_loader: SchemaLoader for dependency injection
        """
        self._schema_loader = schema_loader
        self._config_manager = config_manager

        # Lock guards runtime mutation of self.profiles / self.default_profiles
        # via add_profile / remove_profile. Reads in get_search_results take the
        # same lock so a concurrent add can't produce a torn read.
        self._profiles_lock = threading.RLock()

        if config is not None:
            self.backend_url = config.get("url", "http://localhost")
            self.backend_port = config.get("port", 8080)
            self.profiles = dict(config.get("profiles", {}))
            self.default_profiles = dict(config.get("default_profiles", {}))
            self.schema_name = None
            self.query_encoder = query_encoder or config.get("query_encoder")
        else:
            self.backend_url = backend_url
            self.backend_port = backend_port
            self.schema_name = schema_name
            self.query_encoder = query_encoder
            self.profiles = {}
            self.default_profiles = {}

        full_url = f"{self.backend_url}:{self.backend_port}"
        self.output_manager = OutputManager()

        if enable_connection_pool:
            self.pool = ConnectionPool(full_url, pool_config or ConnectionPoolConfig())
        else:
            self.pool = None
            self.vespa = make_vespa_app(url=self.backend_url, port=self.backend_port)

        self.retry_config = retry_config or RetryConfig(
            max_attempts=3, initial_delay=0.5, exceptions=(Exception,)
        )

        # Breaker around the (retried) search path, keyed per endpoint so a
        # down Vespa trips fast instead of every query burning its retries.
        # Load-bearing: an open breaker propagates CircuitOpenError.
        from cogniverse_core.common.utils.circuit_breaker import (
            BreakerConfig,
            CircuitBreaker,
        )

        self._search_breaker = CircuitBreaker.get(
            BreakerConfig(
                name=f"vespa_search:{full_url}",
                failure_threshold=5,
                reset_timeout_s=15.0,
                counted_exceptions=_TRANSIENT_SEARCH_ERRORS,
            )
        )

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
            config: Configuration with keys url, port, schema_name, profile,
                profiles, default_profiles, tenant_id, config_manager
        """
        self.backend_url = config.get("url", "http://localhost")
        self.backend_port = config.get("port", 8080)
        self.schema_name = config.get("schema_name")
        self.query_encoder = None

        # Populate profiles/default_profiles from config. Earlier versions
        # of initialize() dropped these on the floor, so any caller that
        # constructed a backend with no-args and then called initialize()
        # ended up with an empty profiles dict at query time.
        backend_section = config.get("backend", {}) or {}
        with self._profiles_lock:
            self.profiles = dict(
                config.get("profiles") or backend_section.get("profiles") or {}
            )
            self.default_profiles = dict(
                config.get("default_profiles")
                or backend_section.get("default_profiles")
                or {}
            )

        # Combine URL and port
        full_url = f"{self.backend_url}:{self.backend_port}"

        # Initialize output manager
        self.output_manager = OutputManager()

        self.pool = ConnectionPool(full_url, ConnectionPoolConfig())
        self.retry_config = RetryConfig(
            max_attempts=3, initial_delay=0.5, exceptions=(Exception,)
        )

        # Initialize metrics
        self.metrics = SearchMetrics()

        logger.info(
            f"VespaSearchBackend initialized for schema '{self.schema_name}' "
            f"with pool=True, metrics=True, {len(self.profiles)} profiles "
            f"(query-time mode)"
        )

    def add_profile(self, profile_name: str, profile_config: Dict[str, Any]) -> None:
        """Register a profile config at runtime so queries can target it.

        Used by BackendRegistry fanout when ConfigManager.add_backend_profile
        fires its change-listener — closes the gap where dynamically-created
        profiles were only visible to ingestion, not search.
        """
        with self._profiles_lock:
            self.profiles[profile_name] = dict(profile_config)

    def remove_profile(self, profile_name: str) -> None:
        """Unregister a profile from the in-memory dict. Idempotent."""
        with self._profiles_lock:
            self.profiles.pop(profile_name, None)

    def batch_get_documents(
        self, document_ids: List[str], schema_name: Optional[str] = None
    ) -> List[Optional[Document]]:
        """
        Retrieve multiple documents by ID using batch search query (primary batch method).

        Args:
            document_ids: List of document IDs to retrieve
            schema_name: Vespa schema to read from. This backend is shared across
                all tenants and ``self.schema_name`` is rewritten by every search
                request, so a caller MUST pass the schema it means to read —
                relying on the shared attribute races concurrent requests and can
                read another tenant's schema. Falls back to ``self.schema_name``
                only for legacy callers.

        Returns:
            List of Document objects (None for not found), in the same order as document_ids
        """
        if not document_ids:
            return []

        schema = schema_name or self.schema_name

        # Document v1 point GETs — an O(1) dictionary lookup per id. The
        # previous `id contains "<docid>"` YQL substring-matched the internal
        # document URI, which no index serves, so cost grew with corpus size.
        def _fetch(handle) -> Dict[str, Document]:
            results: Dict[str, Document] = {}
            for doc_id in document_ids:
                response = handle.get_data(
                    schema=schema,
                    data_id=doc_id,
                    namespace="content",
                    raise_on_not_found=False,
                )
                if response.status_code != 200:
                    continue
                fields = response.json.get("fields", {})
                doc = Document(
                    id=doc_id,
                    content_type=ContentType.VIDEO,
                    text_content=fields.get("content", ""),
                    status=ProcessingStatus.COMPLETED,
                )
                for key, value in fields.items():
                    if value is not None:
                        doc.add_metadata(key, value)
                results[doc_id] = doc
            return results

        try:
            if self.pool:
                with self.pool.get_connection() as conn:
                    results_dict = _fetch(conn._sync)
            else:
                with self.vespa.syncio() as sync_app:
                    results_dict = _fetch(sync_app)

            return [results_dict.get(doc_id) for doc_id in document_ids]

        except Exception as e:
            logger.error(f"Batch document retrieval failed: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search backend statistics.

        Returns:
            Statistics including document count, metrics, etc.
        """
        return self.get_metrics()

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

    # Retry only for transient failures (network blips, Vespa hiccups).
    # ValueErrors are permanent config problems — wrong content_type,
    # missing profile, malformed inputs — and retrying them just multiplies
    # the wasted latency (e.g. "No profiles found for type 'wiki'" from an
    # empty wiki cache would burn 3×~5s backoffs through the orchestrator
    # LLM chain).
    def search(self, query_dict: Dict[str, Any]) -> List[SearchResult]:
        """Search, guarding the retried path with the per-endpoint breaker.

        Vespa is load-bearing, so an open breaker propagates ``CircuitOpenError``
        (the caller surfaces it) rather than degrading to empty results.
        """
        return self._search_breaker.call(self._search_retried, query_dict)

    def _load_tenant_profiles(self, tenant_id):
        """Return a tenant's (profiles, default_profiles) for per-request
        resolution. Empty dicts when the tenant or config manager is absent."""
        if not tenant_id or self._config_manager is None:
            return {}, {}
        from cogniverse_foundation.config.utils import get_config

        cfg = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        backend_section = cfg.get("backend", {}) or {}
        return (
            dict(backend_section.get("profiles", {}) or {}),
            dict(backend_section.get("default_profiles", {}) or {}),
        )

    def _resolve_encoder_for_profile(self, profile_name, profile_config, tenant_id):
        """Build the query encoder a profile declares so callers that delegate
        encoding to the backend need not thread one. Dense profiles resolve to
        the shared SemanticEmbedder; multi-vector profiles to the video encoder
        factory. Returns None when no encoder is resolvable."""
        embedding_type = str(profile_config.get("embedding_type") or "").lower()
        encoder_name = str(profile_config.get("encoder") or "").lower()
        try:
            if embedding_type == "dense" or encoder_name == "denseon":
                from cogniverse_core.common.models.semantic_embedder import (
                    get_semantic_embedder,
                )

                return _DenseQueryEncoder(get_semantic_embedder())

            model_name = profile_config.get("embedding_model")
            if not model_name or self._config_manager is None:
                return None
            from cogniverse_core.query.encoders import QueryEncoderFactory
            from cogniverse_foundation.config.utils import get_config

            cfg = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
            return QueryEncoderFactory.create_encoder(
                profile_name, model_name, config=cfg
            )
        except Exception as exc:
            logger.warning(
                "Could not resolve an encoder for profile '%s': %r",
                profile_name,
                exc,
            )
            return None

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            exceptions=_TRANSIENT_SEARCH_ERRORS,
        )
    )
    def _search_retried(self, query_dict: Dict[str, Any]) -> List[SearchResult]:
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
        query_embeddings = query_dict.get("query_embeddings")
        # The backend is shared across profiles and created once (encoder-less
        # at startup), so callers that delegate encoding pass the per-profile
        # encoder per request rather than relying on a baked-in one.
        request_encoder = query_dict.get("query_encoder") or self.query_encoder
        # Either text or pre-computed embeddings is sufficient. Mem0 in
        # particular hands us a 768-dim embedding with an empty query string
        # during its internal duplicate/context scans; forcing text here
        # triggered a ValueError that Mem0 retried 3× per call, hanging the
        # orchestrator's detailed_report path past the client timeout.
        if not query_text and query_embeddings is None:
            raise ValueError(
                "query_dict must contain 'query' text or 'query_embeddings'"
            )

        content_type = query_dict.get("type")
        if not content_type:
            raise ValueError("query_dict must contain 'type' key (e.g., 'video')")

        top_k = query_dict.get("top_k", 10)
        filters = query_dict.get("filters", {})  # Get filters from query_dict

        # Profile resolution
        # Snapshot profiles+default_profiles under the lock so a concurrent
        # add_profile / remove_profile can't produce a torn read here.
        with self._profiles_lock:
            profiles_snapshot = dict(self.profiles)
            default_profiles_snapshot = dict(self.default_profiles)
        requested_profile = query_dict.get("profile")

        # The shared backend freezes whichever tenant's profiles created it, so
        # a per-tenant profile (or a backend built config-less at startup) may
        # be absent. Merge the query tenant's profiles per-request — locally,
        # so tenants can't leak profiles into each other's snapshots.
        if self._config_manager is not None and (
            not profiles_snapshot
            or (requested_profile and requested_profile not in profiles_snapshot)
        ):
            tenant_profiles, tenant_defaults = self._load_tenant_profiles(
                query_dict.get("tenant_id")
            )
            profiles_snapshot = {**profiles_snapshot, **tenant_profiles}
            default_profiles_snapshot = {
                **default_profiles_snapshot,
                **tenant_defaults,
            }

        if requested_profile:
            # 1. Use explicitly requested profile
            if requested_profile not in profiles_snapshot:
                raise ValueError(
                    f"Requested profile '{requested_profile}' not found. "
                    f"Available profiles: {list(profiles_snapshot.keys())}"
                )
            profile_name = requested_profile
            logger.info(f"[{correlation_id}] Using requested profile: {profile_name}")
        else:
            # 2. Auto-select based on type
            # Get all profiles for this type
            type_profiles = {
                name: config
                for name, config in profiles_snapshot.items()
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
                default_config = default_profiles_snapshot.get(content_type, {})
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
                    f"Available types: {set(p.get('type') for p in profiles_snapshot.values())}"
                )

        # Get profile config from the same snapshot used above
        profile_config = profiles_snapshot[profile_name]

        # Determine schema_name from profile (base name)
        base_schema_name = profile_config.get("schema_name", profile_name)

        # Apply tenant scoping - tenant_id is REQUIRED in query_dict
        tenant_id = query_dict.get("tenant_id")
        if not tenant_id:
            raise ValueError(
                "tenant_id is required in query_dict for search operations. "
                f"Profile '{profile_name}' cannot be used without tenant isolation."
            )

        # Canonicalize tenant_id (e.g. "test_tenant" → "test_tenant:test_tenant") so the
        # schema name matches the double-suffix form written by the ingestion/deploy path.
        # Without this, "test_tenant" maps to "..._test_tenant" but the deployed schema
        # is "..._test_tenant_test_tenant".
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        safe_tenant_id = canonical_tenant_id(tenant_id).replace(":", "_")
        schema_name = f"{base_schema_name}_{safe_tenant_id}"
        logger.info(
            f"[{correlation_id}] Applied tenant scoping: {base_schema_name} → {schema_name}"
        )

        # Load ranking strategies from internal cache (not from Strategy object)
        global _RANKING_STRATEGIES_CACHE
        with _CACHE_LOCK:
            if _RANKING_STRATEGIES_CACHE is None:
                loaded = self._load_ranking_strategies()
                if loaded:
                    # Only cache non-empty results — a backend with an invalid
                    # schema_loader (e.g. Mock) returns {} which must NOT poison
                    # the global cache for other backends with valid loaders.
                    _RANKING_STRATEGIES_CACHE = loaded

        cache = _RANKING_STRATEGIES_CACHE or {}
        available_strategies = cache.get(base_schema_name, {})
        if not available_strategies:
            raise ValueError(
                f"No ranking strategies found for schema '{base_schema_name}'. "
                f"Available schemas: {list(cache.keys())}"
            )

        # Strategy resolution
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
                    # Fall back to the schema's conventional default profile,
                    # mirroring how the profile registry derives a default
                    # (default → hybrid → first available).
                    strategy_name = next(
                        (s for s in ("default", "hybrid") if s in available_strategies),
                        next(iter(available_strategies)),
                    )

                logger.info(
                    f"[{correlation_id}] Using default strategy: {strategy_name}"
                )
            else:
                raise ValueError(f"No strategies found in profile '{profile_name}'")

        self.schema_name = schema_name

        logger.info(
            f"[{correlation_id}] Query resolved: type={content_type}, profile={profile_name}, "
            f"schema={schema_name}, strategy={strategy_name}"
        )

        start_time = time.time()

        try:
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
                # No caller-supplied encoder: build the one the profile
                # declares so a search that needs embeddings never runs
                # without them (dense -> SemanticEmbedder, video -> factory).
                if request_encoder is None:
                    request_encoder = self._resolve_encoder_for_profile(
                        profile_name, profile_config, tenant_id
                    )
                if request_encoder:
                    from cogniverse_foundation.telemetry.context import (
                        add_embedding_details_to_span,
                        encode_span,
                    )

                    logger.info(
                        f"[{correlation_id}] Generating embeddings on-demand "
                        f"for strategy '{strategy_name}'"
                    )
                    encoder_type = (
                        type(request_encoder)
                        .__name__.lower()
                        .replace("queryencoder", "")
                    )
                    with encode_span(
                        tenant_id=tenant_id,
                        encoder_type=encoder_type,
                        query_length=len(query_text),
                        query=query_text,
                    ) as encode_span_ctx:
                        query_embeddings = request_encoder.encode(query_text)
                        add_embedding_details_to_span(encode_span_ctx, query_embeddings)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"[{correlation_id}] Embeddings shape="
                            f"{query_embeddings.shape} dtype={query_embeddings.dtype} "
                            f"min/max={query_embeddings.min():.4f}/"
                            f"{query_embeddings.max():.4f}"
                        )
                else:
                    raise ValueError(
                        f"Strategy '{strategy_name}' needs query embeddings but "
                        f"none were provided and no encoder is available. Pass "
                        f"'query_embeddings' or a 'query_encoder' in query_dict."
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
                f"in {(time.time() - start_time) * 1000:.2f}ms"
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
            if value is None:
                # None means "no filter on this field", not a match on the
                # literal string "None".
                continue
            _validate_filter_field_name(field_name)
            value = _coerce_numpy_scalar(value)
            if isinstance(value, dict):
                # Range filter, e.g. {"gte": 100, "lte": 200} ->
                # 'field >= 100 AND field <= 200'. Numeric range values must
                # be finite numbers; reject NaN/Inf to avoid malformed YQL
                # (Vespa rejects ``field >= nan``). String range values are
                # quoted so they survive the parser.
                for key, sql_op in (
                    ("gte", ">="),
                    ("gt", ">"),
                    ("lte", "<="),
                    ("lt", "<"),
                ):
                    if key not in value:
                        continue
                    bound = value[key]
                    conditions.append(
                        f"{field_name} {sql_op} {_yql_scalar(bound, field_name)}"
                    )
            elif isinstance(value, bool):
                # Boolean values (checked before int — bool is an int subclass)
                conditions.append(f"{field_name} = {str(value).lower()}")
            elif isinstance(value, str):
                # String attributes use 'contains' for matching
                # This works for string attributes in Vespa
                conditions.append(f"{field_name} contains {yql_quote(value)}")
            elif isinstance(value, (int, float)):
                # Numeric values use equality
                conditions.append(f"{field_name} = {_yql_scalar(value, field_name)}")
            else:
                # For other types, convert to string and use contains
                conditions.append(f"{field_name} contains {yql_quote(str(value))}")

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

        # Per-strategy request timeout (seconds). Without it a hung Vespa query
        # runs unbounded and the connection pool (bounded) drains, stalling
        # every concurrent search. Loaded from the strategy config but was
        # never forwarded before.
        strategy_timeout = rank_config.get("timeout")
        if strategy_timeout:
            query_params["timeout"] = f"{float(strategy_timeout)}s"

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

        if query_embeddings is not None:
            inputs_needed = rank_config.get("inputs", {})

            logger.debug(
                f"[{correlation_id}] Strategy '{ranking_profile}' needs inputs: {list(inputs_needed.keys())}"
            )
            logger.debug(
                f"[{correlation_id}] Input query_embeddings shape: {query_embeddings.shape}, dtype: {query_embeddings.dtype}"
            )
            if query_embeddings.ndim == 2:
                logger.debug(
                    f"[{correlation_id}] Input query_embeddings (2D) first vector (first 5): {query_embeddings[0][:5].tolist()}"
                )
            else:
                logger.debug(
                    f"[{correlation_id}] Input query_embeddings (1D) first 5 values: {query_embeddings[:5].tolist()}"
                )

            for input_name, input_type in inputs_needed.items():
                # The input names are like 'qt', 'qtb', etc. We need to construct the full Vespa query parameter name
                vespa_param_name = f"input.query({input_name})"

                if input_name in ("qt", "acoustic_query") and "float" in input_type:
                    logger.debug(
                        f"[{correlation_id}] Adding float embeddings for {vespa_param_name}"
                    )
                    query_params[vespa_param_name] = _format_query_vector_param(
                        query_embeddings, schema_name
                    )

                elif input_name == "qtb" and "int8" in input_type:
                    logger.debug(
                        f"[{correlation_id}] Adding binary embeddings for {vespa_param_name}"
                    )
                    binary_embeddings = self._generate_binary_embeddings(
                        query_embeddings
                    )
                    logger.debug(
                        f"[{correlation_id}] Binary embeddings shape: {binary_embeddings.shape}, dtype: {binary_embeddings.dtype}"
                    )
                    query_params[vespa_param_name] = _format_query_vector_param(
                        binary_embeddings, schema_name
                    )

                elif input_name == "q":
                    logger.debug(
                        f"[{correlation_id}] Adding generic embeddings for {vespa_param_name}"
                    )
                    emb = query_embeddings
                    if emb.ndim == 2 and emb.shape[0] == 1:
                        emb = emb[0]
                    query_params[vespa_param_name] = emb.tolist()

                else:
                    # Unknown input name - this shouldn't happen with proper configuration
                    logger.warning(
                        f"[{correlation_id}] Unknown input name '{input_name}' with type '{input_type}' "
                        f"in ranking strategy '{ranking_profile}'. This input will be skipped. "
                        f"Known input names: 'qt'/'acoustic_query' (float), "
                        f"'qtb' (binary), 'q' (generic)."
                    )

        query_params["model.restrict"] = schema_name
        logger.info(
            f"[{correlation_id}] Set model.restrict='{query_params['model.restrict']}'"
        )

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

        doc_id = result.get("id", "").split("::")[-1]

        document = Document(
            id=doc_id, content_type=ContentType.VIDEO, status=ProcessingStatus.COMPLETED
        )

        for text_field_name in ["text", "transcription", "text_content", "content"]:
            if text_field_name in fields and fields[text_field_name]:
                document.text_content = fields[text_field_name]
                break

        for key, value in fields.items():
            if value is not None:
                document.add_metadata(key, value)

        document.add_metadata("source_id", fields.get("video_id", doc_id.split("_")[0]))

        return document

    def _process_results(
        self, response: Any, correlation_id: str
    ) -> List[SearchResult]:
        """Process Vespa response into SearchResult objects.

        Raises:
            VespaError: If the response body carries ``root.errors``. Vespa
                reports soft timeouts and container errors as HTTP 200 with
                an errors list and partial/empty children — consuming hits
                without this check turns a degraded backend into "no
                results" recorded as success.
        """
        results = []

        if not response or not hasattr(response, "hits"):
            logger.warning(f"[{correlation_id}] Empty response from Vespa")
            return results

        root = {}
        if hasattr(response, "get_json"):
            root = (response.get_json() or {}).get("root", {})
        errors = root.get("errors", [])
        if errors:
            raise VespaError(
                f"[{correlation_id}] Vespa query returned errors: {errors}"
            )
        coverage = root.get("coverage", {})
        if coverage.get("degraded"):
            logger.warning(
                f"[{correlation_id}] Vespa coverage degraded: "
                f"{coverage.get('coverage')}% of corpus searched "
                f"({coverage.get('degraded')})"
            )

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
                    f"[{correlation_id}] Failed to process hit: {e}, hit data: {hit}"
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    def get_document(
        self, document_id: str, schema_name: Optional[str] = None
    ) -> Optional[Document]:
        """
        Retrieve a specific document by ID (uses batch method).

        Args:
            document_id: Document ID to retrieve
            schema_name: Vespa schema to read from — pass it explicitly; this
                backend is shared across tenants (see batch_get_documents).

        Returns:
            Document if found, None otherwise
        """
        # Use batch method for consistency and optimization
        results = self.batch_get_documents([document_id], schema_name=schema_name)
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

        # Use visit API for bulk export
        namespace = "content"
        target_schema = schema or self.schema_name
        visit_url = f"{self.backend_url}:{self.backend_port}/document/v1/{namespace}/{target_schema}/docid"

        # Document v1 visit selection. Each value is escaped via _yql_scalar
        # (yql_quote for strings, finite-checked literal for numbers) so a
        # value with an embedded quote can't break the expression or inject.
        if filters:
            selection = " and ".join(
                f"{target_schema}.{key} == {_yql_scalar(value, key)}"
                for key, value in filters.items()
            )
        else:
            selection = "true"

        try:
            response = requests.get(
                visit_url,
                params={
                    "selection": selection,
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
                            "selection": selection,
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
                loaded = self._load_ranking_strategies()
                if loaded:
                    _RANKING_STRATEGIES_CACHE = loaded

        # Get strategies for this schema
        cache = _RANKING_STRATEGIES_CACHE or {}
        schema_strategies = cache.get(schema_name, {})

        if not schema_strategies:
            raise ValueError(
                f"No ranking strategies found for schema '{schema_name}'. "
                f"Available schemas: {list(cache.keys())}"
            )

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
            raise


# Factory function for creating search backend
def create_vespa_search_backend(
    schema_name: str, backend_url: str = "http://localhost:8080", **kwargs
) -> VespaSearchBackend:
    """
    Create a production-ready Vespa search backend

    Args:
        schema_name: Vespa schema name
        backend_url: Backend URL
        **kwargs: Additional configuration

    Returns:
        VespaSearchBackend instance
    """
    return VespaSearchBackend(
        backend_url=backend_url, schema_name=schema_name, **kwargs
    )
