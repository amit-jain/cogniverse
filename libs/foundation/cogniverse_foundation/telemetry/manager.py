"""
Multi-tenant telemetry manager.

Handles:
- Lazy initialization of tenant-specific tracer providers
- LRU caching of tenant providers
- Batch export with proper queue management
- Graceful degradation when telemetry unavailable
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode, Tracer

from .config import TelemetryConfig

logger = logging.getLogger(__name__)


class TelemetryManager:
    """
    Multi-tenant telemetry manager with lazy initialization and caching.

    Usage:
        # Initialize once (singleton pattern)
        telemetry = TelemetryManager()

        # Use per-request with tenant-id
        with telemetry.span("search_service.search", tenant_id="tenant-123") as span:
            span.set_attribute("query", "test")
            # ... search logic ...
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[TelemetryConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[TelemetryConfig] = None):
        if self._initialized:
            return

        if config is None:
            raise ValueError(
                "TelemetryConfig is required. Use get_telemetry_manager() "
                "which loads config from ConfigManager automatically."
            )
        self.config = config
        self.config.validate()

        # Thread-safe caches
        self._tenant_providers: Dict[str, TracerProvider] = {}
        self._tenant_tracers: Dict[str, Tracer] = {}
        # cache_key -> the _tenant_providers key that tracer was built from.
        # Lets eviction drop providers no live tracer still references,
        # without parsing the (colon-bearing) tenant id out of cache_key.
        self._tracer_provider_keys: Dict[str, str] = {}
        # cache_key -> time.monotonic() at insert; entries older than
        # config.tenant_cache_ttl_seconds are rebuilt on access.
        self._tracer_created_at: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Per-project configs (single source of truth for project-specific settings)
        self._project_configs: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._failed_initializations = 0

        self._initialized = True
        logger.info(f"TelemetryManager initialized with config: {self.config}")

    def get_tracer(
        self, tenant_id: str, project_name: Optional[str] = None
    ) -> Optional[Tracer]:
        """
        Get tracer for a specific tenant project (prefer ``span()``).

        Args:
            tenant_id: Tenant identifier for project isolation
            project_name: Optional service name for management operations
                         (e.g., "experiments", "synthetic_data", "system")
                         If None, uses tenant-only project (user operations)

        Returns:
            Tracer instance or None if telemetry disabled/failed
        """
        # Enforce mandatory tenant_id - no non-tenant tracers allowed
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for all tracers. "
                "Non-tenant-specific tracers are not allowed."
            )

        if not self.config.enabled:
            return None

        full_project_name = self.config.get_project_name(tenant_id, project_name)
        cache_key = f"{tenant_id}:{full_project_name}"

        # Check cache first
        with self._lock:
            cached = self._cached_tracer(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

            # Create tracer provider for tenant if needed
            try:
                if tenant_id not in self._tenant_providers:
                    self._tenant_providers[tenant_id] = self._create_tenant_provider(
                        tenant_id
                    )

                tracer_provider = self._tenant_providers[tenant_id]
                tracer = tracer_provider.get_tracer(full_project_name)

                # Cache with LRU eviction
                self._tenant_tracers[cache_key] = tracer
                self._tracer_provider_keys[cache_key] = tenant_id
                self._tracer_created_at[cache_key] = time.monotonic()
                self._evict_old_tracers()

                return tracer

            except Exception as e:
                self._failed_initializations += 1
                logger.warning(f"Failed to create tracer for tenant {tenant_id}: {e}")
                return None

    def register_project(self, tenant_id: str, project_name: str, **kwargs) -> None:
        """
        Register a project with optional config overrides.

        Allows per-project configuration (e.g., different telemetry endpoints for tests).
        Single source of truth for project settings.

        Args:
            tenant_id: Tenant identifier
            project_name: Project name (e.g., "search", "synthetic_data", "routing")
            **kwargs: Optional overrides:
                - otlp_endpoint: Override OTLP gRPC endpoint for span export
                - http_endpoint: Override HTTP endpoint for span queries
                - use_sync_export: Override batch/sync export mode

        Examples:
            # Use defaults from config
            manager.register_project(tenant_id="customer-123", project_name="search")

            # Override endpoints for tests
            manager.register_project(
                tenant_id="test-tenant1",
                project_name="synthetic_data",
                otlp_endpoint="http://localhost:24317",
                http_endpoint="http://localhost:26006",
                use_sync_export=True
            )

        Project naming: cogniverse-{tenant_id}-{project_name}
        """
        project_key = f"{tenant_id}:{project_name}"

        # Build config with optional overrides
        project_config = {
            "tenant_id": tenant_id,
            "project_name": project_name,
            "otlp_endpoint": kwargs.get("otlp_endpoint", self.config.otlp_endpoint),
            "http_endpoint": kwargs.get("http_endpoint", None),
            "grpc_endpoint": kwargs.get("grpc_endpoint", None),
            "use_sync_export": kwargs.get(
                "use_sync_export", self.config.batch_config.use_sync_export
            ),
        }

        with self._lock:
            self._project_configs[project_key] = project_config

        logger.info(
            f"Registered project {project_key} "
            f"(otlp_endpoint={project_config['otlp_endpoint']}, "
            f"http_endpoint={project_config.get('http_endpoint')}, "
            f"grpc_endpoint={project_config.get('grpc_endpoint')}, "
            f"sync_export={project_config['use_sync_export']})"
        )

    @contextmanager
    def span(
        self,
        name: str,
        tenant_id: str,
        project_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        component: str = "agents",
    ):
        """
        Context manager for creating tenant-specific spans.

        Args:
            name: Span name
            tenant_id: Tenant identifier
            project_name: Optional service name for management operations
                         (e.g., "experiments", "synthetic_data", "system")
                         If None, uses tenant-only project for user operations
            attributes: Optional span attributes
            component: Telemetry-level component tag — one of
                ``search_service`` / ``agents`` / ``backend`` /
                ``pipeline`` / ``encoder``. ``TelemetryConfig.level``
                controls which components emit (see
                ``should_instrument_component``). Default ``agents`` —
                emits at DETAILED and VERBOSE (the default and
                higher). Callers emitting per-inference model details
                should pass ``encoder``; ingestion-stage spans should
                pass ``pipeline``; the top-level search HTTP entry
                should pass ``search_service`` (admitted at BASIC).

        Usage:
            # User operation (search, routing, etc.) - unified tenant project.
            # The request handler must have resolved an explicit tenant_id
            # before this call; there is no silent default.
            tenant_id = require_tenant_id(context.get("tenant_id"),
                                          source="telemetry.span")
            with telemetry.span("search", tenant_id=tenant_id) as span:
                span.set_attribute("query", "test")

            # Management operation - separate project
            with telemetry.span("evaluate", tenant_id=tenant_id, project_name="experiments") as span:
                span.set_attribute("experiment.name", "test_experiment")
        """
        # Enforce mandatory tenant_id - no non-tenant tracers allowed
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for all spans. "
                "Non-tenant-specific spans are not allowed."
            )

        # Telemetry-level filter: if the component is below the
        # configured level, yield a NoOpSpan so callers' set_attribute /
        # set_status calls become no-ops. Lets ops dial down telemetry
        # cost in production without code changes.
        if not self.config.should_instrument_component(component):
            yield NoOpSpan()
            return

        tracer = self._get_tracer_for_project(tenant_id, project_name)

        if tracer is None:
            # Graceful degradation - yield no-op span
            yield NoOpSpan()
            return

        with tracer.start_as_current_span(name) as span:
            # Add tenant context to all spans
            span.set_attribute("tenant.id", tenant_id)
            span.set_attribute("service.name", self.config.service_name)
            span.set_attribute("environment", self.config.environment)

            # Add user-provided attributes (including openinference.project.name for routing)
            if attributes:
                for key, value in attributes.items():
                    # Skip None values as OpenTelemetry will reject them
                    if value is not None:
                        span.set_attribute(key, value)
                    else:
                        logger.debug(f"Skipping None attribute: {key}")

            try:
                yield span
            except Exception as e:
                # Record exception in span and re-raise
                logger.warning(f"Error in span {name} for tenant {tenant_id}: {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def session_span(
        self,
        name: str,
        tenant_id: str,
        session_id: str,
        project_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        component: str = "search_service",
    ):
        """
        Context manager for creating a span within a session context.

        This combines session tracking with span creation:
        1. Establishes session context (via provider)
        2. Creates a span within that context
        3. All nested spans inherit the session_id

        Args:
            name: Span name
            tenant_id: Tenant identifier
            session_id: Session identifier for cross-request correlation
            project_name: Optional project name suffix
            attributes: Optional span attributes

        Usage:
            # At API entry point - establishes session context
            with telemetry.session_span(
                "api.search.request",
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                attributes={"query": request.query}
            ) as span:
                # All child spans automatically inherit session_id
                result = await search_service.search(query=request.query)
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for session_span")

        if not session_id:
            raise ValueError("session_id is required for session_span")

        try:
            provider = self.get_provider(tenant_id=tenant_id, project_name=project_name)
        except Exception as e:
            logger.warning(f"Failed to get provider for session tracking: {e}")
            provider = None

        if provider is None:
            # Graceful degradation - no session tracking, just create span
            with self.span(
                name, tenant_id, project_name, attributes, component=component
            ) as span:
                yield span
            return

        # Wrap span creation in session context
        with provider.session_context(session_id):
            with self.span(
                name, tenant_id, project_name, attributes, component=component
            ) as span:
                # Session ID is now propagated via the session context
                yield span

    def _get_tracer_for_project(
        self, tenant_id: str, project_name: Optional[str]
    ) -> Optional[Tracer]:
        """Get tracer for specific project."""
        if not self.config.enabled:
            return None

        full_project_name = self.config.get_project_name(tenant_id, project_name)
        cache_key = f"{tenant_id}:{full_project_name}"

        # Check cache first
        with self._lock:
            cached = self._cached_tracer(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

            # Create tracer provider for project if needed
            try:
                provider_key = f"{tenant_id}:{full_project_name}"
                if provider_key not in self._tenant_providers:
                    self._tenant_providers[provider_key] = (
                        self._create_tenant_provider_for_project(
                            tenant_id, project_name
                        )
                    )

                tracer_provider = self._tenant_providers[provider_key]
                tracer = tracer_provider.get_tracer(full_project_name)

                # Cache with LRU eviction
                self._tenant_tracers[cache_key] = tracer
                self._tracer_provider_keys[cache_key] = provider_key
                self._tracer_created_at[cache_key] = time.monotonic()
                self._evict_old_tracers()

                return tracer

            except Exception as e:
                self._failed_initializations += 1
                logger.warning(f"Failed to create tracer for {cache_key}: {e}")
                return None

    def get_provider(self, tenant_id: str, project_name: Optional[str] = None):
        """
        Get telemetry provider for querying spans/annotations/datasets.

        This is separate from span export (which uses OpenTelemetry OTLP).
        Providers are used for reading data from the telemetry backend.

        Args:
            tenant_id: Tenant identifier
            project_name: Optional project name to get project-specific config

        Returns:
            TelemetryProvider instance

        Raises:
            ValueError: If no providers available or provider initialization fails

        Usage:
            # Get provider for querying
            provider = telemetry_manager.get_provider(tenant_id="customer-123")

            # Query spans
            spans_df = await provider.traces.get_spans(
                project="cogniverse-customer-123-search",
                start_time=datetime(2025, 1, 1),
                limit=1000
            )

            # Add annotations
            await provider.annotations.add_annotation(
                span_id="abc123",
                name="human_approval",
                label="approved",
                score=1.0,
                metadata={"reviewer": "alice"},
                project="cogniverse-customer-123-synthetic_data"
            )
        """
        # Enforce mandatory tenant_id - no non-tenant providers allowed
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for all providers. "
                "Non-tenant-specific providers are not allowed."
            )

        from cogniverse_foundation.telemetry.registry import get_telemetry_registry

        registry = get_telemetry_registry()

        # Build generic config for provider (provider interprets keys)
        # Derive grpc_endpoint and http_endpoint from otlp_endpoint when not
        # explicitly set in provider_config, so providers like Phoenix can
        # initialise without requiring manual provider_config entries.
        otlp_ep = self.config.otlp_endpoint  # e.g. "localhost:4317"
        scheme = "https" if self.config.otlp_use_tls else "http"
        grpc_default = f"{scheme}://{otlp_ep}" if "://" not in otlp_ep else otlp_ep
        # HTTP endpoint: replace gRPC port (4317) with HTTP port (6006)
        http_default = grpc_default.replace(":4317", ":6006")

        provider_config = {
            "tenant_id": tenant_id,
            "grpc_endpoint": grpc_default,
            "http_endpoint": http_default,
            **self.config.provider_config,  # Explicit overrides take precedence
        }
        # Carry the project so the registry caches one provider PER project —
        # otherwise the project-specific endpoint overlay below is computed but
        # a second project for the same tenant reuses the first's cached
        # provider (and its endpoints).
        if project_name:
            provider_config["project_name"] = project_name

        # Check for project-specific endpoints in registered project configs
        if project_name:
            project_key = f"{tenant_id}:{project_name}"
            if project_key in self._project_configs:
                cfg = self._project_configs[project_key]
                if cfg.get("http_endpoint"):
                    provider_config["http_endpoint"] = cfg["http_endpoint"]
                if cfg.get("grpc_endpoint"):
                    provider_config["grpc_endpoint"] = cfg["grpc_endpoint"]

        # Get provider from registry (auto-discovers via entry points)
        # If config.provider is None, registry auto-selects first available
        return registry.get(
            name=self.config.provider,  # None = auto-detect
            tenant_id=tenant_id,
            config=provider_config,  # Generic dict - provider interprets
        )

    def _create_tenant_provider(self, tenant_id: str) -> TracerProvider:
        """Create and configure TracerProvider for a tenant."""
        return self._create_tenant_provider_for_project(
            tenant_id, self.config.service_name
        )

    def _create_tenant_provider_for_project(
        self, tenant_id: str, project_suffix: str
    ) -> TracerProvider:
        """Create and configure TracerProvider for a specific tenant project."""
        if not self.config.otlp_enabled:
            raise RuntimeError("OTLP span export not enabled")

        project_key = f"{tenant_id}:{project_suffix}"

        # Check for registered project config (single source of truth)
        if project_key in self._project_configs:
            cfg = self._project_configs[project_key]
            endpoint = cfg["otlp_endpoint"]
            use_sync_export = cfg["use_sync_export"]
            logger.debug(f"Using registered config for {project_key}")
        else:
            # Fall back to default config
            endpoint = self.config.otlp_endpoint
            use_sync_export = self.config.batch_config.use_sync_export
            logger.debug(f"Using default config for {project_key}")

        try:
            # Determine full project name: cogniverse-{tenant_id}-{project_suffix}
            project_name = self.config.get_project_name(tenant_id, project_suffix)

            # Get telemetry provider (auto-discovered via registry)
            # Pass project_suffix to get project-specific endpoint overrides
            provider = self.get_provider(
                tenant_id=tenant_id, project_name=project_suffix
            )

            # Use provider to configure span export (backend-agnostic)
            use_batch_export = not use_sync_export
            resource_attributes = {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                **self.config.extra_resource_attributes,
            }
            tracer_provider = provider.configure_span_export(
                endpoint=endpoint,
                project_name=project_name,
                use_batch_export=use_batch_export,
                batch_config=self.config.batch_config,
                resource_attributes=resource_attributes,
            )

            mode = "BATCH" if use_batch_export else "SYNC"
            logger.info(
                f"Created {mode} tracer provider: {project_name} (endpoint={endpoint})"
            )
            return tracer_provider

        except Exception as e:
            logger.error(f"Failed to create tracer provider for {project_key}: {e}")
            raise

    def _cached_tracer(self, cache_key: str) -> Optional[Tracer]:
        """Return the cached tracer, or None when absent or expired.

        An entry older than ``config.tenant_cache_ttl_seconds`` is dropped
        (together with its provider, once no other tracer references it) so
        the caller rebuilds both. A TTL of 0 or less disables expiry.
        Caller must hold ``self._lock``.
        """
        tracer = self._tenant_tracers.get(cache_key)
        if tracer is None:
            return None
        ttl = self.config.tenant_cache_ttl_seconds
        created = self._tracer_created_at.get(cache_key)
        if ttl > 0 and created is not None and time.monotonic() - created > ttl:
            del self._tenant_tracers[cache_key]
            self._tracer_created_at.pop(cache_key, None)
            self._tracer_provider_keys.pop(cache_key, None)
            self._evict_orphaned_providers()
            logger.debug(f"Expired tracer from cache (TTL): {cache_key}")
            return None
        return tracer

    def _evict_old_tracers(self):
        """Evict old tracers (LRU) and any providers no tracer still references.

        Providers are evicted only once no remaining tracer maps to them, so a
        provider shared across one tenant's projects survives until its last
        tracer is gone. ``shutdown()`` flushes the provider's pending spans
        before it is dropped.
        """
        if len(self._tenant_tracers) > self.config.max_cached_tenants:
            items_to_remove = len(self._tenant_tracers) - self.config.max_cached_tenants
            oldest_keys = list(self._tenant_tracers.keys())[:items_to_remove]
            for key in oldest_keys:
                del self._tenant_tracers[key]
                self._tracer_provider_keys.pop(key, None)
                self._tracer_created_at.pop(key, None)
                logger.debug(f"Evicted tracer from cache: {key}")

        self._evict_orphaned_providers()

    def _evict_orphaned_providers(self):
        """Shut down and drop providers no cached tracer references."""
        still_referenced = set(self._tracer_provider_keys.values())
        for provider_key in list(self._tenant_providers.keys()):
            if provider_key in still_referenced:
                continue
            provider = self._tenant_providers.pop(provider_key)
            try:
                provider.shutdown()
            except Exception:
                logger.debug(
                    "Provider shutdown failed for %s", provider_key, exc_info=True
                )
            logger.debug(f"Evicted tracer provider from cache: {provider_key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry manager statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "failed_initializations": self._failed_initializations,
            "cached_tenants": len(self._tenant_providers),
            "cached_tracers": len(self._tenant_tracers),
            "config": {
                "enabled": self.config.enabled,
                "level": self.config.level.value,
                "environment": self.config.environment,
            },
        }

    def force_flush(self, timeout_millis: int = 10000) -> bool:
        """
        Force flush all spans for all tenants.

        Args:
            timeout_millis: Timeout in milliseconds for flush operation

        Returns:
            True if all flushes succeeded, False otherwise
        """
        all_success = True
        with self._lock:
            for tenant_id, provider in self._tenant_providers.items():
                try:
                    if hasattr(provider, "force_flush"):
                        success = provider.force_flush(timeout_millis=timeout_millis)
                        if not success:
                            logger.warning(f"Force flush failed for tenant {tenant_id}")
                            all_success = False
                        else:
                            logger.info(
                                f"Successfully flushed spans for tenant {tenant_id}"
                            )
                except Exception as e:
                    logger.error(f"Error flushing spans for tenant {tenant_id}: {e}")
                    all_success = False
        return all_success

    def shutdown(self):
        """Shutdown all tracer providers gracefully."""
        with self._lock:
            for tenant_id, provider in self._tenant_providers.items():
                try:
                    if hasattr(provider, "force_flush"):
                        provider.force_flush(timeout_millis=5000)
                    if hasattr(provider, "shutdown"):
                        provider.shutdown()
                except Exception as e:
                    logger.warning(
                        f"Error shutting down provider for tenant {tenant_id}: {e}"
                    )

            self._tenant_providers.clear()
            self._tenant_tracers.clear()
            self._tracer_provider_keys.clear()
            self._tracer_created_at.clear()

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton instance - FOR TESTS ONLY.

        Shuts down all tracer providers and clears singleton state.
        Allows tests to start with fresh TelemetryManager.
        """
        if cls._instance is not None:
            try:
                cls._instance.shutdown()
                # Clear registered project configs
                cls._instance._project_configs.clear()
            except Exception as e:
                logger.warning(f"Error during reset: {e}")
            finally:
                cls._instance = None
                logger.info("TelemetryManager singleton reset")


class NoOpSpan:
    """No-op span for graceful degradation."""

    def set_attribute(self, key: str, value: Any):
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception: Exception):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Global singleton instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager(config_manager=None) -> TelemetryManager:
    """
    Get the global telemetry manager singleton.

    On first call, loads telemetry config from ConfigManager under
    ``SYSTEM_TENANT_ID`` — telemetry config is cluster-wide (OTLP
    endpoint, batch/export settings). Per-request tenant scoping
    happens inside ``TelemetryManager.span(tenant_id=...)``, not here.

    Args:
        config_manager: Optional ConfigManager to load config from.
            If None on first call, creates one via create_default_config_manager().
    """
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

    global _telemetry_manager
    if _telemetry_manager is None:
        if config_manager is None:
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
            )

            config_manager = create_default_config_manager()
        config = config_manager.get_telemetry_config(SYSTEM_TENANT_ID)
        _telemetry_manager = TelemetryManager(config)

        # Apply TELEMETRY_OTLP_ENDPOINT env var override (set by Helm chart
        # in k3d deployments, pointing to cogniverse-phoenix:4317).
        # Without this, the config defaults to localhost:4317 which is wrong
        # inside a pod. This mirrors what main.py does at startup.
        import os

        otlp_override = os.environ.get("TELEMETRY_OTLP_ENDPOINT")
        if otlp_override and otlp_override != _telemetry_manager.config.otlp_endpoint:
            _telemetry_manager.config.otlp_endpoint = otlp_override
            _telemetry_manager._tenant_providers.clear()
            _telemetry_manager._tenant_tracers.clear()
            _telemetry_manager._tracer_provider_keys.clear()
    return _telemetry_manager
