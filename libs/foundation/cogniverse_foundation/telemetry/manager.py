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
        Get tracer for a specific tenant project (LEGACY - use span() instead).

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
            if cache_key in self._tenant_tracers:
                self._cache_hits += 1
                return self._tenant_tracers[cache_key]

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

        Usage:
            # User operation (search, routing, etc.) - unified tenant project
            tenant_id = context.get("tenant_id", config.default_tenant_id)
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
    def session(
        self,
        tenant_id: str,
        session_id: str,
        project_name: Optional[str] = None,
    ):
        """
        Context manager for session tracking without creating a span.

        Use this to wrap multiple operations that should share a session_id.
        All spans created within this context will be associated with the session.

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier for cross-request correlation
            project_name: Optional project name suffix

        Usage:
            with telemetry.session(tenant_id, session_id="user-session-abc"):
                # Multiple spans created here share the session_id
                with telemetry.span("operation1", tenant_id):
                    pass
                with telemetry.span("operation2", tenant_id):
                    pass
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for session tracking")

        if not session_id:
            raise ValueError("session_id is required for session tracking")

        try:
            provider = self.get_provider(tenant_id=tenant_id, project_name=project_name)
        except Exception as e:
            logger.warning(f"Failed to get provider for session tracking: {e}")
            provider = None

        if provider is None:
            yield  # No-op if telemetry disabled
            return

        with provider.session_context(session_id):
            yield

    @contextmanager
    def session_span(
        self,
        name: str,
        tenant_id: str,
        session_id: str,
        project_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
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
            with self.span(name, tenant_id, project_name, attributes) as span:
                yield span
            return

        # Wrap span creation in session context
        with provider.session_context(session_id):
            with self.span(name, tenant_id, project_name, attributes) as span:
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
            if cache_key in self._tenant_tracers:
                self._cache_hits += 1
                return self._tenant_tracers[cache_key]

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
        return registry.get_telemetry_provider(
            name=self.config.provider,  # None = auto-detect
            tenant_id=tenant_id,
            config=provider_config,  # Generic dict - provider interprets
        )

    def _create_tenant_provider(self, tenant_id: str) -> TracerProvider:
        """Create and configure TracerProvider for a tenant (legacy method)."""
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
            tracer_provider = provider.configure_span_export(
                endpoint=endpoint,
                project_name=project_name,
                use_batch_export=use_batch_export,
            )

            mode = "BATCH" if use_batch_export else "SYNC"
            logger.info(
                f"Created {mode} tracer provider: {project_name} (endpoint={endpoint})"
            )
            return tracer_provider

        except Exception as e:
            logger.error(f"Failed to create tracer provider for {project_key}: {e}")
            raise

    def _evict_old_tracers(self):
        """Evict old tracers using LRU policy."""
        if len(self._tenant_tracers) <= self.config.max_cached_tenants:
            return

        # Simple LRU - remove oldest entries
        # In production, you'd want a proper LRU cache
        items_to_remove = len(self._tenant_tracers) - self.config.max_cached_tenants
        oldest_keys = list(self._tenant_tracers.keys())[:items_to_remove]

        for key in oldest_keys:
            del self._tenant_tracers[key]
            logger.debug(f"Evicted tracer from cache: {key}")

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

    def unregister_project(self, tenant_id: str, project_name: str) -> None:
        """
        Unregister and shutdown specific project's tracer provider.

        Args:
            tenant_id: Tenant identifier
            project_name: Project name (e.g., "search", "synthetic_data")
        """
        project_key = f"{tenant_id}:{project_name}"

        with self._lock:
            # Shutdown and remove tracer provider
            if project_key in self._tenant_providers:
                provider = self._tenant_providers[project_key]
                try:
                    if hasattr(provider, "force_flush"):
                        provider.force_flush(timeout_millis=5000)
                    if hasattr(provider, "shutdown"):
                        provider.shutdown()
                    del self._tenant_providers[project_key]
                    logger.debug(f"Shutdown tracer provider for {project_key}")
                except Exception as e:
                    logger.error(f"Error shutting down provider for {project_key}: {e}")

            # Remove cached tracers for this project
            tracers_to_remove = [
                k for k in self._tenant_tracers if k.startswith(project_key)
            ]
            for k in tracers_to_remove:
                del self._tenant_tracers[k]
                logger.debug(f"Removed cached tracer: {k}")

            # Remove project config
            self._project_configs.pop(project_key, None)

            logger.info(f"Unregistered project {project_key}")


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


def get_telemetry_manager(
    config_manager=None, tenant_id: str = "default"
) -> TelemetryManager:
    """
    Get global telemetry manager instance.

    On first call, creates TelemetryManager with config loaded from ConfigManager.

    Args:
        config_manager: Optional ConfigManager to load config from.
            If None on first call, creates one via create_default_config_manager().
        tenant_id: Tenant ID for loading telemetry config.
    """
    global _telemetry_manager
    if _telemetry_manager is None:
        if config_manager is None:
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
            )

            config_manager = create_default_config_manager()
        config = config_manager.get_telemetry_config(tenant_id)
        _telemetry_manager = TelemetryManager(config)
    return _telemetry_manager
