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
            
        self.config = config or TelemetryConfig.from_env()
        self.config.validate()
        
        # Thread-safe caches
        self._tenant_providers: Dict[str, TracerProvider] = {}
        self._tenant_tracers: Dict[str, Tracer] = {}
        self._lock = threading.RLock()
        
        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._failed_initializations = 0
        
        self._initialized = True
        logger.info(f"TelemetryManager initialized with config: {self.config}")
    
    def get_tracer(self, tenant_id: str, service_name: Optional[str] = None) -> Optional[Tracer]:
        """
        Get tracer for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier for project isolation
            service_name: Optional service name override
            
        Returns:
            Tracer instance or None if telemetry disabled/failed
        """
        if not self.config.enabled:
            return None
            
        service_name = service_name or self.config.service_name
        cache_key = f"{tenant_id}:{service_name}"
        
        # Check cache first
        with self._lock:
            if cache_key in self._tenant_tracers:
                self._cache_hits += 1
                return self._tenant_tracers[cache_key]
            
            self._cache_misses += 1
            
            # Create tracer provider for tenant if needed
            try:
                if tenant_id not in self._tenant_providers:
                    self._tenant_providers[tenant_id] = self._create_tenant_provider(tenant_id)
                
                tracer_provider = self._tenant_providers[tenant_id]
                tracer = tracer_provider.get_tracer(service_name)
                
                # Cache with LRU eviction
                self._tenant_tracers[cache_key] = tracer
                self._evict_old_tracers()
                
                return tracer
                
            except Exception as e:
                self._failed_initializations += 1
                logger.warning(f"Failed to create tracer for tenant {tenant_id}: {e}")
                return None
    
    @contextmanager
    def span(self,
             name: str,
             tenant_id: str,
             service_name: Optional[str] = None,
             attributes: Optional[Dict[str, Any]] = None,
             use_routing_project: bool = False):
        """
        Context manager for creating tenant-specific spans.

        Args:
            name: Span name
            tenant_id: Tenant identifier
            service_name: Optional service name
            attributes: Optional span attributes
            use_routing_project: If True, create span in routing optimization project

        Usage:
            # Regular service span
            tenant_id = context.get("tenant_id", config.default_tenant_id)
            with telemetry.span("search", tenant_id=tenant_id) as span:
                span.set_attribute("query", "test")

            # For routing optimization spans
            with telemetry.span("cogniverse.routing", tenant_id=tenant_id, use_routing_project=True) as span:
                span.set_attribute("routing.chosen_agent", "video_search")
        """
        # Determine which project to use
        if use_routing_project:
            # Use special routing optimization project
            project_key = f"{tenant_id}:routing-optimization"
        else:
            project_key = f"{tenant_id}:{service_name or self.config.service_name}"

        tracer = self._get_tracer_for_project(tenant_id, service_name, use_routing_project)

        if tracer is None:
            # Graceful degradation - yield no-op span
            yield NoOpSpan()
            return

        with tracer.start_as_current_span(name) as span:
            # Add tenant context to all spans
            span.set_attribute("tenant.id", tenant_id)
            span.set_attribute("service.name", service_name or self.config.service_name)
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

    def _get_tracer_for_project(self, tenant_id: str, service_name: Optional[str], use_routing_project: bool) -> Optional[Tracer]:
        """Get tracer for specific project (routing optimization or regular service)."""
        if not self.config.enabled:
            return None

        # Determine project key and name
        if use_routing_project:
            cache_key = f"{tenant_id}:routing-optimization"
            project_suffix = "routing-optimization"
        else:
            service_name = service_name or self.config.service_name
            cache_key = f"{tenant_id}:{service_name}"
            project_suffix = service_name

        # Check cache first
        with self._lock:
            if cache_key in self._tenant_tracers:
                self._cache_hits += 1
                return self._tenant_tracers[cache_key]

            self._cache_misses += 1

            # Create tracer provider for project if needed
            try:
                provider_key = f"{tenant_id}:{project_suffix}"
                if provider_key not in self._tenant_providers:
                    self._tenant_providers[provider_key] = self._create_tenant_provider_for_project(
                        tenant_id, project_suffix, use_routing_project
                    )

                tracer_provider = self._tenant_providers[provider_key]
                tracer = tracer_provider.get_tracer(project_suffix)

                # Cache with LRU eviction
                self._tenant_tracers[cache_key] = tracer
                self._evict_old_tracers()

                return tracer

            except Exception as e:
                self._failed_initializations += 1
                logger.warning(f"Failed to create tracer for {cache_key}: {e}")
                return None
    
    def _create_tenant_provider(self, tenant_id: str) -> TracerProvider:
        """Create and configure TracerProvider for a tenant (legacy method)."""
        return self._create_tenant_provider_for_project(tenant_id, self.config.service_name, use_routing_project=False)

    def _create_tenant_provider_for_project(self, tenant_id: str, project_suffix: str, use_routing_project: bool) -> TracerProvider:
        """Create and configure TracerProvider for a specific tenant project."""
        if not self.config.phoenix_enabled:
            raise RuntimeError("Phoenix not enabled")

        try:
            # Determine project name based on type
            if use_routing_project:
                project_name = self.config.get_routing_optimization_project_name(tenant_id)
            else:
                project_name = self.config.get_project_name(tenant_id, project_suffix)

            # For test mode with synchronous export, create provider manually
            if self.config.batch_config.use_sync_export:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                from opentelemetry.sdk.resources import Resource

                # Create OTLP exporter pointing to Phoenix
                endpoint = self.config.phoenix_endpoint
                if not endpoint.startswith("http"):
                    endpoint = f"http://{endpoint}"

                exporter = OTLPSpanExporter(
                    endpoint=endpoint,
                    insecure=not self.config.phoenix_use_tls
                )

                # Create resource with project name using OpenInference semantic convention
                resource = Resource.create({
                    "service.name": project_suffix,
                    "openinference.project.name": project_name,  # OpenInference semantic convention
                    "telemetry.sdk.name": "opentelemetry",
                    "telemetry.sdk.language": "python",
                })

                # Create tracer provider with synchronous processor
                tracer_provider = TracerProvider(resource=resource)
                tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

                logger.info(f"Created SYNC tracer provider for tenant {tenant_id} -> project {project_name}")
                return tracer_provider

            # Production mode: use Phoenix register with batch export
            from phoenix.otel import register

            tracer_provider = register(
                project_name=project_name,
                batch=True,
                auto_instrument=False,
                set_global_tracer_provider=False
            )

            logger.info(f"Created BATCH tracer provider for tenant {tenant_id} -> project {project_name}")
            return tracer_provider

        except Exception as e:
            logger.error(f"Failed to create tracer provider for tenant {tenant_id}, project {project_suffix}: {e}")
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
                "environment": self.config.environment
            }
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
                    if hasattr(provider, 'force_flush'):
                        success = provider.force_flush(timeout_millis=timeout_millis)
                        if not success:
                            logger.warning(f"Force flush failed for tenant {tenant_id}")
                            all_success = False
                        else:
                            logger.info(f"Successfully flushed spans for tenant {tenant_id}")
                except Exception as e:
                    logger.error(f"Error flushing spans for tenant {tenant_id}: {e}")
                    all_success = False
        return all_success

    def shutdown(self):
        """Shutdown all tracer providers gracefully."""
        with self._lock:
            for tenant_id, provider in self._tenant_providers.items():
                try:
                    if hasattr(provider, 'force_flush'):
                        provider.force_flush(timeout_millis=5000)
                    if hasattr(provider, 'shutdown'):
                        provider.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down provider for tenant {tenant_id}: {e}")

            self._tenant_providers.clear()
            self._tenant_tracers.clear()


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


def get_telemetry_manager() -> TelemetryManager:
    """Get global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager
