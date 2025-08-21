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
from typing import Dict, Optional, Any
from functools import lru_cache
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Tracer, Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

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
             attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for creating tenant-specific spans.
        
        Args:
            name: Span name
            tenant_id: Tenant identifier
            service_name: Optional service name
            attributes: Optional span attributes
            
        Usage:
            with telemetry.span("search", tenant_id="tenant-123") as span:
                span.set_attribute("query", "test")
        """
        tracer = self.get_tracer(tenant_id, service_name)
        
        if tracer is None:
            # Graceful degradation - yield no-op span
            yield NoOpSpan()
            return
        
        try:
            with tracer.start_as_current_span(name) as span:
                # Add tenant context to all spans
                span.set_attribute("tenant.id", tenant_id)
                span.set_attribute("service.name", service_name or self.config.service_name)
                span.set_attribute("environment", self.config.environment)
                
                # Add user-provided attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                yield span
                
        except Exception as e:
            logger.warning(f"Error in span {name} for tenant {tenant_id}: {e}")
            yield NoOpSpan()
    
    def _create_tenant_provider(self, tenant_id: str) -> TracerProvider:
        """Create and configure TracerProvider for a tenant."""
        if not self.config.phoenix_enabled:
            raise RuntimeError("Phoenix not enabled")
        
        try:
            # Use Phoenix register for project creation with batch export config
            from phoenix.otel import register
            from .exporter import QueueManagedBatchExporter
            
            project_name = self.config.get_project_name(tenant_id)
            
            # Create tracer provider with Phoenix but get the underlying exporter
            tracer_provider = register(
                project_name=project_name,
                batch=True,
                auto_instrument=False,  # We handle instrumentation manually
                set_global_tracer_provider=False  # Don't override global
            )
            
            # Wrap the Phoenix exporter with our queue-managed one
            # This is a more advanced pattern - for now we'll use Phoenix's built-in batching
            # In production, you might want to replace Phoenix's exporter with QueueManagedBatchExporter
            
            logger.info(f"Created tracer provider for tenant {tenant_id} -> project {project_name}")
            return tracer_provider
            
        except Exception as e:
            logger.error(f"Failed to create tracer provider for tenant {tenant_id}: {e}")
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