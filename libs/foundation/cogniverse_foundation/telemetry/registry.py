"""
Telemetry provider registry with entry points discovery.

Providers self-register via setuptools entry points (like Backend pattern).
Core auto-discovers installed providers without hardcoded imports.
"""

import importlib.metadata
import logging
import os
from typing import Any, Dict, List, Optional, Type

from cogniverse_foundation.caching import TenantLRUCache
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


_DEFAULT_TENANT_CACHE_CAPACITY = 16


def _tenant_cache_capacity() -> int:
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "COGNIVERSE_TENANT_CACHE_CAPACITY",
                    _DEFAULT_TENANT_CACHE_CAPACITY,
                )
            ),
        )
    except (TypeError, ValueError):
        return _DEFAULT_TENANT_CACHE_CAPACITY


def _on_provider_evicted(key: str, instance: TelemetryProvider) -> None:
    """Close a provider's underlying transport when evicted from the LRU.

    Phoenix/OTLP providers hold a gRPC channel plus an HTTP client. If
    they expose ``close()`` or ``shutdown()``, call it so we release the
    sockets instead of relying on GC to chase them down.
    """
    for method in ("close", "shutdown"):
        closer = getattr(instance, method, None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                logger.debug("Provider %s.%s() failed: %s", key, method, exc)
            return
    logger.debug("Evicted telemetry provider %s (no close method)", key)


class TelemetryRegistry:
    """
    Registry for auto-discovering telemetry providers via entry points.

    Providers register via setuptools entry points:
        [project.entry-points."cogniverse.telemetry.providers"]
        phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"

    Lazy discovery on first get_telemetry_provider() call (~5-10ms overhead).
    """

    _instance = None
    _providers: Dict[str, Type[TelemetryProvider]] = {}
    # LRU-bounded per-tenant cache. Each Phoenix/OTLP instance keeps a
    # gRPC channel + HTTP client open; an unbounded dict would leak one
    # of each per test tenant for the life of the process.
    _provider_instances: TenantLRUCache[TelemetryProvider] = TenantLRUCache(
        capacity=_tenant_cache_capacity(),
        on_evict=_on_provider_evicted,
    )
    _entry_points_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def discover_providers(cls) -> None:
        """
        Auto-discover providers via entry points.

        Scans installed packages for entry points in group
        'cogniverse.telemetry.providers'. Called lazily on first
        get_telemetry_provider() call.

        Raises:
            ValueError: If duplicate provider names found from different packages
        """
        if cls._entry_points_loaded:
            return

        logger.info("Discovering telemetry providers via entry points...")

        try:
            # Python 3.10+ API
            entry_points = importlib.metadata.entry_points(
                group="cogniverse.telemetry.providers"
            )
        except TypeError:
            # Python 3.9 fallback
            entry_points = importlib.metadata.entry_points().get(
                "cogniverse.telemetry.providers", []
            )

        for entry_point in entry_points:
            name = entry_point.name

            # Check for conflicts (same name from different packages)
            if name in cls._providers:
                existing_module = cls._providers[name].__module__
                new_module = entry_point.value.split(":")[0]

                if existing_module != new_module:
                    raise ValueError(
                        f"Conflict: Telemetry provider '{name}' registered by multiple packages:\n"
                        f"  - {existing_module}\n"
                        f"  - {new_module}\n"
                        f"Only one can be installed. Uninstall one package."
                    )
                logger.debug(
                    f"Provider '{name}' already registered, skipping duplicate entry point"
                )
                continue

            try:
                provider_class = entry_point.load()
                cls._providers[name] = provider_class
                logger.info(
                    f"Discovered telemetry provider: {name} ({entry_point.value})"
                )
            except Exception as e:
                logger.error(f"Failed to load telemetry provider '{name}': {e}")

        cls._entry_points_loaded = True

        if not cls._providers:
            logger.warning(
                "No telemetry providers discovered. "
                "Install cogniverse-telemetry-phoenix or another provider package."
            )
        else:
            logger.info(f"Telemetry providers available: {list(cls._providers.keys())}")

    @classmethod
    def get_telemetry_provider(
        cls,
        name: Optional[str] = None,
        tenant_id: str = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TelemetryProvider:
        """
        Get tenant-specific telemetry provider instance (cached per tenant).

        Args:
            name: Provider name (e.g., "phoenix", "langsmith").
                  If None, uses first available provider (fallback mode).
            tenant_id: Tenant identifier (required — per-tenant Phoenix
                project / per-tenant LangSmith scope). Pass
                ``SYSTEM_TENANT_ID`` explicitly for cluster-wide startup
                telemetry.
            config: Generic configuration dictionary (provider interprets)

        Returns:
            Initialized TelemetryProvider instance

        Raises:
            ValueError: If no providers available, specified provider not
                found, or tenant_id not supplied.

        Example:
            # Explicit selection
            provider = registry.get_telemetry_provider(
                name="phoenix",
                tenant_id="customer-123",
                config={"http_endpoint": "http://localhost:6006"}
            )

            # Fallback (auto-detect first installed provider)
            provider = registry.get_telemetry_provider(
                name=None,
                tenant_id="customer-123"
            )
        """
        from cogniverse_core.common.tenant_utils import require_tenant_id

        tenant_id = require_tenant_id(tenant_id, source="get_telemetry_provider")

        # Lazy discovery on first call
        if not cls._entry_points_loaded:
            cls.discover_providers()

        # Fallback: use first available provider if name not specified
        if name is None:
            if not cls._providers:
                raise ValueError(
                    "No telemetry providers installed. "
                    "Install cogniverse-telemetry-phoenix or another provider package:\n"
                    "  pip install cogniverse-telemetry-phoenix"
                )
            name = list(cls._providers.keys())[0]
            logger.info(
                f"No provider specified in config, using first available: {name}"
            )

        # Check if provider exists
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Telemetry provider '{name}' not found. "
                f"Available providers: {available or 'none'}\n"
                f"Install provider package: pip install cogniverse-telemetry-{name}"
            )

        # Check cache (per-tenant instances for isolation)
        instance_key = f"{name}_{tenant_id}"
        cached = cls._provider_instances.get(instance_key)
        if cached is not None:
            logger.debug(f"Returning cached telemetry provider: {instance_key}")
            return cached

        # Create and initialize new instance
        provider_class = cls._providers[name]
        instance = provider_class()

        # Build config with tenant_id
        provider_config = {"tenant_id": tenant_id}
        if config:
            provider_config.update(config)

        # Provider validates and interprets config keys
        instance.initialize(provider_config)

        # Cache instance per tenant (LRU eviction if over capacity)
        cls._provider_instances.set(instance_key, instance)
        logger.info(f"Created telemetry provider: {instance_key}")

        return instance

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all discovered providers.

        Returns:
            List of provider names
        """
        if not cls._entry_points_loaded:
            cls.discover_providers()
        return list(cls._providers.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear cached provider instances.

        Useful for testing or forcing provider re-initialization.
        """
        cls._provider_instances.clear()
        logger.info("Cleared telemetry provider cache")

    @classmethod
    def is_provider_available(cls, name: str) -> bool:
        """
        Check if a provider is available.

        Args:
            name: Provider name

        Returns:
            True if provider is installed and registered
        """
        if not cls._entry_points_loaded:
            cls.discover_providers()
        return name in cls._providers


# Global registry instance
_telemetry_registry = TelemetryRegistry()


def get_telemetry_registry() -> TelemetryRegistry:
    """Get the global TelemetryRegistry instance."""
    return _telemetry_registry
