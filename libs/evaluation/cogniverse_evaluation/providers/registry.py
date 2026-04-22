"""
Evaluation provider registry with entry points discovery.

Providers self-register via setuptools entry points (like telemetry providers).
Core auto-discovers installed providers without hardcoded imports.
"""

import importlib.metadata
import logging
import os
from typing import Any, Dict, List, Optional, Type

from cogniverse_evaluation.providers.base import EvaluationProvider
from cogniverse_foundation.caching import TenantLRUCache

logger = logging.getLogger(__name__)


_DEFAULT_TENANT_CACHE_CAPACITY = 16


def _tenant_cache_capacity() -> int:
    try:
        return max(1, int(os.environ.get(
            "COGNIVERSE_TENANT_CACHE_CAPACITY",
            _DEFAULT_TENANT_CACHE_CAPACITY,
        )))
    except (TypeError, ValueError):
        return _DEFAULT_TENANT_CACHE_CAPACITY


def _on_provider_evicted(key: str, instance: EvaluationProvider) -> None:
    for method in ("close", "shutdown"):
        closer = getattr(instance, method, None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                logger.debug("Evaluation provider %s.%s() failed: %s", key, method, exc)
            return
    logger.debug("Evicted evaluation provider %s (no close method)", key)


class EvaluationRegistry:
    """
    Registry for auto-discovering evaluation providers via entry points.

    Providers register via setuptools entry points:
        [project.entry-points."cogniverse.evaluation.providers"]
        phoenix = "cogniverse_telemetry_phoenix.evaluation:PhoenixEvaluationProvider"
        langsmith = "cogniverse_langsmith.evaluation:LangsmithEvaluationProvider"

    Lazy discovery on first get_evaluation_provider() call (~5-10ms overhead).
    """

    _instance = None
    _providers: Dict[str, Type[EvaluationProvider]] = {}
    _provider_instances: TenantLRUCache[EvaluationProvider] = TenantLRUCache(
        capacity=_tenant_cache_capacity(),
        on_evict=_on_provider_evicted,
    )
    _entry_points_loaded = False
    _default_provider: Optional[EvaluationProvider] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def discover_providers(cls) -> None:
        """
        Auto-discover providers via entry points.

        Scans installed packages for entry points in group
        'cogniverse.evaluation.providers'. Called lazily on first
        get_evaluation_provider() call.

        Raises:
            ValueError: If duplicate provider names found from different packages
        """
        if cls._entry_points_loaded:
            return

        logger.info("Discovering evaluation providers via entry points...")

        try:
            # Python 3.10+ API
            entry_points = importlib.metadata.entry_points(
                group="cogniverse.evaluation.providers"
            )
        except TypeError:
            # Python 3.9 fallback
            entry_points = importlib.metadata.entry_points().get(
                "cogniverse.evaluation.providers", []
            )

        for entry_point in entry_points:
            name = entry_point.name

            # Check for conflicts (same name from different packages)
            if name in cls._providers:
                existing_module = cls._providers[name].__module__
                new_module = entry_point.value.split(":")[0]

                if existing_module != new_module:
                    raise ValueError(
                        f"Conflict: Evaluation provider '{name}' registered by multiple packages:\n"
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
                    f"Discovered evaluation provider: {name} ({entry_point.value})"
                )
            except Exception as e:
                logger.error(f"Failed to load evaluation provider '{name}': {e}")

        cls._entry_points_loaded = True

        if not cls._providers:
            logger.warning(
                "No evaluation providers discovered. "
                "Install cogniverse-telemetry-phoenix or another provider package."
            )
        else:
            logger.info(
                f"Evaluation providers available: {list(cls._providers.keys())}"
            )

    @classmethod
    def get_evaluation_provider(
        cls,
        tenant_id: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationProvider:
        """
        Get tenant-specific evaluation provider instance (cached per tenant).

        Args:
            name: Provider name (e.g., "phoenix", "langsmith").
                  If None, uses first available provider (fallback mode).
            tenant_id: Tenant identifier (required for multi-tenancy)
            config: Generic configuration dictionary (provider interprets)

        Returns:
            Initialized EvaluationProvider instance

        Raises:
            ValueError: If no providers available or specified provider not found

        Example:
            # Explicit selection
            provider = registry.get_evaluation_provider(
                name="phoenix",
                tenant_id="customer-123",
                config={"http_endpoint": "http://localhost:6006"}
            )

            # Fallback (auto-detect first installed provider)
            provider = registry.get_evaluation_provider(
                name=None,
                tenant_id="customer-123"
            )
        """
        # Lazy discovery on first call
        if not cls._entry_points_loaded:
            cls.discover_providers()

        # Fallback: use first available provider if name not specified
        if name is None:
            if not cls._providers:
                raise ValueError(
                    "No evaluation providers installed. "
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
                f"Evaluation provider '{name}' not found. "
                f"Available providers: {available or 'none'}\n"
                f"Install provider package: pip install cogniverse-telemetry-{name}"
            )

        # Check cache (per-tenant instances for isolation)
        instance_key = f"{name}_{tenant_id}"
        cached = cls._provider_instances.get(instance_key)
        if cached is not None:
            logger.debug(f"Returning cached evaluation provider: {instance_key}")
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
        logger.info(f"Created evaluation provider: {instance_key}")

        return instance

    @classmethod
    def set_default_provider(cls, provider: EvaluationProvider) -> None:
        """
        Set the default evaluation provider for convenience methods.

        Args:
            provider: Initialized EvaluationProvider instance
        """
        cls._default_provider = provider
        logger.info(f"Set default evaluation provider: {type(provider).__name__}")

    @classmethod
    def get_default_provider(cls) -> EvaluationProvider:
        """
        Get the default evaluation provider.

        Returns:
            Default EvaluationProvider instance

        Raises:
            ValueError: If no default provider has been set
        """
        if cls._default_provider is None:
            # Try to auto-initialize with first available provider
            if not cls._entry_points_loaded:
                cls.discover_providers()

            if cls._providers:
                logger.info("Auto-initializing default evaluation provider")
                cls._default_provider = cls.get_evaluation_provider()
            else:
                raise ValueError(
                    "No default evaluation provider set. "
                    "Call set_evaluation_provider() first or install a provider package."
                )

        return cls._default_provider

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
        cls._default_provider = None
        logger.info("Cleared evaluation provider cache")

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
_evaluation_registry = EvaluationRegistry()


def get_evaluation_registry() -> EvaluationRegistry:
    """Get the global EvaluationRegistry instance."""
    return _evaluation_registry


def get_evaluation_provider(
    tenant_id: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> EvaluationProvider:
    """
    Convenience function to get evaluation provider.

    When ``tenant_id`` is omitted, lookup runs under the cluster-wide
    ``SYSTEM_TENANT_ID``. This keeps framework-level callers (that only
    read static provider metadata such as the evaluator base class) from
    having to fabricate a tenant id; callers doing per-tenant work must
    still pass their tenant explicitly.

    See EvaluationRegistry.get_evaluation_provider() for details.
    """
    if tenant_id is None:
        from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

        tenant_id = SYSTEM_TENANT_ID
    return _evaluation_registry.get_evaluation_provider(
        name=name, tenant_id=tenant_id, config=config
    )


def set_evaluation_provider(provider: EvaluationProvider) -> None:
    """
    Convenience function to set default evaluation provider.

    Args:
        provider: Initialized EvaluationProvider instance
    """
    _evaluation_registry.set_default_provider(provider)


def register_evaluation_provider(
    name: str, provider_class: Type[EvaluationProvider]
) -> None:
    """
    Manually register an evaluation provider (for testing).

    Args:
        name: Provider name
        provider_class: EvaluationProvider class (not instance)
    """
    _evaluation_registry._providers[name] = provider_class
    logger.info(f"Manually registered evaluation provider: {name}")


def reset_evaluation_provider() -> None:
    """
    Reset evaluation provider cache and default provider.

    Useful for testing or forcing provider re-initialization.
    """
    _evaluation_registry.clear_cache()
    logger.info("Reset evaluation provider")
