"""
Backend Registry - Auto-discovery and registration of backend implementations.

This module provides a registry pattern for backends to self-register,
enabling a pluggable architecture where new backends can be added without
modifying core code.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Type

from cogniverse_sdk.interfaces.backend import Backend, IngestionBackend, SearchBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Central registry for backend auto-discovery and management.

    Backends self-register when imported, allowing the app layer
    to discover and use them without direct imports.
    """

    _instance = None
    _ingestion_backends: Dict[str, Type[IngestionBackend]] = {}
    _search_backends: Dict[str, Type[SearchBackend]] = {}
    _full_backends: Dict[str, Type[Backend]] = {}
    _backend_instances: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "BackendRegistry":
        """Get the singleton BackendRegistry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register_ingestion(
        cls, name: str, backend_class: Type[IngestionBackend]
    ) -> None:
        """
        Register an ingestion backend.

        Args:
            name: Backend name (e.g., "vespa", "milvus")
            backend_class: Backend class implementing IngestionBackend
        """
        cls._ingestion_backends[name] = backend_class
        logger.info(f"Registered ingestion backend: {name}")

    @classmethod
    def register_search(cls, name: str, backend_class: Type[SearchBackend]) -> None:
        """
        Register a search backend.

        Args:
            name: Backend name
            backend_class: Backend class implementing SearchBackend
        """
        cls._search_backends[name] = backend_class
        logger.info(f"Registered search backend: {name}")

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]) -> None:
        """
        Register a full backend (supports both ingestion and search).

        Args:
            name: Backend name
            backend_class: Backend class implementing Backend
        """
        cls._full_backends[name] = backend_class
        # Also register in individual registries
        cls._ingestion_backends[name] = backend_class
        cls._search_backends[name] = backend_class
        logger.info(f"Registered full backend: {name}")

    @classmethod
    def get_ingestion_backend(
        cls,
        name: str,
        tenant_id: str,
        config: Optional[Dict[str, Any]] = None,
        config_manager=None,
        schema_loader=None,
    ) -> IngestionBackend:
        """
        Get a tenant-specific ingestion backend instance.

        Args:
            name: Backend name
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Optional configuration for initialization
            config_manager: ConfigManager instance for dependency injection (REQUIRED)
            schema_loader: SchemaLoader instance for dependency injection (REQUIRED)

        Returns:
            Initialized IngestionBackend instance

        Raises:
            ValueError: If backend not found, tenant_id not provided, config_manager not provided, or schema_loader not provided

        Note:
            Backend instances are cached per tenant for isolation.
            Each tenant gets their own backend instance.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        if config_manager is None:
            raise ValueError("config_manager is required for backend initialization")

        if schema_loader is None:
            raise ValueError("schema_loader is required for backend initialization")

        # Cache key includes tenant_id for isolation
        if name in cls._full_backends:
            instance_key = f"backend_{name}_{tenant_id}"
        else:
            instance_key = f"ingestion_{name}_{tenant_id}"

        # Check if instance already exists
        if instance_key in cls._backend_instances:
            logger.debug(f"Returning cached backend: {instance_key}")
            return cls._backend_instances[instance_key]

        # Try to auto-import if not registered
        if name not in cls._ingestion_backends:
            cls._try_import_backend(name)

        if name not in cls._ingestion_backends:
            available = list(cls._ingestion_backends.keys())
            raise ValueError(
                f"Ingestion backend '{name}' not found. Available backends: {available}"
            )

        # Create BackendConfig from config_manager
        from cogniverse_core.factories.backend_factory import BackendFactory
        from cogniverse_foundation.config.unified_config import BackendConfig

        # Get system config for defaults
        system_config = config_manager.get_system_config(tenant_id)

        # Generic merge: Start with system config defaults, override with config["backend"] if provided
        backend_url = system_config.backend_url
        backend_port = system_config.backend_port
        backend_profiles = {}
        backend_metadata = {}

        if config and "backend" in config:
            backend_section = config["backend"]
            # Override any BackendConfig field if provided in config["backend"]
            backend_url = backend_section.get("url", backend_url)
            backend_port = backend_section.get("port", backend_port)
            backend_profiles = backend_section.get("profiles", backend_profiles)
            backend_metadata = backend_section.get("metadata", backend_metadata)

        # Create BackendConfig object with merged values
        backend_config_obj = BackendConfig(
            tenant_id=tenant_id,
            backend_type=name,
            url=backend_url,
            port=backend_port,
            profiles=backend_profiles,
            metadata=backend_metadata,
        )

        # Build backend initialization config - merge top-level keys with backend section
        backend_init_config = {"tenant_id": tenant_id}

        if config:
            # Merge all top-level config keys into backend initialization config
            backend_init_config.update(config)

            # Preserve nested backend section for backend-specific configuration
            # This allows backend-specific config to override top-level
            if "backend" in config:
                backend_init_config["backend"] = config["backend"]

        # Use factory to create backend with all dependencies properly initialized
        # Factory handles: backend creation, SchemaRegistry creation, injection, and initialize()
        backend_class = cls._ingestion_backends[name]

        instance = BackendFactory.create_backend_with_dependencies(
            backend_class=backend_class,
            backend_config=backend_config_obj,
            config_manager=config_manager,
            schema_loader=schema_loader,
            backend_init_config=backend_init_config,
        )

        # Cache instance with tenant_id
        cls._backend_instances[instance_key] = instance
        logger.info(f"Created and cached backend: {instance_key}")

        return instance

    @classmethod
    def get_search_backend(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        config_manager=None,
        schema_loader=None,
    ) -> SearchBackend:
        """
        Get a shared search backend instance.

        Search backends are shared across all tenants. Tenant isolation is
        achieved by passing tenant_id in query_dict at search() time, where
        it is used for schema name derivation.

        Args:
            name: Backend name
            config: Optional configuration for initialization
            config_manager: ConfigManager instance for dependency injection (REQUIRED)
            schema_loader: SchemaLoader instance for dependency injection (REQUIRED)

        Returns:
            Initialized SearchBackend instance

        Raises:
            ValueError: If backend not found, config_manager not provided, or schema_loader not provided
        """
        if config_manager is None:
            raise ValueError("config_manager is required for backend initialization")

        if schema_loader is None:
            raise ValueError("schema_loader is required for backend initialization")

        # Search backends are shared — cache key has no tenant suffix
        instance_key = f"search_{name}"

        # Check if instance already exists
        if instance_key in cls._backend_instances:
            logger.debug(f"Returning cached backend: {instance_key}")
            return cls._backend_instances[instance_key]

        # Try to auto-import if not registered
        if name not in cls._search_backends:
            cls._try_import_backend(name)

        if name not in cls._search_backends:
            available = list(cls._search_backends.keys())
            raise ValueError(
                f"Search backend '{name}' not found. Available backends: {available}"
            )

        # Create BackendConfig from config_manager
        from cogniverse_core.factories.backend_factory import BackendFactory
        from cogniverse_foundation.config.unified_config import BackendConfig

        # Get system config for defaults (URL, port) using the default tenant
        system_config = config_manager.get_system_config("default")

        # Generic merge: Start with system config defaults, override with config["backend"] if provided
        backend_url = system_config.backend_url
        backend_port = system_config.backend_port
        backend_profiles = {}
        backend_metadata = {}

        if config and "backend" in config:
            backend_section = config["backend"]
            # Override any BackendConfig field if provided in config["backend"]
            backend_url = backend_section.get("url", backend_url)
            backend_port = backend_section.get("port", backend_port)
            backend_profiles = backend_section.get("profiles", backend_profiles)
            backend_metadata = backend_section.get("metadata", backend_metadata)

        # Also check top-level config for direct overrides (takes precedence over backend section)
        if config:
            backend_url = config.get("url", backend_url)
            backend_port = config.get("port", backend_port)
            if "profiles" in config:
                backend_profiles = config["profiles"]
            if "metadata" in config:
                backend_metadata = config["metadata"]

        # Create BackendConfig object with merged values
        backend_config_obj = BackendConfig(
            tenant_id="default",
            backend_type=name,
            url=backend_url,
            port=backend_port,
            profiles=backend_profiles,
            metadata=backend_metadata,
        )

        # Build backend initialization config
        backend_init_config = {}

        if config:
            # Merge all top-level config keys into backend initialization config
            backend_init_config.update(config)

            # Preserve nested backend section for backend-specific configuration
            # This allows backend-specific config to override top-level
            if "backend" in config:
                backend_init_config["backend"] = config["backend"]

        # Use factory to create backend with all dependencies properly initialized
        # Factory handles: backend creation, SchemaRegistry creation, injection, and initialize()
        backend_class = cls._search_backends[name]
        instance = BackendFactory.create_backend_with_dependencies(
            backend_class=backend_class,
            backend_config=backend_config_obj,
            config_manager=config_manager,
            schema_loader=schema_loader,
            backend_init_config=backend_init_config,
        )

        # Cache shared instance
        cls._backend_instances[instance_key] = instance
        logger.info(f"Created and cached shared search backend: {instance_key}")

        return instance

    @classmethod
    def _try_import_backend(cls, name: str) -> None:
        """
        Try to import a backend module to trigger self-registration.

        Args:
            name: Backend name to import
        """
        module_path = f"cogniverse_{name}.backend"
        try:
            importlib.import_module(module_path)
            logger.info(f"Auto-imported backend module: {module_path}")
        except ImportError as e:
            logger.debug(f"Could not auto-import backend module {module_path}: {e}")

    @classmethod
    def list_ingestion_backends(cls) -> list:
        """List all registered ingestion backends."""
        return list(cls._ingestion_backends.keys())

    @classmethod
    def list_search_backends(cls) -> list:
        """List all registered search backends."""
        return list(cls._search_backends.keys())

    @classmethod
    def list_full_backends(cls) -> list:
        """List all registered full backends."""
        return list(cls._full_backends.keys())

    @classmethod
    def list_backends(cls) -> list:
        """List all registered backends (de-duplicated across all types)."""
        all_backends = set()
        all_backends.update(cls._ingestion_backends.keys())
        all_backends.update(cls._search_backends.keys())
        all_backends.update(cls._full_backends.keys())
        return list(all_backends)

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached backend instances."""
        cls._backend_instances.clear()
        logger.info("Cleared all backend instances")

    @classmethod
    def is_registered(cls, name: str, backend_type: str = "any") -> bool:
        """
        Check if a backend is registered.

        Args:
            name: Backend name
            backend_type: Type to check ("ingestion", "search", "full", "any")

        Returns:
            True if registered, False otherwise
        """
        if backend_type == "ingestion":
            return name in cls._ingestion_backends
        elif backend_type == "search":
            return name in cls._search_backends
        elif backend_type == "full":
            return name in cls._full_backends
        else:  # "any"
            return (
                name in cls._ingestion_backends
                or name in cls._search_backends
                or name in cls._full_backends
            )


# Global registry instance
_backend_registry = BackendRegistry()


def get_backend_registry() -> BackendRegistry:
    """Get the global BackendRegistry instance."""
    return _backend_registry


def register_ingestion_backend(
    name: str, backend_class: Type[IngestionBackend]
) -> None:
    """
    Convenience function to register an ingestion backend.

    Args:
        name: Backend name
        backend_class: Backend class implementing IngestionBackend
    """
    _backend_registry.register_ingestion(name, backend_class)


def register_search_backend(name: str, backend_class: Type[SearchBackend]) -> None:
    """
    Convenience function to register a search backend.

    Args:
        name: Backend name
        backend_class: Backend class implementing SearchBackend
    """
    _backend_registry.register_search(name, backend_class)


def register_backend(name: str, backend_class: Type[Backend]) -> None:
    """
    Convenience function to register a full backend.

    Args:
        name: Backend name
        backend_class: Backend class implementing Backend
    """
    _backend_registry.register_backend(name, backend_class)
