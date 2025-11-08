"""
Backend Registry - Auto-discovery and registration of backend implementations.

This module provides a registry pattern for backends to self-register,
enabling a pluggable architecture where new backends can be added without
modifying core code.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Type

from cogniverse_core.interfaces.backend import Backend, IngestionBackend, SearchBackend

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
        cls, name: str, tenant_id: str, config: Optional[Dict[str, Any]] = None, config_manager=None, schema_loader=None
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
                f"Ingestion backend '{name}' not found. "
                f"Available backends: {available}"
            )

        # Create BackendConfig from config_manager
        from cogniverse_core.config.unified_config import BackendConfig

        # Get system config for URL and port
        system_config = config_manager.get_system_config(tenant_id)

        # Create BackendConfig object (required by new VespaBackend signature)
        backend_config_obj = BackendConfig(
            tenant_id=tenant_id,
            backend_type=name,  # e.g., "vespa"
            url=system_config.backend_url,
            port=system_config.backend_port,
        )

        # Create backend instance with BackendConfig
        backend_class = cls._ingestion_backends[name]
        instance = backend_class(backend_config=backend_config_obj, schema_loader=schema_loader, config_manager=config_manager)

        # Create and inject SchemaRegistry (fixes circular dependency)
        from cogniverse_core.registries.schema_registry import SchemaRegistry
        schema_registry = SchemaRegistry(
            config_manager=config_manager,
            backend=instance,
            schema_loader=schema_loader
        )
        instance.schema_registry = schema_registry

        # Build backend config dict for initialize() - merge top-level keys with backend section
        # Start with tenant_id, then add all top-level config keys
        backend_init_config = {"tenant_id": tenant_id}

        if config:
            # Merge all top-level config keys into backend initialization config
            backend_init_config.update(config)

            # Preserve nested backend section for backend-specific configuration
            # This allows backend-specific config to override top-level
            if "backend" in config:
                backend_init_config["backend"] = config["backend"]

        instance.initialize(backend_init_config)

        # CRITICAL FIX: Inject schema_registry into schema_manager AFTER initialize()
        # VespaSchemaManager is created during initialize() and needs schema_registry
        # for tenant_schema_exists() method
        if hasattr(instance, 'schema_manager') and instance.schema_manager:
            instance.schema_manager._schema_registry = schema_registry

        # Cache instance with tenant_id
        cls._backend_instances[instance_key] = instance
        logger.info(f"Created and cached backend: {instance_key}")

        return instance

    @classmethod
    def get_search_backend(
        cls, name: str, tenant_id: str, config: Optional[Dict[str, Any]] = None, config_manager=None, schema_loader=None
    ) -> SearchBackend:
        """
        Get a tenant-specific search backend instance.

        Args:
            name: Backend name
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Optional configuration for initialization
            config_manager: ConfigManager instance for dependency injection (REQUIRED)
            schema_loader: SchemaLoader instance for dependency injection (REQUIRED)

        Returns:
            Initialized SearchBackend instance

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
        # For full backends, use same key as ingestion to share instance
        if name in cls._full_backends:
            instance_key = f"backend_{name}_{tenant_id}"
        else:
            instance_key = f"search_{name}_{tenant_id}"

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
                f"Search backend '{name}' not found. "
                f"Available backends: {available}"
            )

        # Create BackendConfig from config_manager
        from cogniverse_core.config.unified_config import BackendConfig

        # Get system config for URL and port
        system_config = config_manager.get_system_config(tenant_id)

        # Create BackendConfig object (required by new VespaBackend signature)
        backend_config_obj = BackendConfig(
            tenant_id=tenant_id,
            backend_type=name,  # e.g., "vespa"
            url=system_config.backend_url,
            port=system_config.backend_port,
        )

        # Create backend instance with BackendConfig
        backend_class = cls._search_backends[name]
        instance = backend_class(backend_config=backend_config_obj, schema_loader=schema_loader, config_manager=config_manager)

        # Create and inject SchemaRegistry (fixes circular dependency)
        from cogniverse_core.registries.schema_registry import SchemaRegistry
        schema_registry = SchemaRegistry(
            config_manager=config_manager,
            backend=instance,
            schema_loader=schema_loader
        )
        instance.schema_registry = schema_registry

        # Build backend config dict for initialize() - merge top-level keys with backend section
        # Start with tenant_id, then add all top-level config keys
        backend_init_config = {"tenant_id": tenant_id}

        if config:
            # Merge all top-level config keys into backend initialization config
            backend_init_config.update(config)

            # Preserve nested backend section for backend-specific configuration
            # This allows backend-specific config to override top-level
            if "backend" in config:
                backend_init_config["backend"] = config["backend"]

        instance.initialize(backend_init_config)

        # Cache instance with tenant_id
        cls._backend_instances[instance_key] = instance
        logger.info(f"Created and cached backend: {instance_key}")

        return instance

    @classmethod
    def _try_import_backend(cls, name: str) -> None:
        """
        Try to import a backend module to trigger self-registration.

        Args:
            name: Backend name to import
        """
        try:
            # Try standard backend location
            module_path = f"src.backends.{name}"
            importlib.import_module(module_path)
            logger.info(f"Auto-imported backend module: {module_path}")
        except ImportError:
            # Try with underscore (e.g., "vespa_backend")
            try:
                module_path = f"src.backends.{name}_backend"
                importlib.import_module(module_path)
                logger.info(f"Auto-imported backend module: {module_path}")
            except ImportError:
                logger.debug(f"Could not auto-import backend: {name}")

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
