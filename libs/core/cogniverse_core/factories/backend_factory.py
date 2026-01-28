"""
Factory for creating backend instances with proper dependency injection.

This factory centralizes the complex initialization logic for backends,
ensuring all dependencies are created and injected in the correct order.
"""

import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class BackendFactory:
    """
    Factory for creating backend instances with all dependencies properly initialized.

    This factory solves the circular dependency between Backend and SchemaRegistry
    by orchestrating the creation and injection in a single, well-defined location.
    """

    @staticmethod
    def create_backend_with_dependencies(
        backend_class: Type,
        backend_config: Any,
        config_manager: Any,
        schema_loader: Any,
        backend_init_config: Dict[str, Any],
    ) -> Any:
        """
        Create backend instance with all dependencies properly initialized.

        This method orchestrates the complex initialization sequence:
        1. Create backend instance (without schema_registry)
        2. Create SchemaRegistry (with backend reference)
        3. Inject schema_registry into backend
        4. Call backend.initialize()
        5. Inject schema_registry into backend's internal schema manager (if exists)

        All implementation-specific details are handled internally to prevent
        errors from scattered injection logic.

        Args:
            backend_class: Backend class to instantiate (must implement Backend interface)
            backend_config: Configuration object with connection details
            config_manager: Configuration manager instance
            schema_loader: Schema loader instance for loading schema templates
            backend_init_config: Configuration dict passed to backend.initialize()

        Returns:
            Fully initialized backend instance with all dependencies injected

        Raises:
            ValueError: If required dependencies are None
            Exception: If initialization fails

        Note:
            This method is backend-agnostic and works with any Backend implementation.
            No references to concrete backend types (Vespa, Qdrant, etc.).
        """
        if backend_config is None:
            raise ValueError("backend_config is required")
        if config_manager is None:
            raise ValueError("config_manager is required")
        if schema_loader is None:
            raise ValueError("schema_loader is required")
        if backend_init_config is None:
            raise ValueError("backend_init_config is required")

        logger.debug(f"Creating backend {backend_class.__name__} with factory")

        # Phase 1: Create backend instance
        # Backend is created without schema_registry to break circular dependency
        backend_instance = backend_class(
            backend_config=backend_config,
            schema_loader=schema_loader,
            config_manager=config_manager,
        )

        # Phase 2: Create SchemaRegistry with backend reference
        # SchemaRegistry needs backend for deployment operations
        from cogniverse_core.registries.schema_registry import SchemaRegistry

        # Debug: Log which DB config_manager is using
        db_path = (
            getattr(config_manager.store, "db_path", "unknown")
            if config_manager and hasattr(config_manager, "store")
            else "no store"
        )
        logger.warning(
            f"🔍 BACKEND_FACTORY creating SchemaRegistry with config_manager DB: {db_path}"
        )

        schema_registry = SchemaRegistry(
            config_manager=config_manager,
            backend=backend_instance,
            schema_loader=schema_loader,
            strict_mode=True,  # Production default - fail fast on errors
        )

        # Phase 3: Inject schema_registry into backend
        # Backend can now use schema_registry for schema operations
        backend_instance.schema_registry = schema_registry

        # Phase 4: Initialize backend with configuration
        # This may create internal components that also need schema_registry
        backend_instance.initialize(backend_init_config)

        # Phase 5: Inject schema_registry into backend's internal schema manager
        # Some backends have an internal schema manager that also needs registry access
        # This is backend-specific but safe to attempt on any backend
        if (
            hasattr(backend_instance, "schema_manager")
            and backend_instance.schema_manager
        ):
            if hasattr(backend_instance.schema_manager, "_schema_registry"):
                backend_instance.schema_manager._schema_registry = schema_registry
                logger.debug(
                    f"Injected schema_registry into {backend_class.__name__}.schema_manager"
                )

        logger.info(
            f"Successfully created {backend_class.__name__} with all dependencies"
        )

        return backend_instance
