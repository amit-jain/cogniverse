#!/usr/bin/env python3
"""
Backend Factory - Creates backend clients
"""

import logging
from typing import Any

from cogniverse_core.interfaces.backend import IngestionBackend


class BackendFactory:
    """Factory for creating backends using the backend registry"""

    @classmethod
    def create(
        cls,
        backend_type: str,
        tenant_id: str,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
        config_manager=None,
        schema_loader=None,
    ) -> IngestionBackend:
        """Create a backend of the specified type using the backend registry

        Args:
            backend_type: Type of backend (e.g., "vespa")
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Full application config
            logger: Optional logger
            config_manager: ConfigManager instance for dependency injection
            schema_loader: SchemaLoader instance for dependency injection

        Returns:
            IngestionBackend: The backend that implements the ingestion interface

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        backend_type = backend_type.lower()

        # CRITICAL: Log backend creation to track when new instances are made
        if not logger:
            logger = logging.getLogger(__name__)

        logger.warning(f"🏭 BACKEND FACTORY: Getting {backend_type} backend for tenant: {tenant_id}")
        logger.warning(f"   Config keys: {list(config.keys())[:10]}")

        # Require config_manager via dependency injection
        if config_manager is None:
            raise ValueError(
                "config_manager is required for BackendFactory.create(). "
                "Dependency injection is mandatory - pass ConfigManager instance explicitly."
            )

        # Require schema_loader via dependency injection
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for BackendFactory.create(). "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )

        # Get backend from registry with dependency injection
        from cogniverse_core.registries.backend_registry import BackendRegistry
        registry = BackendRegistry()

        backend = registry.get_ingestion_backend(backend_type, tenant_id, config, config_manager=config_manager, schema_loader=schema_loader)

        logger.warning(f"   Got backend instance for tenant {tenant_id}: {id(backend)}")

        return backend
