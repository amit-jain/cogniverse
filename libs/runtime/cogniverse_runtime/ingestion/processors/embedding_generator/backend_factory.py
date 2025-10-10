#!/usr/bin/env python3
"""
Backend Factory - Creates backend clients
"""

import logging
from typing import Any

from cogniverse_core.registries.backend_registry import get_backend_registry
from cogniverse_core.registries.interfaces import IngestionBackend


class BackendFactory:
    """Factory for creating backends using the backend registry"""

    @classmethod
    def create(
        cls,
        backend_type: str,
        tenant_id: str,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> IngestionBackend:
        """Create a backend of the specified type using the backend registry

        Args:
            backend_type: Type of backend (e.g., "vespa")
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Full application config
            logger: Optional logger

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

        # Get backend from registry (singleton per tenant)
        registry = get_backend_registry()

        backend = registry.get_ingestion_backend(backend_type, tenant_id, config)

        logger.warning(f"   Got backend instance for tenant {tenant_id}: {id(backend)}")

        return backend
