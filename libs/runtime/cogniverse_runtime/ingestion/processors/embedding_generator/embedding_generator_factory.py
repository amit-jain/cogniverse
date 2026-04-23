#!/usr/bin/env python3
"""
Factory for creating embedding generators based on backend type
"""

import logging
from typing import Any

from .backend_factory import BackendFactory
from .embedding_generator_impl import EmbeddingGeneratorImpl


class EmbeddingGeneratorFactory:
    """Factory for creating embedding generators"""

    @staticmethod
    def create(
        backend: str,
        tenant_id: str,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
        profile_config: dict[str, Any] = None,
        config_manager=None,
        schema_loader=None,
    ):
        """
        Create an embedding generator for the specified backend

        Args:
            backend: Backend type ("vespa", "elasticsearch" in future, etc)
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Configuration dictionary
            logger: Logger instance
            profile_config: Profile configuration containing process_type and model info
            config_manager: ConfigManager instance for dependency injection (REQUIRED)
            schema_loader: SchemaLoader instance for dependency injection (REQUIRED)

        Returns:
            Embedding generator instance

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        # Get backend client (singleton per tenant)
        backend_client = BackendFactory.create(
            backend_type=backend,
            tenant_id=tenant_id,
            config=config,
            logger=logger,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        return EmbeddingGeneratorImpl(
            config=profile_config,  # This becomes self.profile_config
            logger=logger,
            backend_client=backend_client,
        )


def create_embedding_generator(
    config: dict[str, Any],
    schema_name: str,
    tenant_id: str,
    logger: logging.Logger | None = None,
    config_manager=None,
    schema_loader=None,
):
    """
    Creates an embedding generator for a specific schema and tenant.

    Args:
        config: Main configuration dictionary
        schema_name: Schema/profile name to use
        tenant_id: Tenant identifier (REQUIRED - no default)
        logger: Optional logger
        config_manager: ConfigManager instance for dependency injection (REQUIRED)
        schema_loader: SchemaLoader instance for dependency injection (REQUIRED)

    Raises:
        ValueError: If tenant_id is empty or None
    """
    if not tenant_id:
        raise ValueError("tenant_id is required - no default tenant")

    # Get backend from config
    backend = config.get("embedding_backend", config.get("search_backend", "vespa"))

    # Get profile configuration using schema_name as key
    backend_config = config.get("backend", {})
    profiles = backend_config.get("profiles", {})
    if schema_name not in profiles:
        raise ValueError(f"Profile '{schema_name}' not found in backend.profiles")

    profile_config = profiles[schema_name]
    # Use the profile's own schema_name if present (may differ from profile key),
    # otherwise fall back to the profile key as schema_name
    if "schema_name" not in profile_config:
        profile_config["schema_name"] = schema_name

    # Inject remote inference URL from SystemConfig if available. All model
    # loaders route via ``inference_service_urls`` keyed by the profile's
    # ``inference_service`` name; fall back to the legacy
    # ``colpali_inference_url`` field when a ColPali/ColQwen profile has no
    # ``inference_service`` set.
    if "remote_inference_url" not in profile_config and config_manager is not None:
        system_config = config_manager.get_system_config()
        loader = profile_config.get("model_loader", "")
        service_name = profile_config.get("inference_service")
        service_urls = getattr(system_config, "inference_service_urls", {}) or {}

        if service_name:
            url = service_urls.get(service_name)
            if not url:
                available = sorted(service_urls)
                raise ValueError(
                    f"Profile {schema_name!r} specifies "
                    f"inference_service={service_name!r} but no URL is "
                    f"configured. Deployed services: {available}."
                )
            profile_config["remote_inference_url"] = url
        elif loader in ("colpali", "colqwen") and system_config.colpali_inference_url:
            profile_config["remote_inference_url"] = system_config.colpali_inference_url

    # Create and return generator
    return EmbeddingGeneratorFactory.create(
        backend=backend,
        tenant_id=tenant_id,
        config=config,
        logger=logger,
        profile_config=profile_config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
