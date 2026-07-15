"""
Adapter Loader for Agents

Utility for loading trained LoRA adapters from the registry.
Agents can use this to dynamically load fine-tuned adapters.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_active_adapter_path(
    tenant_id: str,
    agent_type: str,
    adapter_cache_dir: str,
) -> Optional[str]:
    """
    Get the active adapter's local path for a tenant and agent type.

    Resolves the adapter's effective URI through ``resolve_adapter_path``: a
    local ``file://`` adapter yields its path directly, while a cloud-backed
    adapter (``s3://`` / ``modal://``) is downloaded under ``adapter_cache_dir``.
    Returning ``adapter.adapter_path`` verbatim only ever worked for locally
    trained adapters — cloud URIs need this resolution step.

    Args:
        tenant_id: Tenant identifier
        agent_type: Agent type (routing, profile_selection, entity_extraction)
        adapter_cache_dir: Directory cloud-backed adapters download into —
            source it from ``SystemConfig.adapter_cache_dir``. Required even for
            local adapters so the call signature is uniform.

    Returns:
        Local filesystem path to the active adapter, or None if none is active

    Example:
        >>> cache_dir = config_manager.get_system_config().adapter_cache_dir
        >>> adapter_path = get_active_adapter_path("tenant1", "routing", cache_dir)
        >>> if adapter_path:
        ...     model.load_adapter(adapter_path)
    """
    try:
        from cogniverse_finetuning.registry import (
            AdapterRegistry,
            resolve_adapter_path,
        )

        registry = AdapterRegistry()
        adapter = registry.get_active_adapter(tenant_id, agent_type)

        if adapter:
            logger.info(
                f"Found active adapter for {tenant_id}/{agent_type}: "
                f"{adapter.name} v{adapter.version}"
            )
            return resolve_adapter_path(adapter.get_effective_uri(), adapter_cache_dir)

        logger.debug(f"No active adapter for {tenant_id}/{agent_type}")
        return None

    except ImportError:
        logger.debug("cogniverse_finetuning not available, skipping adapter lookup")
        return None
    except Exception as e:
        logger.warning(f"Failed to load adapter from registry: {e}")
        return None


def get_adapter_metadata(
    tenant_id: str,
    agent_type: str,
) -> Optional[dict]:
    """
    Get full metadata for the active adapter.

    Args:
        tenant_id: Tenant identifier
        agent_type: Agent type

    Returns:
        Dict with adapter metadata, or None if no active adapter

    Example:
        >>> metadata = get_adapter_metadata("tenant1", "routing")
        >>> if metadata:
        ...     print(f"Using adapter: {metadata['name']} v{metadata['version']}")
        ...     print(f"Trained with: {metadata['training_method']}")
    """
    try:
        from cogniverse_finetuning.registry import AdapterRegistry

        registry = AdapterRegistry()
        adapter = registry.get_active_adapter(tenant_id, agent_type)

        if adapter:
            return {
                "adapter_id": adapter.adapter_id,
                "name": adapter.name,
                "version": adapter.version,
                "base_model": adapter.base_model,
                "training_method": adapter.training_method,
                "adapter_path": adapter.adapter_path,
                "metrics": adapter.metrics,
            }

        return None

    except ImportError:
        logger.debug("cogniverse_finetuning not available, skipping adapter lookup")
        return None
    except Exception as e:
        logger.warning(f"Failed to get adapter metadata: {e}")
        return None


class AdapterAwareMixin:
    """
    Mixin for agents that can use fine-tuned adapters.

    Provides adapter loading capability that integrates with the registry.

    Example:
        >>> class MyAgent(A2AAgent, AdapterAwareMixin):
        ...     def __init__(self, deps):
        ...         super().__init__(...)
        ...         self.tenant_id = deps.tenant_id
        ...         # Try to load adapter during initialization
        ...         adapter_path = self.load_adapter_if_available("my_agent_type")
        ...         if adapter_path:
        ...             self._apply_adapter(adapter_path)
    """

    def load_adapter_if_available(
        self, agent_type: str, adapter_cache_dir: str
    ) -> Optional[str]:
        """
        Load active adapter from registry if available.

        Args:
            agent_type: Agent type to look up
            adapter_cache_dir: Directory cloud-backed adapters download into
                (``SystemConfig.adapter_cache_dir``)

        Returns:
            Adapter path if found, None otherwise
        """
        tenant_id = getattr(self, "tenant_id", None)
        if not tenant_id:
            logger.debug("No tenant_id available, skipping adapter lookup")
            return None

        return get_active_adapter_path(tenant_id, agent_type, adapter_cache_dir)

    def get_adapter_info(self, agent_type: str) -> Optional[dict]:
        """
        Get metadata for the active adapter.

        Args:
            agent_type: Agent type to look up

        Returns:
            Adapter metadata dict if found, None otherwise
        """
        tenant_id = getattr(self, "tenant_id", None)
        if not tenant_id:
            return None

        return get_adapter_metadata(tenant_id, agent_type)
