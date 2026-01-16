"""
Inference Integration for Adapter Registry

Provides functions for inference stacks (vLLM, etc.) to query and load adapters.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """
    Adapter information for inference.

    Contains just the essential info needed to load an adapter.
    """

    adapter_id: str
    name: str
    version: str
    base_model: str
    adapter_uri: str  # Effective URI (file://, s3://, modal://)
    adapter_path: str  # Local path (for file:// URIs)


def get_active_adapter_for_inference(
    tenant_id: str,
    agent_type: str,
) -> Optional[AdapterInfo]:
    """
    Get the active adapter for inference.

    This is the primary function for inference stacks to query the registry.
    Returns the effective URI that can be used to load the adapter.

    Args:
        tenant_id: Tenant identifier
        agent_type: Agent type (routing, profile_selection, entity_extraction)

    Returns:
        AdapterInfo with URI for loading, or None if no active adapter

    Example (vLLM integration):
        >>> from cogniverse_finetuning.registry.inference import get_active_adapter_for_inference
        >>>
        >>> adapter = get_active_adapter_for_inference("tenant1", "routing")
        >>> if adapter:
        ...     # For local file:// URIs, use adapter_path directly
        ...     if adapter.adapter_uri.startswith("file://"):
        ...         lora_path = adapter.adapter_path
        ...     else:
        ...         # For cloud URIs, download first
        ...         from cogniverse_finetuning.registry import download_adapter
        ...         lora_path = download_adapter(adapter.adapter_uri, "/tmp/adapters")
        ...
        ...     # Load in vLLM
        ...     llm = LLM(model=adapter.base_model, enable_lora=True)
        ...     output = llm.generate(prompts, lora_request=LoRARequest("adapter", 1, lora_path))
    """
    try:
        from cogniverse_finetuning.registry import AdapterRegistry

        registry = AdapterRegistry()
        adapter = registry.get_active_adapter(tenant_id, agent_type)

        if adapter is None:
            logger.debug(f"No active adapter for {tenant_id}/{agent_type}")
            return None

        return AdapterInfo(
            adapter_id=adapter.adapter_id,
            name=adapter.name,
            version=adapter.version,
            base_model=adapter.base_model,
            adapter_uri=adapter.get_effective_uri(),
            adapter_path=adapter.adapter_path,
        )

    except Exception as e:
        logger.warning(f"Failed to get active adapter: {e}")
        return None


def list_available_adapters(
    tenant_id: str,
    agent_type: Optional[str] = None,
    model_type: Optional[str] = None,
) -> list[AdapterInfo]:
    """
    List all available adapters for a tenant.

    Useful for inference servers that want to pre-load multiple adapters.

    Args:
        tenant_id: Tenant identifier
        agent_type: Filter by agent type (optional)
        model_type: Filter by model type - "llm" or "embedding" (optional)

    Returns:
        List of AdapterInfo for all matching adapters

    Example:
        >>> adapters = list_available_adapters("tenant1", agent_type="routing")
        >>> for adapter in adapters:
        ...     print(f"{adapter.name} v{adapter.version}: {adapter.adapter_uri}")
    """
    try:
        from cogniverse_finetuning.registry import AdapterRegistry

        registry = AdapterRegistry()
        adapters = registry.list_adapters(
            tenant_id=tenant_id,
            agent_type=agent_type,
            model_type=model_type,
            status="active",  # Only active adapters
        )

        return [
            AdapterInfo(
                adapter_id=a.adapter_id,
                name=a.name,
                version=a.version,
                base_model=a.base_model,
                adapter_uri=a.get_effective_uri(),
                adapter_path=a.adapter_path,
            )
            for a in adapters
        ]

    except Exception as e:
        logger.warning(f"Failed to list adapters: {e}")
        return []


def resolve_adapter_path(adapter_uri: str, cache_dir: str = "/tmp/adapters") -> str:
    """
    Resolve adapter URI to a local path.

    Downloads the adapter if necessary (for cloud URIs).
    For file:// URIs, returns the path directly.

    Args:
        adapter_uri: Adapter URI (file://, s3://, modal://)
        cache_dir: Directory to download adapters to

    Returns:
        Local filesystem path to adapter

    Example:
        >>> path = resolve_adapter_path("file:///data/adapters/routing_sft")
        '/data/adapters/routing_sft'

        >>> path = resolve_adapter_path("s3://bucket/adapters/routing_sft", "/tmp/cache")
        '/tmp/cache/routing_sft'  # Downloaded from S3
    """
    if adapter_uri.startswith("file://"):
        return adapter_uri[7:]  # Strip file://

    if not adapter_uri.startswith(("s3://", "gs://", "modal://")):
        # Assume it's a local path
        return adapter_uri

    # Download from cloud storage
    from pathlib import Path

    from cogniverse_finetuning.registry import download_adapter

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate local path from URI
    # e.g., s3://bucket/adapters/routing_sft -> /tmp/adapters/routing_sft
    uri_parts = adapter_uri.split("/")
    adapter_name = uri_parts[-1] if uri_parts[-1] else uri_parts[-2]
    local_path = str(cache_path / adapter_name)

    return download_adapter(adapter_uri, local_path)
