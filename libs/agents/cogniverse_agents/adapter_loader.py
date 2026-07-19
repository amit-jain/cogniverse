"""
Adapter Loader for Agents

Utility for loading trained LoRA adapters from the registry.
Agents can use this to dynamically load fine-tuned adapters.
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Last successful registry answer per (tenant, agent) — including a
# successful "no active adapter" (stored as None). A registry outage reuses
# this instead of silently reverting a finetuned tenant to the base model.
_LAST_KNOWN_ADAPTERS: Dict[Tuple[str, str], Any] = {}


def _resolve_active_adapter(tenant_id: str, agent_type: str) -> Any:
    """Return the active adapter object for ``(tenant, agent_type)``, or None
    when there is genuinely no active adapter.

    Outage-aware: ``AdapterRegistry.get_active_adapter`` RAISES on a Vespa /
    registry outage (a successful "no adapter" returns None). Flattening that
    raise to None silently reverts a finetuned tenant to the base model, so on
    an outage this reuses the last successful answer for the key; only when
    there is no prior state does it degrade to the base model (None) with an
    error log. On success it records the answer for future outages. Raises
    ``ImportError`` when cogniverse_finetuning isn't installed — callers treat
    that as "no adapter".
    """
    from cogniverse_finetuning.registry import AdapterRegistry

    key = (tenant_id, agent_type)
    try:
        adapter = AdapterRegistry().get_active_adapter(tenant_id, agent_type)
    except Exception as exc:  # noqa: BLE001 — outage, not absence
        if key in _LAST_KNOWN_ADAPTERS:
            cached = _LAST_KNOWN_ADAPTERS[key]
            logger.warning(
                "active-adapter lookup failed for %s/%s: %r — reusing last "
                "known state (%s)",
                tenant_id,
                agent_type,
                exc,
                cached.name if cached else "no active adapter",
            )
            return cached
        logger.error(
            "active-adapter lookup failed for %s/%s: %r — no known prior state, "
            "serving the base model",
            tenant_id,
            agent_type,
            exc,
        )
        return None
    _LAST_KNOWN_ADAPTERS[key] = adapter
    return adapter


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
        adapter = _resolve_active_adapter(tenant_id, agent_type)
    except ImportError:
        logger.debug("cogniverse_finetuning not available, skipping adapter lookup")
        return None

    if not adapter:
        logger.debug(f"No active adapter for {tenant_id}/{agent_type}")
        return None

    try:
        from cogniverse_finetuning.registry import resolve_adapter_path

        logger.info(
            f"Found active adapter for {tenant_id}/{agent_type}: "
            f"{adapter.name} v{adapter.version}"
        )
        return resolve_adapter_path(adapter.get_effective_uri(), adapter_cache_dir)
    except Exception as e:  # noqa: BLE001 — download/resolution failed, degrade
        logger.warning(f"Failed to resolve adapter path for {adapter.name}: {e}")
        return None


def adapter_lm_context(tenant_id: str, agent_type: str, config_manager=None):
    """A ``dspy.context`` binding the tenant's active adapter LM for
    ``agent_type``, or a ``nullcontext`` when no adapter is active.

    For agents whose DSPy call runs on the shared global LM (not a per-agent
    one), this is the seam that applies a tenant's fine-tuning: the active
    adapter's registry ``name`` becomes the LM model (vLLM serves the LoRA by
    name), built from the tenant's own LM endpoint. Any missing registry,
    absent adapter, or endpoint-resolution failure degrades to the base model.

    Args:
        tenant_id: Tenant whose active adapter applies.
        agent_type: Adapter agent type (e.g. ``profile_selection``).
        config_manager: ConfigManager for resolving the tenant LM endpoint —
            pass the dispatcher-injected manager when available. When None,
            the process-default manager is resolved so the standalone-agent
            path still binds the adapter instead of silently degrading.
    """
    from contextlib import nullcontext

    try:
        adapter = _resolve_active_adapter(tenant_id, agent_type)
    except ImportError:
        return nullcontext()

    if not adapter:
        return nullcontext()

    try:
        import dataclasses

        import dspy

        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )
        from cogniverse_foundation.dspy.model_format import ensure_provider_prefix

        if config_manager is None:
            # Standalone-agent path (no dispatcher injection). get_config
            # REQUIRES a manager — passing None raises, and the except below
            # would then silently strand every tenant on the base model, so
            # resolve the process-default manager here (same bootstrap the
            # AdapterRegistry above uses). An unresolvable environment still
            # degrades via the except.
            config_manager = create_default_config_manager()
        system_config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        primary = system_config.get_llm_config().primary
        endpoint = dataclasses.replace(
            primary, model=ensure_provider_prefix(adapter.name)
        )
        return dspy.context(lm=create_dspy_lm(endpoint))
    except Exception as exc:  # noqa: BLE001 — degrade to the base model
        logger.warning(
            "failed to build adapter LM for %s/%s (%s): %r — base model",
            tenant_id,
            agent_type,
            adapter.name,
            exc,
        )
        return nullcontext()


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
        adapter = _resolve_active_adapter(tenant_id, agent_type)
    except ImportError:
        logger.debug("cogniverse_finetuning not available, skipping adapter lookup")
        return None

    if not adapter:
        return None

    return {
        "adapter_id": adapter.adapter_id,
        "name": adapter.name,
        "version": adapter.version,
        "base_model": adapter.base_model,
        "training_method": adapter.training_method,
        "adapter_path": adapter.adapter_path,
        "metrics": adapter.metrics,
    }


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
