"""Shared helper: resolve an LM endpoint config via the agent's
``_config_manager`` when no caller-supplied ``llm_config`` is set.

The knowledge-family agents (multi-doc synthesis, federated query, KG
traversal) accept both an explicit ``llm_config`` and a ``config_manager``.
The audit flagged ``_config_manager`` as a dead injection point — set
but never read. This helper gives the param a real consumer: it's the
fallback source for the per-tenant LM endpoint when the explicit one
is omitted.
"""

from __future__ import annotations

from typing import Any, Optional


def resolve_llm_config(
    explicit: Optional[Any], config_manager: Optional[Any]
) -> Optional[Any]:
    """Return ``explicit`` when set; otherwise look up the system primary
    LM endpoint via ``config_manager``. Returns ``None`` when neither
    yields a config — caller handles the missing-LM case (typically
    skips the RLM/DSPy path and falls through to a deterministic
    branch)."""
    if explicit is not None:
        return explicit
    if config_manager is None:
        return None
    try:
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_foundation.config.utils import get_config
    except ImportError:
        return None

    cfg = get_config(tenant_id="__system__", config_manager=config_manager)
    primary = cfg.get("llm_config", {}).get("primary")
    if not primary:
        return None
    return LLMEndpointConfig(**primary)
