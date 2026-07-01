"""Apply opt-in semantic routing to an LLM endpoint config.

When ``SemanticRouterConfig.enabled`` is set, an ``LLMEndpointConfig`` is
rewritten to target the semantic router (``semantic_router_url``) instead of the model
backend, and per-request routing metadata is attached as HTTP headers so a
semantic router can pick the backend model and reasoning mode:

  - tenant tier  -> ``tier_header`` (default ``x-authz-user-groups``),
    resolved from ``tenant_tiers[tenant_id]`` with ``default_tier`` fallback
  - task label   -> ``task_header`` (default ``x-vsr-task``),
    resolved from ``agent_tasks[agent_name]`` with ``default_task`` fallback

The resolved headers are merged onto any ``extra_headers`` already on the
endpoint (tier/task win on key collision). ``create_dspy_lm`` then forwards
them to litellm. When routing is disabled the endpoint is returned unchanged,
so the direct-to-backend path is byte-for-byte identical.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, Optional

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

if TYPE_CHECKING:
    import dspy


def resolve_semantic_router_headers(
    config: SemanticRouterConfig, tenant_id: str, agent_name: str
) -> Optional[Dict[str, str]]:
    """Resolve the tier/task routing headers for a (tenant, agent) pair.

    Returns ``None`` when routing is disabled. Otherwise returns exactly two
    headers keyed by ``config.tier_header`` and ``config.task_header``. An
    unknown tenant maps to ``config.default_tier``; an unknown agent maps to
    ``config.default_task``.
    """
    if not config.enabled:
        return None
    tier = config.tenant_tiers.get(tenant_id, config.default_tier)
    task = config.agent_tasks.get(agent_name, config.default_task)
    return {config.tier_header: tier, config.task_header: task}


def apply_semantic_routing(
    endpoint: LLMEndpointConfig,
    config: SemanticRouterConfig,
    tenant_id: str,
    agent_name: str,
) -> LLMEndpointConfig:
    """Return an endpoint config routed through the semantic router, or the original.

    When ``config.enabled`` is False the input ``endpoint`` is returned as-is
    (same object). When enabled, a deep copy is returned with ``api_base`` set
    to ``config.semantic_router_url`` and the resolved tier/task headers merged
    onto ``extra_headers``.

    Raises ``ValueError`` if routing is enabled but ``semantic_router_url`` is
    empty — a misconfiguration that would otherwise silently send LLM traffic
    to the wrong place.
    """
    if not config.enabled:
        return endpoint
    if not config.semantic_router_url:
        raise ValueError(
            "SemanticRouterConfig.enabled is True but semantic_router_url is empty; "
            "set semantic_router_url to the semantic router's OpenAI-compatible endpoint"
        )

    headers = resolve_semantic_router_headers(config, tenant_id, agent_name)
    merged = dict(endpoint.extra_headers or {})
    merged.update(headers or {})

    routed = copy.deepcopy(endpoint)
    routed.api_base = config.semantic_router_url
    routed.extra_headers = merged
    return routed


def resolve_semantic_router_config(config_accessor: object) -> SemanticRouterConfig:
    """Best-effort ``SemanticRouterConfig`` from an object exposing
    ``get_semantic_router()`` (e.g. ``ConfigUtils``).

    Returns a disabled default when the accessor is absent, raises, or yields a
    non-``SemanticRouterConfig`` value. The ``isinstance`` guard keeps a mocked
    accessor (whose auto attributes look truthy) from being treated as enabled.
    """
    accessor = getattr(config_accessor, "get_semantic_router", None)
    if not callable(accessor):
        return SemanticRouterConfig()
    try:
        router = accessor()
    except Exception:  # noqa: BLE001 — never block LM construction on config
        return SemanticRouterConfig()
    return (
        router if isinstance(router, SemanticRouterConfig) else SemanticRouterConfig()
    )


def create_routed_lm(
    endpoint: LLMEndpointConfig,
    config: SemanticRouterConfig,
    tenant_id: str,
    agent_name: str,
) -> "dspy.LM":
    """Build a ``dspy.LM`` for ``(tenant_id, agent_name)`` routed through the
    router when ``config.enabled``.

    Composes ``apply_semantic_routing`` + ``create_dspy_lm`` — the single way an
    agent should build a semantic-router-aware LM. When routing is disabled the LM
    targets the endpoint's own ``api_base`` unchanged.
    """
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=config,
        tenant_id=tenant_id,
        agent_name=agent_name,
    )
    return create_dspy_lm(routed)


def routed_lm_context_for(
    config_manager: object,
    tenant_id: str,
    agent_name: str,
    endpoint: Optional[LLMEndpointConfig] = None,
):
    """Return a ``dspy.context`` binding ``agent_name``'s LM for a request,
    routed through the semantic router when ``semantic_router`` is enabled.

    The single entry point every agent uses to make its per-request DSPy call
    semantic-router-aware. There is exactly one input for the LM — the agent's
    ``endpoint`` (an ``LLMEndpointConfig``):

    - Enabled: the endpoint is routed through the semantic router (api_base + tier/task
      headers) for ``tenant_id``, preserving the endpoint's model/adapter.
    - Disabled: the LM is built directly from the same endpoint — this is the
      plain direct-to-backend path, not a fallback.

    When ``endpoint`` is omitted the endpoint is resolved from config for
    ``agent_name`` on the enabled path, and the ambient ``dspy.settings.lm`` is
    left in place on the direct path (for callers that rely on the global LM,
    e.g. the orchestrator).
    """
    from contextlib import nullcontext

    import dspy

    from cogniverse_foundation.config.utils import get_config

    def _direct():
        # Semantic router off or config unreachable: the endpoint's own LM, or the
        # ambient dspy.settings.lm when the agent supplied no endpoint.
        if endpoint is not None:
            return dspy.context(lm=create_dspy_lm(endpoint))
        return nullcontext()

    try:
        cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
        router = resolve_semantic_router_config(cfg)
        if not router.enabled:
            return _direct()
        ep = (
            endpoint
            if endpoint is not None
            else cfg.get_llm_config().resolve(agent_name)
        )
        return dspy.context(lm=create_routed_lm(ep, router, tenant_id, agent_name))
    except Exception:  # noqa: BLE001 — never block a request on routing config
        return _direct()
