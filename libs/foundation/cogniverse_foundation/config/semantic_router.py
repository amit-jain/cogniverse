"""Apply opt-in semantic routing to an LLM endpoint config.

When ``SemanticRouterConfig.enabled`` is set, an ``LLMEndpointConfig`` is
rewritten to target the semantic router (``semantic_router_url``) instead of
the model backend, and two authz headers are attached per request:

  - tenant identity -> ``user_id_header`` (default ``x-authz-user-id``) = the
    ``tenant_id``
  - tenant tier     -> ``tier_header`` (default ``x-authz-user-groups``),
    resolved from ``tenant_tiers[tenant_id]`` with ``default_tier`` fallback

The router's authz signal requires the identity header and refuses to evaluate
role bindings without it (no silent bypass); it then gates the tenant's
allowed model set by tier and classifies the request content itself
(domain/complexity) to pick the model + reasoning mode — so cogniverse sends
only *who* the tenant is, not *what kind* of request it is. The resolved
headers are merged onto any ``extra_headers`` already on the endpoint (they
win on key collision). ``create_dspy_lm`` then forwards them to litellm. When
routing is disabled the endpoint is returned unchanged, so the
direct-to-backend path is identical.
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
    config: SemanticRouterConfig, tenant_id: str
) -> Optional[Dict[str, str]]:
    """Resolve the authz routing headers for a request.

    Returns ``None`` when routing is disabled. Otherwise returns two headers:
    the tenant identity (``user_id_header`` = ``tenant_id``) and the tenant
    tier (``tier_header``; an unknown tenant maps to ``config.default_tier``).
    The router's authz signal requires the identity header — it refuses to
    evaluate role bindings on the tier/group header alone.
    """
    if not config.enabled:
        return None
    tier = config.tenant_tiers.get(tenant_id, config.default_tier)
    return {
        config.user_id_header: tenant_id,
        config.tier_header: tier,
    }


def apply_semantic_routing(
    endpoint: LLMEndpointConfig,
    config: SemanticRouterConfig,
    tenant_id: str,
) -> LLMEndpointConfig:
    """Return an endpoint routed through the semantic router, or the original.

    When ``config.enabled`` is False the input ``endpoint`` is returned as-is
    (same object). When enabled, a deep copy is returned with ``api_base`` set
    to ``config.semantic_router_url``, ``model`` set to ``config.routed_model``
    (the router resolves models by its own catalog names / auto alias and
    rejects raw provider model ids with a 400), and the resolved tier header
    merged onto ``extra_headers``.

    Raises ``ValueError`` if routing is enabled but ``semantic_router_url`` is
    empty — a misconfiguration that would otherwise silently send LLM traffic
    to the wrong place.
    """
    if not config.enabled:
        return endpoint
    if not config.semantic_router_url:
        raise ValueError(
            "SemanticRouterConfig.enabled is True but semantic_router_url is "
            "empty; set it to the semantic router's OpenAI-compatible endpoint"
        )

    headers = resolve_semantic_router_headers(config, tenant_id)
    merged = dict(endpoint.extra_headers or {})
    merged.update(headers or {})

    routed = copy.deepcopy(endpoint)
    routed.api_base = config.semantic_router_url
    routed.model = config.routed_model
    routed.extra_headers = merged
    return routed


def resolve_semantic_router_config(config_accessor: object) -> SemanticRouterConfig:
    """Return the ``SemanticRouterConfig`` from an object exposing
    ``get_semantic_router()`` (e.g. ``ConfigUtils``).

    A disabled default is returned only when routing is genuinely not
    configured: the accessor is absent/not callable, or (a test hazard) yields
    a non-``SemanticRouterConfig`` such as a bare ``MagicMock`` whose auto
    attributes would otherwise look enabled. Any error raised by the accessor
    propagates — a broken config store must surface, not silently disable
    routing.
    """
    accessor = getattr(config_accessor, "get_semantic_router", None)
    if not callable(accessor):
        return SemanticRouterConfig()
    router = accessor()
    return (
        router if isinstance(router, SemanticRouterConfig) else SemanticRouterConfig()
    )


def create_routed_lm(
    endpoint: LLMEndpointConfig,
    config: SemanticRouterConfig,
    tenant_id: str,
) -> "dspy.LM":
    """Build a ``dspy.LM`` for ``tenant_id`` routed through the router when
    ``config.enabled``.

    Composes ``apply_semantic_routing`` + ``create_dspy_lm`` — the single way
    an agent builds a semantic-router-aware LM. When routing is disabled the LM
    targets the endpoint's own ``api_base`` unchanged.
    """
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=config,
        tenant_id=tenant_id,
    )
    return create_dspy_lm(routed)


def routed_lm_context_for(
    config_manager: object,
    tenant_id: str,
    agent_name: str,
    endpoint: Optional[LLMEndpointConfig] = None,
):
    """Return a ``dspy.context`` binding the LM for a request, routed through
    the semantic router when ``semantic_router`` is enabled.

    The single entry point every agent uses to make its per-request DSPy call
    semantic-router-aware. There is exactly one input for the LM — the agent's
    ``endpoint`` (an ``LLMEndpointConfig``):

    - Enabled: the endpoint is routed through the semantic router (api_base +
      tenant-tier header) for ``tenant_id``; the model becomes the router's
      auto alias and the router picks the concrete model itself.
    - Disabled: the LM is built directly from the same endpoint — the plain
      direct-to-backend path, not a fallback.

    When ``endpoint`` is omitted the endpoint is resolved from config for
    ``agent_name`` on the enabled path (``agent_name`` selects which endpoint,
    not any routing signal), and the ambient ``dspy.settings.lm`` is left in
    place on the direct path (for callers that rely on the global LM, e.g. the
    orchestrator).
    """
    from contextlib import nullcontext

    import dspy

    from cogniverse_foundation.config.utils import get_config

    def _direct():
        # Router disabled: the endpoint's own LM, or the ambient
        # dspy.settings.lm when the agent supplied no endpoint. This is the
        # feature being off, NOT an error fallback.
        if endpoint is not None:
            return dspy.context(lm=create_dspy_lm(endpoint))
        return nullcontext()

    # No config manager (standalone agent process without a config store)
    # means there is nowhere to read semantic_router config from — the
    # feature cannot be enabled, so the direct path is the defined
    # behaviour, not a fallback.
    if config_manager is None:
        return _direct()

    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    router = resolve_semantic_router_config(cfg)
    if not router.enabled:
        return _direct()
    ep = endpoint if endpoint is not None else cfg.get_llm_config().resolve(agent_name)
    return dspy.context(lm=create_routed_lm(ep, router, tenant_id))
