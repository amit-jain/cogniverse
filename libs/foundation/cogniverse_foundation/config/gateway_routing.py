"""Apply opt-in gateway routing to an LLM endpoint config.

When ``GatewayRoutingConfig.enabled`` is set, an ``LLMEndpointConfig`` is
rewritten to target the gateway (``gateway_base_url``) instead of the model
backend, and per-request routing metadata is attached as HTTP headers so a
semantic-router gateway can pick the backend model and reasoning mode:

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
    GatewayRoutingConfig,
    LLMEndpointConfig,
)

if TYPE_CHECKING:
    import dspy


def resolve_gateway_headers(
    config: GatewayRoutingConfig, tenant_id: str, agent_name: str
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


def apply_gateway_routing(
    endpoint: LLMEndpointConfig,
    config: GatewayRoutingConfig,
    tenant_id: str,
    agent_name: str,
) -> LLMEndpointConfig:
    """Return an endpoint config routed through the gateway, or the original.

    When ``config.enabled`` is False the input ``endpoint`` is returned as-is
    (same object). When enabled, a deep copy is returned with ``api_base`` set
    to ``config.gateway_base_url`` and the resolved tier/task headers merged
    onto ``extra_headers``.

    Raises ``ValueError`` if routing is enabled but ``gateway_base_url`` is
    empty — a misconfiguration that would otherwise silently send LLM traffic
    to the wrong place.
    """
    if not config.enabled:
        return endpoint
    if not config.gateway_base_url:
        raise ValueError(
            "GatewayRoutingConfig.enabled is True but gateway_base_url is empty; "
            "set gateway_base_url to the gateway's OpenAI-compatible endpoint"
        )

    headers = resolve_gateway_headers(config, tenant_id, agent_name)
    merged = dict(endpoint.extra_headers or {})
    merged.update(headers or {})

    routed = copy.deepcopy(endpoint)
    routed.api_base = config.gateway_base_url
    routed.extra_headers = merged
    return routed


def resolve_gateway_config(config_accessor: object) -> GatewayRoutingConfig:
    """Best-effort ``GatewayRoutingConfig`` from an object exposing
    ``get_gateway_routing()`` (e.g. ``ConfigUtils``).

    Returns a disabled default when the accessor is absent, raises, or yields a
    non-``GatewayRoutingConfig`` value. The ``isinstance`` guard keeps a mocked
    accessor (whose auto attributes look truthy) from being treated as enabled.
    """
    accessor = getattr(config_accessor, "get_gateway_routing", None)
    if not callable(accessor):
        return GatewayRoutingConfig()
    try:
        gateway = accessor()
    except Exception:  # noqa: BLE001 — never block LM construction on config
        return GatewayRoutingConfig()
    return (
        gateway if isinstance(gateway, GatewayRoutingConfig) else GatewayRoutingConfig()
    )


def create_routed_lm(
    endpoint: LLMEndpointConfig,
    config: GatewayRoutingConfig,
    tenant_id: str,
    agent_name: str,
) -> "dspy.LM":
    """Build a ``dspy.LM`` for ``(tenant_id, agent_name)`` routed through the
    gateway when ``config.enabled``.

    Composes ``apply_gateway_routing`` + ``create_dspy_lm`` — the single way an
    agent should build a gateway-aware LM. When routing is disabled the LM
    targets the endpoint's own ``api_base`` unchanged.
    """
    routed = apply_gateway_routing(
        endpoint=endpoint,
        config=config,
        tenant_id=tenant_id,
        agent_name=agent_name,
    )
    return create_dspy_lm(routed)
