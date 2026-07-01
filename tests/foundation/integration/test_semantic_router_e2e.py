"""Real-boundary integration test for semantic routing through a live router.

This exercises the actual system boundary: cogniverse's own
``apply_semantic_routing`` + ``create_dspy_lm`` path, sending real chat
completions through Envoy -> vLLM Semantic Router -> a reflecting stub backend.
It does NOT mock the router — the assertions read what the stub reflects back
about the request the router actually forwarded, so they prove the router's
decisions (tenant-tier gating, content-driven model + reasoning selection,
header forwarding) end to end.

The ``semantic_router_stack`` fixture (``conftest.py``) launches and tears down
the whole stack itself via ``docker run`` — no compose file, no manual startup.

Routing model: cogniverse sends the tenant identity (x-authz-user-id) + tier
(x-authz-user-groups); the router gates the model set by tier (authz) and
classifies the prompt's domain to pick the model + reasoning:
  * free tier              -> basic-chat, no reasoning
  * pro tier + technical   -> pro-reasoning, reasoning ON
  * pro tier + non-technical -> pro-reasoning, reasoning OFF
"""

from __future__ import annotations

import json

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)


@pytest.fixture(scope="module")
def sr_base_url(semantic_router_stack) -> str:
    """Envoy base URL from the self-launched semantic-router stack."""
    return semantic_router_stack["base_url"]


def _semantic_router_config(base_url: str) -> SemanticRouterConfig:
    return SemanticRouterConfig(
        enabled=True,
        semantic_router_url=base_url,
        tenant_tiers={"pro-tenant": "pro", "free-tenant": "free"},
        default_tier="free",
    )


def _call(base_url: str, tenant_id: str, prompt: str) -> dict:
    """Route a real completion through the router; return the stub's reflection."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=_semantic_router_config(base_url),
        tenant_id=tenant_id,
    )
    lm = create_dspy_lm(routed)
    lm.cache = False
    out = lm(prompt)
    item = out[0] if isinstance(out, list) else out
    # In reasoning mode dspy returns a dict ({"text", "reasoning"}) rather than
    # a bare string; the stub's reflection payload is the message content.
    content = (
        item.get("text") or item.get("content") if isinstance(item, dict) else item
    )
    return json.loads(content)


_TECHNICAL = (
    "Write a recursive algorithm to balance a binary search tree and "
    "analyze its worst-case time complexity"
)


def test_both_authz_headers_reach_the_router(sr_base_url):
    reflected = _call(sr_base_url, "pro-tenant", "hello there")
    headers = reflected["routing_headers"]
    assert headers["x-authz-user-id"] == "pro-tenant"
    assert headers["x-authz-user-groups"] == "pro"


def test_free_tier_routes_to_basic_model_without_reasoning(sr_base_url):
    reflected = _call(sr_base_url, "free-tenant", _TECHNICAL)
    assert reflected["served_model"] == "basic-chat"
    assert reflected["reasoning"] is False


def test_pro_tier_technical_routes_to_reasoning_model(sr_base_url):
    reflected = _call(sr_base_url, "pro-tenant", _TECHNICAL)
    assert reflected["served_model"] == "pro-reasoning"
    assert reflected["reasoning"] is True


def test_pro_tier_non_technical_keeps_reasoning_off(sr_base_url):
    reflected = _call(sr_base_url, "pro-tenant", "what's a fun weekend activity?")
    assert reflected["served_model"] == "pro-reasoning"
    assert reflected["reasoning"] is False
