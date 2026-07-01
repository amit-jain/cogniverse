"""Real-boundary integration test for semantic routing through a live router.

This exercises the actual system boundary: cogniverse's own
``apply_semantic_routing`` + ``create_dspy_lm`` path, sending real chat
completions through Envoy -> vLLM Semantic Router -> a reflecting stub backend.
It does NOT mock the router — the assertions read what the stub reflects back
about the request the router actually forwarded, so they prove the router's
decisions (model choice, reasoning toggle, header forwarding) end to end.

The ``semantic_router_stack`` fixture (``conftest.py``) launches and tears down
the whole stack itself via ``docker run`` — no compose file, no manual startup.
When the Docker daemon is absent the fixture skips the module cleanly.
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
        agent_tasks={
            "orchestrator_agent": "orchestrator_plan",
            "query_enhancement_agent": "enhance",
        },
        default_task="general",
    )


def _call(base_url: str, tenant_id: str, agent_name: str, prompt: str) -> dict:
    """Route a real completion through the semantic router and return the stub's reflection."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=_semantic_router_config(base_url),
        tenant_id=tenant_id,
        agent_name=agent_name,
    )
    lm = create_dspy_lm(routed)
    lm.cache = False
    out = lm(prompt)
    content = out[0] if isinstance(out, list) else out
    return json.loads(content)


def test_completion_survives_the_proxy(sr_base_url):
    sentinel = "router-roundtrip-sentinel-42"
    reflected = _call(sr_base_url, "free-tenant", "query_enhancement_agent", sentinel)
    assert reflected["echo"] == sentinel
    assert reflected["backend_tag"] == "stub"


def test_routing_headers_reach_the_backend(sr_base_url):
    reflected = _call(
        sr_base_url, "pro-tenant", "query_enhancement_agent", "header check"
    )
    headers = reflected["routing_headers"]
    assert headers["x-authz-user-groups"] == "pro"
    assert headers["x-vsr-task"] == "enhance"


def test_free_tier_routes_to_basic_model(sr_base_url):
    reflected = _call(sr_base_url, "free-tenant", "search_agent", "cheap please")
    assert reflected["served_model"] == "basic-chat"
    assert reflected["reasoning"] is False


def test_pro_planning_routes_to_reasoning_model(sr_base_url):
    reflected = _call(sr_base_url, "pro-tenant", "orchestrator_agent", "plan this")
    assert reflected["served_model"] == "pro-reasoning"
    assert reflected["reasoning"] is True


def test_pro_non_planning_task_keeps_reasoning_off(sr_base_url):
    reflected = _call(
        sr_base_url, "pro-tenant", "query_enhancement_agent", "just rewrite"
    )
    assert reflected["served_model"] == "pro-reasoning"
    assert reflected["reasoning"] is False
