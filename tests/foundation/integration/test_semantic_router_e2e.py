"""Real-boundary e2e for router routing through a live semantic router.

This exercises the actual system boundary: cogniverse's own
``apply_semantic_routing`` + ``create_dspy_lm`` path, sending real chat
completions through Envoy -> vLLM Semantic Router -> a stub backend. It does
NOT mock the router — the assertions read what the stub reflects back about
the request the router actually forwarded, so they prove the router's
decisions (model choice, reasoning toggle, header forwarding) end to end.

Bring the stack up first (needs Docker — see
``deploy/semantic-router-local/README.md``)::

    docker compose -f deploy/semantic-router-local/docker-compose.yml up

then run::

    SR_ENVOY_URL=http://localhost:8801/v1 uv run pytest \
        tests/foundation/integration/test_semantic_router_sr_e2e.py -v

When ``SR_ENVOY_URL`` is unreachable (e.g. CI without Docker) the whole
module skips — it is a real-service suite, not a unit test.
"""

from __future__ import annotations

import json
import os
import socket
from urllib.parse import urlparse

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

SR_ENVOY_URL = os.environ.get("SR_ENVOY_URL", "http://localhost:8801/v1")


def _reachable(url: str, timeout: float = 1.5) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def sr_base_url() -> str:
    if not _reachable(SR_ENVOY_URL):
        pytest.skip(
            f"semantic-router stack not reachable at {SR_ENVOY_URL}; "
            "start deploy/semantic-router-local/docker-compose.yml to run this suite"
        )
    # localhost must bypass any outbound HTTPS proxy the environment sets.
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    return SR_ENVOY_URL


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
