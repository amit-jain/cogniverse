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

Picking a model means the router rebuilds the request body from its own
parsed struct, so every field cogniverse sends has to survive that round
trip. ``response_format`` is the one with teeth: agents bind their DSPy
signature to a server-enforced ``json_schema``, and a router that keeps only
``{"type": "json_schema"}`` makes the backend reject the request outright.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.dspy.structured_json_adapter import signature_response_format


@pytest.fixture(scope="module")
def sr_base_url(semantic_router_stack) -> str:
    """Envoy base URL from the self-launched semantic-router stack."""
    return semantic_router_stack["base_url"]


# The tiers these tenants are on. In production each comes from the tenant's
# stored attribute (``cogniverse_foundation.config.tenant_tiers``); here the
# router, not the store, is the boundary under test.
_TIERS = {"pro-tenant": "pro", "free-tenant": "free"}
_FALLBACK_TIER = "free"


def _semantic_router_config(base_url: str) -> SemanticRouterConfig:
    return SemanticRouterConfig(enabled=True, semantic_router_url=base_url)


def _base_tier_config(base_url: str) -> SemanticRouterConfig:
    """A tenant with no stored tier, which is every tenant until an operator
    sets one: the request then carries ``DEFAULT_ROUTER_TIER``."""
    return SemanticRouterConfig(enabled=True, semantic_router_url=base_url)


def _call(base_url: str, tenant_id: str, prompt: str) -> dict:
    """Route a real completion through the router; return the stub's reflection."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=_semantic_router_config(base_url),
        tenant_id=tenant_id,
        tier=_TIERS.get(tenant_id, _FALLBACK_TIER),
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


def test_the_shipped_default_tier_matches_a_decision(sr_base_url):
    """The tier a stock deployment emits must reach a routing decision.

    A tenant with no stored tier carries ``DEFAULT_ROUTER_TIER``, which is
    every tenant until an operator sets one. Before the chart bound a group
    for it, that tier matched no decision: the router classified
    the request, logged ``No decision matched``, discarded the result and fell
    through to ``providers.defaults.default_model``.
    """
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=_base_tier_config(sr_base_url),
        tenant_id="unmapped-tenant",
        tier=DEFAULT_ROUTER_TIER,
    )
    lm = create_dspy_lm(routed)
    lm.cache = False
    out = lm("summarise this paragraph")
    item = out[0] if isinstance(out, list) else out
    content = (
        item.get("text") or item.get("content") if isinstance(item, dict) else item
    )
    reflected = json.loads(content)

    assert reflected["routing_headers"]["x-authz-user-groups"] == "default"
    assert reflected["served_model"] == "basic-chat"
    assert reflected["reasoning"] is False


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


def test_the_semantic_cache_never_answers_one_tenant_from_another(sr_base_url):
    """Two tenants on the SAME tier, byte-identical prompt.

    Every routing decision attaches the router's semantic cache
    (``plugins: - type: semantic-cache`` on each decision in
    ``charts/cogniverse/files/semantic-router/config.yaml``), keyed by prompt
    similarity at a 0.95 threshold. Same tier means the same decision, so both
    tenants land in the same cache; identical prompts are similarity 1.0. If
    that cache is not scoped by tenant, the second tenant is answered out of
    the first's entry.

    The stub reflects the identity header it was called with, so a crossed
    answer shows up as the wrong tenant id coming back - which is a tenant
    isolation failure, not a latency question.
    """
    prompt = "summarise the quarterly outlook for the northern region"

    first = _call(sr_base_url, "free-tenant", prompt)
    second = _call(sr_base_url, "another-free-tenant", prompt)

    assert first["routing_headers"]["x-authz-user-id"] == "free-tenant"
    assert second["routing_headers"]["x-authz-user-id"] == "another-free-tenant"
    # Same tier, so the decision (and its model) must be identical - the point
    # is that identity did not leak, not that routing differed.
    assert first["served_model"] == second["served_model"] == "basic-chat"


def test_concurrent_completions_each_get_their_own_routing(sr_base_url):
    """Queued routing decisions must still complete, not time out mid-decision.

    The router classifies on CPU inside the ext_proc call, so concurrent
    requests queue behind one another there. Envoy cancels the routing stream
    at its per-message deadline and answers 504 without ever reaching the
    backend, which is invisible to a one-request-at-a-time suite. The free
    tier is gated by authz alone, so every one of these routes to the same
    model whatever the classifier makes of the prompt.
    """
    prompts = [f"tell me something interesting about topic {i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        reflected = list(
            pool.map(lambda prompt: _call(sr_base_url, "free-tenant", prompt), prompts)
        )
    assert [item["echo"] for item in reflected] == prompts
    assert [item["served_model"] for item in reflected] == ["basic-chat"] * len(prompts)
    assert [item["reasoning"] for item in reflected] == [False] * len(prompts)


class _SchemaProbe(dspy.Signature):
    """Probe signature whose server-enforced schema must survive the router."""

    query: str = dspy.InputField(desc="text to label")
    entities: str = dspy.OutputField(desc="entities as text|type|confidence lines")


# What ``signature_response_format`` builds for ``_SchemaProbe``: the whole
# payload, written out, so a router that forwards a hollowed-out
# ``{"type": "json_schema"}`` cannot read as a pass.
_PROBE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "_SchemaProbe",
        "strict": True,
        "schema": {
            "type": "object",
            "title": "_SchemaProbe",
            "properties": {"entities": {"title": "Entities", "type": "string"}},
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


def _call_with_response_format(
    base_url: str, tenant_id: str, prompt: str, response_format: dict
) -> dict:
    """Route a completion carrying ``response_format``; return the reflection."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=_semantic_router_config(base_url),
        tenant_id=tenant_id,
        tier=_TIERS.get(tenant_id, _FALLBACK_TIER),
    )
    lm = create_dspy_lm(routed)
    lm.cache = False
    out = lm(prompt, response_format=response_format)
    item = out[0] if isinstance(out, list) else out
    content = (
        item.get("text") or item.get("content") if isinstance(item, dict) else item
    )
    return json.loads(content)


def test_signature_response_format_is_the_probe_golden():
    """The probe's payload is what production builds, not a hand-written twin."""
    assert signature_response_format(_SchemaProbe) == _PROBE_RESPONSE_FORMAT


def test_json_schema_response_format_survives_the_router(sr_base_url):
    """The backend must receive the schema byte-for-byte, not just its type.

    ``served_model`` pins that the model-rewriting path ran on this request:
    that is the path that re-serializes the body, so a reflection showing the
    client's own ``auto`` would prove nothing about it.
    """
    reflected = _call_with_response_format(
        sr_base_url,
        "free-tenant",
        "label the entities in this sentence",
        signature_response_format(_SchemaProbe),
    )
    assert reflected["served_model"] == "basic-chat"
    assert reflected["response_format"] == _PROBE_RESPONSE_FORMAT


def test_request_without_response_format_stays_without_one(sr_base_url):
    """The router must not invent a response_format the client never sent."""
    reflected = _call(sr_base_url, "free-tenant", "label the entities in this sentence")
    assert reflected["served_model"] == "basic-chat"
    assert reflected["response_format"] is None
