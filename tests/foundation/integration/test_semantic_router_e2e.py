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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
import pytest
import requests
import yaml

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import (
    apply_semantic_routing,
    resolve_semantic_router_headers,
)
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

_STACK_ROUTER_CONFIG = Path(__file__).parent / "_sr_stack" / "sr-config.yaml"


def _response_cache_plugin() -> dict:
    """The response-cache policy the stack's router is running.

    Read from the config the container mounts rather than restated, so a
    changed header name or mode reaches these tests instead of being asserted
    against a copy of the old contract.
    """
    decisions = yaml.safe_load(_STACK_ROUTER_CONFIG.read_text())["routing"]["decisions"]
    policies = [
        plugin["configuration"]
        for decision in decisions
        for plugin in decision["plugins"]
        if plugin["type"] == "response_cache"
    ]
    assert policies == [policies[0]] * len(decisions)
    return policies[0]


def _chat_body(prompt: str, **fields) -> dict:
    """The body the runtime's routed endpoint puts on the wire, minus DSPy.

    ``routed_model`` is what ``apply_semantic_routing`` stamps; litellm sends
    it without the provider prefix.
    """
    routed_model = _semantic_router_config("http://unused").routed_model
    return {
        "model": routed_model.split("/")[-1],
        "messages": [{"role": "user", "content": prompt}],
        **fields,
    }


def _post(
    base_url: str, tenant_id: str, body: dict, *, cache_control: str | None = None
) -> requests.Response:
    """One completion through Envoy carrying the production authz headers."""
    config = _semantic_router_config(base_url)
    headers = dict(resolve_semantic_router_headers(config, tenant_id) or {})
    if cache_control is not None:
        headers[_response_cache_plugin()["request_controls"]["header"]] = cache_control
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body,
        headers=headers,
        timeout=30,
    )
    assert response.status_code == 200, response.text[:400]
    return response


def _reflected(response: requests.Response) -> dict:
    return json.loads(response.json()["choices"][0]["message"]["content"])


def _served_from(response: requests.Response) -> str:
    """``cache`` or ``upstream`` - the router's own account of the path."""
    return response.headers["x-vsr-response-path"]


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

    Every routing decision attaches the router's response cache
    (``plugins: - type: response_cache`` on each decision in
    ``charts/cogniverse/files/semantic-router/config.yaml``) at
    ``scope: user``, which partitions entries by the tenant identity in
    ``x-authz-user-id``. Same tier means the same decision and the same cache,
    and an identical prompt means an identical request fingerprint, so the
    partition is the only thing keeping the two apart.

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


class TestTheResponseCacheReusesOnlyAnIdenticalRequest:
    """The router's per-decision cache, exercised against the real router.

    Every decision runs ``mode: exact``: a hit requires a SHA-256 match over
    the whole normalized request (messages, response_format, tools, sampling
    parameters, model) inside a partition keyed by the tenant identity. The
    stub's ``call_index`` counts requests that reached the backend, so a
    repeated index is the cache answering and a fresh one is an upstream call
    - which is what separates reuse from the backend answering twice.
    """

    def test_the_same_request_twice_is_answered_from_the_cache(self, sr_base_url):
        body = _chat_body("what is the northern outlook for the third quarter")

        first = _post(sr_base_url, "free-tenant", body)
        second = _post(sr_base_url, "free-tenant", body)

        assert _served_from(first) == "upstream"
        assert _served_from(second) == "cache"
        assert second.headers["x-vsr-cache-hit"] == "true"
        assert _reflected(second) == _reflected(first)

    def test_one_trailing_digit_apart_is_eight_separate_upstream_calls(
        self, sr_base_url
    ):
        """The shape that reddened this suite at ``similarity_threshold: 0.95``:
        eight prompts differing only by a trailing digit came back carrying
        each other's content."""
        prompts = [f"describe the state of affairs in region {i}" for i in range(8)]

        responses = [_post(sr_base_url, "free-tenant", _chat_body(p)) for p in prompts]
        reflected = [_reflected(response) for response in responses]

        assert [item["echo"] for item in reflected] == prompts
        assert [_served_from(response) for response in responses] == ["upstream"] * 8
        assert len({item["call_index"] for item in reflected}) == 8

    def test_the_same_prompt_with_and_without_a_schema_never_cross(self, sr_base_url):
        """A ``response_format`` is part of the key, so the request that sent
        none cannot be answered out of the entry written by the one that did.
        """
        prompt = "list the entities in this sentence and nothing else"
        schema = signature_response_format(_SchemaProbe)

        with_schema = _post(
            sr_base_url, "free-tenant", _chat_body(prompt, response_format=schema)
        )
        without_schema = _post(sr_base_url, "free-tenant", _chat_body(prompt))

        assert _reflected(with_schema)["response_format"] == _PROBE_RESPONSE_FORMAT
        assert _reflected(without_schema)["response_format"] is None
        assert _served_from(without_schema) == "upstream"
        assert (
            _reflected(without_schema)["call_index"]
            != _reflected(with_schema)["call_index"]
        )

    def test_two_tenants_asking_the_same_thing_each_reach_the_backend(
        self, sr_base_url
    ):
        """Same tier, so the same decision and the same cache; the partition is
        the tenant identity, and the stub reflects the identity it was called
        with, so a crossed answer names the wrong tenant."""
        body = _chat_body("compare last year's rainfall across the northern districts")

        first = _post(sr_base_url, "free-tenant", body)
        second = _post(sr_base_url, "another-free-tenant", body)

        assert _reflected(first)["routing_headers"]["x-authz-user-id"] == "free-tenant"
        assert (
            _reflected(second)["routing_headers"]["x-authz-user-id"]
            == "another-free-tenant"
        )
        assert _served_from(second) == "upstream"
        assert _reflected(second)["call_index"] != _reflected(first)["call_index"]
        assert _reflected(first)["served_model"] == "basic-chat"
        assert _reflected(second)["served_model"] == "basic-chat"

    def test_the_same_prompt_at_a_different_temperature_is_a_separate_entry(
        self, sr_base_url
    ):
        """Generation parameters are part of the key: a cached answer sampled
        at one temperature must not be replayed for a request that asked for
        another."""
        prompt = "name one interesting property of the northern region"

        cold = _post(sr_base_url, "free-tenant", _chat_body(prompt, temperature=0.0))
        hot = _post(sr_base_url, "free-tenant", _chat_body(prompt, temperature=0.9))

        assert _reflected(cold)["temperature"] == 0.0
        assert _reflected(hot)["temperature"] == 0.9
        assert _served_from(hot) == "upstream"
        assert _reflected(hot)["call_index"] != _reflected(cold)["call_index"]

    def test_an_entry_older_than_the_accepted_age_is_refetched(self, sr_base_url):
        """Entry age bounds reuse. The shipped ``ttl_seconds`` is the ceiling;
        a request may narrow it with the one client directive the decision
        allows, and this drives that read path rather than reasoning about it.
        """
        body = _chat_body("how long does the northern region keep its records")
        allowed = _response_cache_plugin()["request_controls"]["allowed"]
        assert allowed == ["max-age"]

        stored = _post(sr_base_url, "free-tenant", body)
        fresh = _post(sr_base_url, "free-tenant", body, cache_control="max-age=30")
        time.sleep(2)
        aged = _post(sr_base_url, "free-tenant", body, cache_control="max-age=1")

        assert _served_from(fresh) == "cache"
        assert _reflected(fresh)["call_index"] == _reflected(stored)["call_index"]
        assert _served_from(aged) == "upstream"
        assert _reflected(aged)["call_index"] != _reflected(stored)["call_index"]
