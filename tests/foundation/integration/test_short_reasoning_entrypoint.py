"""Short free-form calls through the real router on the short-reasoning entry.

The router is the stack's twin of the chart (``_sr_stack/sr-config.yaml``,
pinned equal to the chart's rendered recipes). Every assertion reads what the
router did: the decision it names on the answered response, the model and
reasoning flag the stub backend reflects, the router's own
``routing_decision`` log line, and the response path it reports.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import uuid

import pytest
import requests

from cogniverse_foundation.config.semantic_router import (
    create_routed_lm,
    resolve_semantic_router_headers,
)
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

pytestmark = pytest.mark.integration

_CALL_SITE = "deep_research_decomposition"

# The fastest decision the domain classifier made on this stack over 1,060
# auto-alias decisions in two runs (the search rewrite and the deep-research
# decompose/evaluate prompts, 50 ms stub): 146 ms. The tier-only entries
# decided in 0-7 ms over 1,000 decisions (classification 250,
# short-reasoning 750), p50 0 ms. A decision under the classifier's fastest
# ran no classifier.
_FASTEST_CLASSIFIER_DECISION_MS = 146
_TIER_ONLY_DECISION_P50_MS = 1

_EXPECTED = {
    "pro": ("short-reasoning-pro", "pro-reasoning", True, "teacher"),
    "free": ("short-reasoning-free", "basic-chat", False, "student"),
    "default": ("short-reasoning-base", "basic-chat", False, "student"),
}


def _config(stack) -> SemanticRouterConfig:
    return SemanticRouterConfig(enabled=True, semantic_router_url=stack["base_url"])


def _post(stack, tenant_id: str, tier: str, prompt: str) -> requests.Response:
    config = _config(stack)
    response = requests.post(
        f"{stack['base_url'].rstrip('/')}/chat/completions",
        json={
            "model": config.short_reasoning_model.split("/", 1)[1],
            "messages": [{"role": "user", "content": prompt}],
        },
        headers=resolve_semantic_router_headers(config, tenant_id, tier),
        timeout=30,
    )
    assert response.status_code == 200, response.text[:400]
    return response


def _reflected(response: requests.Response) -> dict:
    return json.loads(response.json()["choices"][0]["message"]["content"])


def _lm(stack, tier: str, tenant_id: str, timeout: float = 10.0):
    lm = create_routed_lm(
        LLMEndpointConfig(
            model="unused",
            api_base="http://unused:1/v1",
            num_retries=2,
            request_timeout=timeout,
        ),
        _config(stack),
        tenant_id=tenant_id,
        tier=tier,
        call_site=_CALL_SITE,
    )
    lm.cache = False
    return lm


def _short_reasoning_decisions(stack) -> list[dict]:
    logs = subprocess.run(
        ["docker", "logs", stack["router_container"]],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    decisions = []
    for line in (logs.stdout + logs.stderr).splitlines():
        if '"routing_decision"' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("original_model") == "cogniverse-short-reasoning":
            decisions.append(record)
    return decisions


@pytest.fixture(scope="module")
def routed_by_tier(semantic_router_stack):
    """One fresh short-reasoning request per tier, each its own tenant."""
    return {
        tier: _post(
            semantic_router_stack,
            f"short-{tier}-{uuid.uuid4()}",
            tier,
            f"decompose: effects of rain on a dirt track {uuid.uuid4()}",
        )
        for tier in _EXPECTED
    }


@pytest.mark.parametrize("tier", sorted(_EXPECTED))
def test_each_tier_lands_on_its_tier_only_decision(routed_by_tier, tier):
    decision, model, reasoning, backend = _EXPECTED[tier]
    response = routed_by_tier[tier]
    reflected = _reflected(response)
    assert response.headers["x-vsr-selected-decision"] == decision
    assert reflected["served_model"] == model
    assert reflected["reasoning"] is reasoning
    assert reflected["backend_tag"] == backend


def test_no_decision_on_the_entry_spends_classifier_time(
    semantic_router_stack, routed_by_tier
):
    decisions = _short_reasoning_decisions(semantic_router_stack)
    latencies = [record["routing_latency_ms"] for record in decisions]
    assert {record["decision"] for record in decisions} == {
        "short-reasoning-pro",
        "short-reasoning-free",
        "short-reasoning-base",
    }
    assert max(latencies) < _FASTEST_CLASSIFIER_DECISION_MS
    assert statistics.median(latencies) <= _TIER_ONLY_DECISION_P50_MS


@pytest.mark.parametrize("tier", sorted(_EXPECTED))
def test_the_production_lm_reaches_the_tiers_model(semantic_router_stack, tier):
    _, model, reasoning, backend = _EXPECTED[tier]
    lm = _lm(semantic_router_stack, tier, f"short-lm-{tier}-{uuid.uuid4()}")
    out = lm(f"decompose: a wedding first dance {uuid.uuid4()}")
    item = out[0] if isinstance(out, list) else out
    content = item.get("text") if isinstance(item, dict) else item
    reflected = json.loads(content)
    assert lm.model == "openai/cogniverse-short-reasoning"
    assert (reflected["served_model"], reflected["reasoning"]) == (model, reasoning)
    assert reflected["backend_tag"] == backend


def test_the_same_request_is_answered_from_the_cache(semantic_router_stack):
    tenant = f"short-cache-{uuid.uuid4()}"
    prompt = f"decompose: cache probe {uuid.uuid4()}"
    first = _post(semantic_router_stack, tenant, "pro", prompt)
    second = _post(semantic_router_stack, tenant, "pro", prompt)
    assert first.headers["x-vsr-response-path"] == "upstream"
    assert second.headers["x-vsr-response-path"] == "cache"
    assert _reflected(first)["call_index"] == _reflected(second)["call_index"]


def test_the_cache_never_answers_one_tenant_from_another(semantic_router_stack):
    prompt = f"decompose: tenant isolation probe {uuid.uuid4()}"
    first = _post(semantic_router_stack, f"short-a-{uuid.uuid4()}", "pro", prompt)
    second = _post(semantic_router_stack, f"short-b-{uuid.uuid4()}", "pro", prompt)
    assert first.headers["x-vsr-response-path"] == "upstream"
    assert second.headers["x-vsr-response-path"] == "upstream"
    assert _reflected(first)["call_index"] != _reflected(second)["call_index"]


@pytest.mark.parametrize("fault,status", [("status:503", 503), ("trickle:5", 408)])
def test_a_teacher_failure_on_pro_is_answered_by_the_student(
    semantic_router_stack, fault, status
):
    prompt = f"TEACHER_FAULT:{fault}|{uuid.uuid4()}"
    lm = _lm(
        semantic_router_stack,
        "pro",
        f"short-fallback-{uuid.uuid4()}",
        timeout=2.0 if fault.startswith("trickle") else 10.0,
    )
    response = lm.forward(prompt=prompt)
    body = response.model_dump()
    reflected = json.loads(body["choices"][0]["message"]["content"])
    assert {
        key: body[key]
        for key in ("tier_degraded", "upstream_status", "upstream_exception_type")
    } == {
        "tier_degraded": "pro_model_unavailable",
        "upstream_status": status,
        "upstream_exception_type": "UpstreamUnavailable",
    }
    assert (reflected["backend_tag"], reflected["served_model"]) == (
        "student",
        "basic-chat",
    )
    assert reflected["echo"] == prompt
