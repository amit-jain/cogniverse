"""The stored tier decides the router, end to end.

One sequence across every boundary the feature spans: the admin route writes
the tier to a real Vespa config store, the production LM seam reads it back for
the same tenant, and a real chat completion crosses Envoy into the vLLM
semantic router, whose decision the stub upstream reflects and whose
``routing_decision`` log line names.

Each tier is set and exercised in turn inside one reader TTL, so a decision
that follows a tier change proves the write invalidated the cached read --
waiting out the TTL would prove only that entries expire. The expected
decision, model and reasoning flag for a tier are read from the router config
the stack runs, never restated here.

The router caches each response under the decision and the tenant identity on
the exact request, so returning to the first tier with the same prompt is
answered from that entry: no new decision is logged and the upstream is not
called. A different prompt at that tier is routed again.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import dspy
import httpx
import pytest
import yaml
from prometheus_client.parser import text_string_to_metric_families

from cogniverse_foundation.config.lm_response_cache import (
    TenantScopedLMCache,
    lm_response_cache,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.semantic_router import routed_lm_context_for
from cogniverse_foundation.config.tenant_tiers import TENANT_TIER_TTL_S
from cogniverse_foundation.config.unified_config import (
    ROUTER_TIERS,
    LLMEndpointConfig,
    SemanticRouterConfig,
    SystemConfig,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.semantic_router_stack import CHART_VALUES

pytestmark = pytest.mark.integration

SR_CONFIG = Path(__file__).resolve().parent / "_sr_stack" / "sr-config.yaml"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"

TENANT_ID = "tierjoin:production"
CREATED_AT = 1757000000000
PROMPT = "summarise this paragraph"
FRESH_PROMPT = "summarise the next paragraph"
IN_PROCESS_PROMPT = "summarise this paragraph once per process"


def _decisions_by_tier() -> dict[str, dict]:
    """tier -> the decision the stack's router binds for it.

    Walks the shipped role bindings (Group name -> role) and the decisions
    gated on that role, so the expectation moves with the config the router
    actually loaded instead of being restated as a literal here.
    """
    config = yaml.safe_load(SR_CONFIG.read_text())
    routing = config["routing"]
    role_by_group = {
        subject["name"]: binding["role"]
        for binding in routing["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }
    by_tier: dict[str, dict] = {}
    for decision in routing["decisions"]:
        conditions = decision["rules"]["conditions"]
        roles = {c["name"] for c in conditions if c["type"] == "authz"}
        # The tier's own decision is the one gated on the tier alone; the
        # content-gated siblings need a classifier verdict this prompt does
        # not carry.
        if len(conditions) != 1 or not roles:
            continue
        (role,) = roles
        tier = next(g for g, r in role_by_group.items() if r == role)
        model_ref = decision["modelRefs"][0]
        by_tier[tier] = {
            "decision": decision["name"],
            "model": model_ref["model"],
            "reasoning": bool(model_ref.get("use_reasoning", False)),
        }
    return by_tier


@pytest.fixture(scope="module")
def tier_stack(semantic_router_stack, shared_vespa):
    """The admin route, the seam and the router over one real config store."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_runtime.admin import tenant_manager as tm

    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=shared_vespa["http_port"]
    )
    config_manager = ConfigManager(store=store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_vespa["http_port"],
            semantic_router=SemanticRouterConfig(
                enabled=True,
                semantic_router_url=semantic_router_stack["base_url"],
            ),
        )
    )

    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(FilesystemSchemaLoader(SCHEMAS_DIR))

    org_id, tenant_name = TENANT_ID.split(":")
    assert tm.get_backend().create_metadata_document(
        schema="tenant_metadata",
        doc_id=TENANT_ID,
        fields={
            "tenant_full_id": TENANT_ID,
            "org_id": org_id,
            "tenant_name": tenant_name,
            "created_at": CREATED_AT,
            "created_by": "tier-join",
            "status": "active",
            "schemas_deployed": [],
        },
    )

    yield {
        "config_manager": config_manager,
        "tenant_manager": tm,
        "router_container": semantic_router_stack["router_container"],
    }

    tm.get_backend().delete_metadata_document(
        schema="tenant_metadata", doc_id=TENANT_ID
    )
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    BackendRegistry.get_instance().clear_instances()


def _router_events(container: str, msg: str) -> list[dict]:
    """Every ``msg`` event the stack's router has logged so far."""
    logs = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True, timeout=60
    )
    events = []
    for line in (logs.stdout + logs.stderr).splitlines():
        line = line.strip()
        if f'"msg":"{msg}"' not in line:
            continue
        try:
            events.append(json.loads(line[line.index("{") :]))
        except ValueError:
            continue
    return events


def _router_counts(container: str) -> dict[str, float]:
    """Routed calls and response-cache hits from the router's metrics endpoint."""
    port = yaml.safe_load(CHART_VALUES.read_text())["semanticRouter"]["router"][
        "metricsPort"
    ]
    response = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "10",
            f"http://localhost:{port}/metrics",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    samples = [
        sample
        for family in text_string_to_metric_families(response.stdout)
        for sample in family.samples
    ]
    return {
        "routed": sum(
            sample.value
            for sample in samples
            if sample.name == "llm_model_routing_modifications_total"
        ),
        "cache_hits": sum(
            sample.value
            for sample in samples
            if sample.name == "llm_cache_plugin_hits_total"
            and sample.labels["plugin_type"] == "response_cache"
        ),
    }


async def _set_tier(tenant_manager, tier: str) -> dict:
    transport = httpx.ASGITransport(app=tenant_manager.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://tier-join"
    ) as client:
        response = await client.put(
            f"/admin/tenants/{TENANT_ID}/tier", json={"tier": tier}
        )
    assert response.status_code == 200, response.text
    return response.json()


def _route_one_completion(
    config_manager, prompt: str, in_process_cache: TenantScopedLMCache
) -> tuple[dict, dict]:
    """One completion through the production seam, answered in process only
    from ``in_process_cache``.

    Returns the stub's reflection and the router's own account of the
    response: the path it was served from, the cache-hit flag and the decision.
    """
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    with routed_lm_context_for(
        config_manager, TENANT_ID, "summarizer_agent", endpoint=endpoint
    ):
        lm = dspy.settings.lm
        lm.response_cache = in_process_cache
        out = lm(prompt)
        headers = lm.history[-1]["response"]._hidden_params["additional_headers"]
    item = out[0] if isinstance(out, list) else out
    content = (
        item.get("text") or item.get("content") if isinstance(item, dict) else item
    )
    return json.loads(content), {
        "response_path": headers["llm_provider-x-vsr-response-path"],
        "cache_hit": headers.get("llm_provider-x-vsr-cache-hit"),
        "decision": headers["llm_provider-x-vsr-selected-decision"],
    }


def _leg_cache() -> TenantScopedLMCache:
    """An empty in-process cache, so the leg's request reaches the router."""
    return TenantScopedLMCache(ttl_seconds=3600, max_entries=1024)


async def test_the_stored_tier_decides_the_router(tier_stack):
    """Set each tier through the admin route; the router's decision follows."""
    config_manager = tier_stack["config_manager"]
    tenant_manager = tier_stack["tenant_manager"]
    container = tier_stack["router_container"]
    expected_by_tier = _decisions_by_tier()

    # The stack's router binds a decision for exactly the vocabulary the
    # runtime can emit -- otherwise a tier below would route by fall-through.
    assert set(expected_by_tier) == set(ROUTER_TIERS)

    # Every tier in turn, then back to the first: that leg repeats the first
    # request byte for byte, so the router answers it from its response cache.
    walk = sorted(ROUTER_TIERS)
    walk.append(walk[0])

    decisions_before = len(_router_events(container, "routing_decision"))
    usage_before = len(_router_events(container, "llm_usage"))
    counts = [_router_counts(container)]
    started = time.monotonic()
    legs = []
    for tier in walk:
        assert await _set_tier(tenant_manager, tier) == {
            "tenant_id": TENANT_ID,
            "tier": tier,
        }
        legs.append(_route_one_completion(config_manager, PROMPT, _leg_cache()))
        counts.append(_router_counts(container))
    elapsed = time.monotonic() - started

    # Same tenant, same tier, a different prompt: routed, not served from cache.
    legs.append(_route_one_completion(config_manager, FRESH_PROMPT, _leg_cache()))
    counts.append(_router_counts(container))
    tiers = [*walk, walk[0]]
    prompts = [PROMPT] * len(walk) + [FRESH_PROMPT]
    reflections = [reflection for reflection, _ in legs]
    served = [served_leg for _, served_leg in legs]
    cached_leg = len(walk) - 1
    deltas = [
        {name: after[name] - before[name] for name in before}
        for before, after in zip(counts, counts[1:])
    ]
    assert deltas[cached_leg]["cache_hits"] == 1
    assert deltas[-1]["routed"] == 1
    assert deltas == [
        {"routed": int(index != cached_leg), "cache_hits": int(index == cached_leg)}
        for index in range(len(tiers))
    ]

    assert [
        {
            "tier": reflection["routing_headers"]["x-authz-user-groups"],
            "model": reflection["served_model"],
            "reasoning": reflection["reasoning"],
            "user_id": reflection["routing_headers"]["x-authz-user-id"],
            "echo": reflection["echo"],
        }
        for reflection in reflections
    ] == [
        {
            "tier": tier,
            "model": expected_by_tier[tier]["model"],
            "reasoning": expected_by_tier[tier]["reasoning"],
            "user_id": TENANT_ID,
            "echo": prompt,
        }
        for tier, prompt in zip(tiers, prompts)
    ]

    # The router's response headers: every leg names its tier's decision, and
    # only the repeated first request comes from the cache.
    assert served == [
        {
            "response_path": "cache" if index == cached_leg else "upstream",
            "cache_hit": "true" if index == cached_leg else None,
            "decision": expected_by_tier[tier]["decision"],
        }
        for index, tier in enumerate(tiers)
    ]

    # A cache hit replays the stored body verbatim, including the stub's
    # per-process call counter; every other leg is one fresh upstream call.
    assert reflections[cached_leg] == reflections[0]
    first_call = reflections[0]["call_index"]
    routed_calls = iter(range(first_call, first_call + len(tiers) - 1))
    assert [reflection["call_index"] for reflection in reflections] == [
        first_call if index == cached_leg else next(routed_calls)
        for index in range(len(tiers))
    ]

    # No other tier is answered out of the first tier's entry.
    assert [
        (tier, reflection["served_model"], served_leg["response_path"])
        for tier, reflection, served_leg in zip(tiers, reflections, served)
        if tier != walk[0]
    ] == [(tier, expected_by_tier[tier]["model"], "upstream") for tier in walk[1:-1]]

    # The router's own log: a decision for every routed leg and none for the
    # cache hit, whose usage record is the only one marked as served from cache.
    routed_tiers = [*walk[:-1], walk[0]]
    logged = _router_events(container, "routing_decision")[decisions_before:]
    assert [
        {
            "decision": entry["decision"],
            "selected_model": entry["selected_model"],
            "original_model": entry["original_model"],
            "reason_code": entry["reason_code"],
            "reasoning_enabled": entry["reasoning_enabled"],
        }
        for entry in logged
    ] == [
        {
            "decision": expected_by_tier[tier]["decision"],
            "selected_model": expected_by_tier[tier]["model"],
            "original_model": "auto",
            "reason_code": "entrypoint_routing",
            "reasoning_enabled": expected_by_tier[tier]["reasoning"],
        }
        for tier in routed_tiers
    ]
    usage = _router_events(container, "llm_usage")[usage_before:]
    assert [
        {
            "model": entry["model"],
            "cache_hit": entry.get("cache_hit", False),
            "from_cache": entry.get("from_cache", False),
        }
        for entry in usage
    ] == [
        {
            "model": expected_by_tier[tier]["model"],
            "cache_hit": index == cached_leg,
            "from_cache": index == cached_leg,
        }
        for index, tier in enumerate(tiers)
    ]

    # Every tier change was read back inside one reader TTL, so the in-process
    # invalidation is what carried it -- not an entry that happened to expire.
    assert elapsed < TENANT_TIER_TTL_S


async def test_a_repeat_answered_in_process_never_reaches_the_router(tier_stack):
    """The process LM response cache answers a tenant's byte-identical repeat,
    so the router records neither a routed call nor a cache hit for it."""
    config_manager = tier_stack["config_manager"]
    container = tier_stack["router_container"]
    tier = sorted(ROUTER_TIERS)[0]
    assert await _set_tier(tier_stack["tenant_manager"], tier) == {
        "tenant_id": TENANT_ID,
        "tier": tier,
    }
    counts = [_router_counts(container)]
    legs = []
    for _ in range(2):
        legs.append(
            _route_one_completion(
                config_manager, IN_PROCESS_PROMPT, lm_response_cache()
            )
        )
        counts.append(_router_counts(container))
    deltas = [
        {name: after[name] - before[name] for name in before}
        for before, after in zip(counts, counts[1:])
    ]
    assert deltas == [{"routed": 1, "cache_hits": 0}, {"routed": 0, "cache_hits": 0}]
    assert legs[0][0]["echo"] == IN_PROCESS_PROMPT
    assert legs[0][1] == {
        "response_path": "upstream",
        "cache_hit": None,
        "decision": _decisions_by_tier()[tier]["decision"],
    }
    assert legs[1] == legs[0]
