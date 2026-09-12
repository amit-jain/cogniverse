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

pytestmark = pytest.mark.integration

SR_CONFIG = Path(__file__).resolve().parent / "_sr_stack" / "sr-config.yaml"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"

TENANT_ID = "tierjoin:production"
CREATED_AT = 1757000000000
PROMPT = "summarise this paragraph"


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


def _routing_decisions(container: str) -> list[dict]:
    """Every ``routing_decision`` the stack's router has logged so far."""
    logs = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True, timeout=60
    )
    decisions = []
    for line in (logs.stdout + logs.stderr).splitlines():
        line = line.strip()
        if '"msg":"routing_decision"' not in line:
            continue
        try:
            decisions.append(json.loads(line[line.index("{") :]))
        except ValueError:
            continue
    return decisions


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


def _route_one_completion(config_manager) -> dict:
    """One completion through the production seam; the stub's reflection."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    with routed_lm_context_for(
        config_manager, TENANT_ID, "summarizer_agent", endpoint=endpoint
    ):
        lm = dspy.settings.lm
        lm.cache = False
        out = lm(PROMPT)
    item = out[0] if isinstance(out, list) else out
    content = (
        item.get("text") or item.get("content") if isinstance(item, dict) else item
    )
    return json.loads(content)


async def test_the_stored_tier_decides_the_router(tier_stack):
    """Set each tier through the admin route; the router's decision follows."""
    config_manager = tier_stack["config_manager"]
    tenant_manager = tier_stack["tenant_manager"]
    container = tier_stack["router_container"]
    expected_by_tier = _decisions_by_tier()

    # The stack's router binds a decision for exactly the vocabulary the
    # runtime can emit -- otherwise a tier below would route by fall-through.
    assert set(expected_by_tier) == set(ROUTER_TIERS)

    # Every tier in turn, then back to the first, so the last leg is a change
    # away from a tier that was already cached.
    walk = sorted(ROUTER_TIERS)
    walk.append(walk[0])

    before = len(_routing_decisions(container))
    started = time.monotonic()
    observed = []
    for tier in walk:
        assert await _set_tier(tenant_manager, tier) == {
            "tenant_id": TENANT_ID,
            "tier": tier,
        }
        reflection = _route_one_completion(config_manager)
        observed.append(
            {
                "tier": reflection["routing_headers"]["x-authz-user-groups"],
                "model": reflection["served_model"],
                "reasoning": reflection["reasoning"],
                "user_id": reflection["routing_headers"]["x-authz-user-id"],
            }
        )
    elapsed = time.monotonic() - started

    assert observed == [
        {
            "tier": tier,
            "model": expected_by_tier[tier]["model"],
            "reasoning": expected_by_tier[tier]["reasoning"],
            "user_id": TENANT_ID,
        }
        for tier in walk
    ]

    # The router's own account of the same requests.
    logged = _routing_decisions(container)[before:]
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
        for tier in walk
    ]

    # Every tier change was read back inside one reader TTL, so the in-process
    # invalidation is what carried it -- not an entry that happened to expire.
    assert elapsed < TENANT_TIER_TTL_S
