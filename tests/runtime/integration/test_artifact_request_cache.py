"""load_for_request caches the per-request artefact reads (real Phoenix).

The dispatch hot path called get_artefact_state + get_dataset on every
request; both change only on promote/retire. A short-TTL cache removes the
repeat reads and is invalidated when the state is written.
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def manager(phoenix_container):
    tenant = f"ac{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant)


def _spy(manager) -> dict:
    counts = {"state": 0, "dataset": 0}
    orig_state = manager.get_artefact_state
    orig_ds = manager._provider.datasets.get_dataset

    async def _state(*a, **k):
        counts["state"] += 1
        return await orig_state(*a, **k)

    async def _ds(*a, **k):
        counts["dataset"] += 1
        return await orig_ds(*a, **k)

    manager.get_artefact_state = _state
    manager._provider.datasets.get_dataset = _ds
    return counts


@pytest.mark.asyncio
async def test_repeat_request_is_served_from_cache(manager):
    await manager.save_prompts("search_agent", {"system": "BASELINE"})

    counts = _spy(manager)
    out1 = await manager.load_for_request("search_agent", request_seed="r1")
    assert out1["served_from"] == "default"
    assert out1["prompts"] == {"system": "BASELINE"}
    assert counts["state"] == 1
    assert counts["dataset"] >= 1

    counts["state"] = 0
    counts["dataset"] = 0
    out2 = await manager.load_for_request("search_agent", request_seed="r1")

    assert out2 == out1
    # Fully served from cache — no repeat artefact reads.
    assert counts["state"] == 0
    assert counts["dataset"] == 0


@pytest.mark.asyncio
async def test_promote_invalidates_request_cache(manager):
    await manager.save_prompts("search_agent", {"system": "BASELINE"})

    counts = _spy(manager)
    await manager.load_for_request("search_agent", request_seed="r1")
    assert counts["state"] == 1  # first read populates the cache

    # Promote writes state -> must invalidate the request cache.
    await manager.promote_to_canary("search_agent", version=1, traffic_pct=1)

    counts["state"] = 0
    await manager.load_for_request("search_agent", request_seed="r1")
    assert counts["state"] >= 1, "promote must invalidate the cached artefact state"
