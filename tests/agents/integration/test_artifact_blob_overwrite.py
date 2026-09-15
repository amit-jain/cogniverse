"""Serving-blob publication against real Phoenix.

save_blob publishes each revision into one of two alternating single-row
datasets, so a reader always resolves a complete committed revision, the store
never holds more than two, and a failed or interrupted publication leaves the
committed revision in place.
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration

_UNUSED_GRPC_ENDPOINT = "http://127.0.0.1:1"


def _is_publication(method: str, path: str) -> bool:
    """The one request that writes a dataset row — not any POST the client makes."""
    return method == "POST" and path.startswith("/v1/datasets/upload")


@pytest.fixture
def manager(phoenix_container) -> ArtifactManager:
    tenant_id = f"blob_{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_save_blob_publishes_single_row_revisions(manager):
    await manager.save_blob("model", "k1", "v1")
    await manager.save_blob("model", "k1", "v2")

    assert await manager.load_blob("model", "k1") == "v2"

    slots = {
        parity: (
            await manager._provider.datasets.get_dataset(
                name=manager._blob_slot_name("model", "k1", parity)
            )
        ).to_dict("records")
        for parity in (0, 1)
    }
    assert slots == {
        0: [
            {
                "input": {"content": "v2", "blob_revision": "2"},
                "output": {},
                "metadata": {},
            }
        ],
        1: [
            {
                "input": {"content": "v1", "blob_revision": "1"},
                "output": {},
                "metadata": {},
            }
        ],
    }


@pytest.mark.asyncio
async def test_gateway_reload_serves_rewritten_thresholds(manager):
    """Re-running _load_artifact on an ALREADY-LOADED gateway agent must apply
    a rewritten gateway_thresholds blob — the contract behind the dispatcher's
    periodic re-load, which is how a warm pod starts serving a recalibration
    without a restart."""
    import asyncio
    import json
    from types import SimpleNamespace

    from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps

    await manager.save_blob(
        "config",
        "gateway_thresholds",
        json.dumps({"fast_path_confidence_threshold": 0.5, "gliner_threshold": 0.49}),
    )

    agent = GatewayAgent(deps=GatewayDeps())
    agent.telemetry_manager = SimpleNamespace(
        get_provider=lambda tenant_id: manager._provider
    )
    agent._artifact_tenant_id = manager._tenant_id

    # Off-loop, exactly as the dispatcher runs it.
    await asyncio.to_thread(agent._load_artifact)
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.49

    await manager.save_blob(
        "config",
        "gateway_thresholds",
        json.dumps({"fast_path_confidence_threshold": 0.35, "gliner_threshold": 0.2}),
    )

    await asyncio.to_thread(agent._load_artifact)
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.35
    assert agent.deps.gliner_threshold == 0.2


def _manager_at(endpoint, tenant):
    """A publisher whose dataset calls go through ``endpoint``.

    Dataset publication is HTTP-only; the gRPC endpoint is required by the
    provider contract and never carries a span here.
    """
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant,
            "http_endpoint": endpoint,
            "grpc_endpoint": _UNUSED_GRPC_ENDPOINT,
        }
    )
    return ArtifactManager(provider, tenant_id=tenant)


@pytest.mark.asyncio
async def test_replacement_readers_keep_committed_blob_until_publication(
    manager, phoenix_container
):
    import asyncio
    import threading

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    await manager.save_blob("config", "quotas", '{"user":1,"admin":2}')
    entered, release = threading.Event(), threading.Event()
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:

        def intercept(method, path, body):
            if _is_publication(method, path):
                entered.set()
                if not release.wait(15):
                    return 504, {"error": "publication barrier expired"}
            return None

        proxy.intercept = intercept
        writer = _manager_at(proxy.url, manager._tenant_id)
        write = asyncio.create_task(
            writer.save_blob("config", "quotas", '{"user":7,"admin":2}')
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5) is True
            observed = await asyncio.gather(
                *[manager.load_blob("config", "quotas") for _ in range(4)]
            )
            assert observed == ['{"user":1,"admin":2}'] * 4
        finally:
            release.set()
            await write
    assert await manager.load_blob("config", "quotas") == '{"user":7,"admin":2}'


@pytest.mark.asyncio
async def test_failed_publication_preserves_original_dataset(
    manager, phoenix_container
):
    from phoenix.client import AsyncClient

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    await manager.save_blob("config", "quotas", "committed")
    client = AsyncClient(base_url=phoenix_container["http_endpoint"])
    before = {
        row["id"]
        for row in await client.datasets.list()
        if manager._tenant_id in row["name"]
    }
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        proxy.intercept = lambda method, path, body: (
            (503, {"error": "publication refused"})
            if _is_publication(method, path)
            else None
        )
        writer = _manager_at(proxy.url, manager._tenant_id)
        with pytest.raises(Exception, match="503"):
            await writer.save_blob("config", "quotas", "uncommitted")
    assert await manager.load_blob("config", "quotas") == "committed"
    after = {
        row["id"]
        for row in await client.datasets.list()
        if manager._tenant_id in row["name"]
    }
    assert after == before


@pytest.mark.asyncio
async def test_serving_revisions_are_bounded_and_separate_from_candidates(
    manager, phoenix_container
):
    from phoenix.client import AsyncClient

    for value in range(5):
        await manager.save_blob("model", "selector", f"served-{value}")
    await manager.save_blob_versioned(
        "model",
        "selector",
        "rejected-candidate",
        consumed_example_ids=["span:example"],
        decision="reject",
        scored=False,
        base_score=None,
        candidate_score=None,
    )
    assert await manager.load_blob("model", "selector") == "served-4"
    candidates = await manager.get_version_lineage("model", "selector")
    assert [(entry["version"], entry["decision"]) for entry in candidates] == [
        (1, "reject")
    ]
    client = AsyncClient(base_url=phoenix_container["http_endpoint"])
    rows = [
        row for row in await client.datasets.list() if manager._tenant_id in row["name"]
    ]
    assert len(rows) == 3
    assert sorted(row["example_count"] for row in rows) == [1, 1, 1]


def _publish_blob_process(endpoint, tenant):
    import asyncio

    asyncio.run(
        _manager_at(endpoint, tenant).save_blob("config", "quotas", "replacement")
    )


@pytest.mark.asyncio
async def test_terminated_publisher_keeps_serving_revisions_readable(
    manager, phoenix_container
):
    import asyncio
    import multiprocessing
    import threading

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    await manager.save_blob("config", "quotas", "committed")
    entered, release = threading.Event(), threading.Event()
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:

        def intercept(method, path, body):
            if _is_publication(method, path):
                entered.set()
                release.wait(30)
                return 503, {"error": "publisher terminated"}
            return None

        proxy.intercept = intercept
        worker = multiprocessing.get_context("spawn").Process(
            target=_publish_blob_process, args=(proxy.url, manager._tenant_id)
        )
        worker.start()
        try:
            assert await asyncio.to_thread(entered.wait, 20) is True
            worker.terminate()
            await asyncio.to_thread(worker.join, 10)
            assert worker.exitcode == -15
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join(10)
            release.set()
    assert await manager.load_blob("config", "quotas") == "committed"
    await manager.save_blob("config", "quotas", "retried")
    assert await manager.load_blob("config", "quotas") == "retried"
