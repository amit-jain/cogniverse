"""Memory deletion, namespace guards, counts and pin-aware retention against
Mem0 and Vespa."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager, affirm_memory_profile
from cogniverse_core.memory.pinning import PIN_AGENT_NAME, PinService
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.main import build_pin_lookup
from cogniverse_runtime.optimization_cli import _run_failed, run_cleanup
from cogniverse_runtime.routers import admin, tenant
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    load_raw_schema_json,
    make_config_manager,
)

pytestmark = pytest.mark.no_shared_vespa


@pytest.fixture(scope="module")
def memory_store(shared_vespa):
    """Bulk-seed stored vectors; list/delete never invoke embedding or generation."""
    schema = load_raw_schema_json("agent_memories")
    assert (
        next(
            field["type"]
            for field in schema["document"]["fields"]
            if field["name"] == "embedding"
        )
        == "tensor<float>(d0[768])"
    )
    with InterceptFaultProxy(shared_vespa["base_url"]) as proxy:
        endpoints = dict(shared_vespa, http_port=proxy.port)
        cm = make_config_manager(endpoints)
        affirm_memory_profile(cm)
        managers = []
        for tid in ("stateclear:a", "stateclear:b"):
            deploy_tenant_schema(
                endpoints,
                tenant_id=tid,
                base_schema_name="agent_memories",
                config_manager=cm,
            )
            mm = Mem0MemoryManager(tid)
            mm.initialize(
                backend_host="http://127.0.0.1",
                backend_port=proxy.port,
                backend_config_port=shared_vespa["config_port"],
                base_schema_name="agent_memories",
                llm_model="storage-test-unused",
                embedding_model="lightonai/DenseOn",
                llm_base_url="http://127.0.0.1:9",
                embedder_base_url="http://127.0.0.1:9",
                auto_create_schema=False,
                config_manager=cm,
                schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            )
            managers.append(mm)
        yield managers, proxy, cm
        for mm in managers:
            Mem0MemoryManager._instances.pop(mm.tenant_id, None)


def _seed(mm, namespace, count, *, prefix="row", archived=False):
    ids = [f"{mm.tenant_id}-{prefix}-{i:03}" for i in range(count)]
    payloads = [
        {
            "data": f"stored content {mid}",
            "user_id": mm.tenant_id,
            "agent_id": namespace,
            "created_at": 1700000000 + i,
            "archived": archived or i % 3 == 0,
        }
        for i, mid in enumerate(ids)
    ]
    mm.memory.vector_store.insert([[0.01] * 768] * count, payloads, ids)
    assert _ids(mm, namespace) == set(ids)
    return set(ids)


def _ids(mm, namespace):
    return {
        row["id"]
        for row in mm.get_all_memories(
            mm.tenant_id, namespace, include_archived=True, limit=None
        )
    }


@pytest.fixture
def memory_app(memory_store, monkeypatch):
    managers, proxy, cm = memory_store
    monkeypatch.setattr(tenant, "_config_manager", cm)
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tenant.router)
    yield app
    proxy.intercept = None
    for mm in managers:
        rows = mm.memory.get_all(user_id=mm.tenant_id, limit=None)["results"]
        for row in rows:
            mm.memory.delete(row["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tenant", "admin", "mixin", "manager"])
async def test_clear_all_removes_205_active_and_archived_rows(
    memory_store, memory_app, entry
):
    managers, _, _ = memory_store
    target, peer = managers
    _seed(target, "_user_memories", 205)
    same_tenant = _seed(target, "untouched", 3, prefix="other")
    other_tenant = _seed(peer, "_user_memories", 4, prefix="peer")
    if entry in {"tenant", "admin"}:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
        ) as client:
            route = (
                f"/{target.tenant_id}/memories"
                if entry == "tenant"
                else f"/admin/memories/{target.tenant_id}?type=preference"
            )
            response = await client.delete(route)
        assert response.status_code == 200
        assert response.json() == (
            {"status": "cleared", "agent_name": "_user_memories"}
            if entry == "tenant"
            else {"status": "cleared", "type": "preference"}
        )
    elif entry == "mixin":
        mixin = MemoryAwareMixin()
        mixin.memory_manager = target
        mixin._memory_initialized = True
        mixin._memory_agent_name = "_user_memories"
        mixin.set_tenant_for_context(target.tenant_id)
        assert await asyncio.to_thread(mixin.clear_memory) is True
    else:
        # A caller holding a manager calls this entrypoint directly.
        assert (
            await asyncio.to_thread(
                target.clear_agent_memory, target.tenant_id, "_user_memories"
            )
            is True
        )
    assert _ids(target, "_user_memories") == set()
    assert _ids(target, "untouched") == same_tenant
    assert _ids(peer, "_user_memories") == other_tenant


def _seed_categorised(mm, namespace, categories):
    """Seed one row per (category, index), a third of them archived."""
    ids, payloads = [], []
    for category, count in categories.items():
        for i in range(count):
            mid = f"{mm.tenant_id}-{category}-{i:03}"
            ids.append(mid)
            payloads.append(
                {
                    "data": f"stored content {mid}",
                    "user_id": mm.tenant_id,
                    "agent_id": namespace,
                    "created_at": 1700000000 + i,
                    "archived": i % 3 == 0,
                    "category": category,
                }
            )
    mm.memory.vector_store.insert([[0.01] * 768] * len(ids), payloads, ids)
    assert _ids(mm, namespace) == set(ids)
    return {
        category: {mid for mid in ids if mid.rsplit("-", 2)[1] == category}
        for category in categories
    }


@pytest.mark.asyncio
async def test_clear_by_category_removes_archived_rows_of_that_category(
    memory_store, memory_app
):
    """A category clear means the same thing the whole-namespace clear means.

    Archived rows are still that tenant's rows; leaving them behind under
    ``{"status": "cleared"}`` is the whole-namespace defect at category scope.
    """
    managers, _, _ = memory_store
    target, _peer = managers
    seeded = _seed_categorised(target, "_user_memories", {"preference": 120, "fact": 5})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        response = await client.delete(
            f"/{target.tenant_id}/memories", params={"category": "preference"}
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "agent_name": "_user_memories",
        "category": "preference",
        "deleted": 120,
    }
    assert _ids(target, "_user_memories") == seeded["fact"]


@pytest.mark.asyncio
async def test_independent_tenant_clears_interleave_without_cross_deletion(
    memory_store, memory_app
):
    managers, proxy, _ = memory_store
    for mm in managers:
        _seed(mm, "_user_memories", 205)
        _seed(mm, "untouched", 2, prefix="keep")
    barrier = threading.Barrier(2, timeout=30)
    seen = set()
    lock = threading.Lock()

    def intercept(method, path, body):
        if method == "DELETE":
            key = next(mm.tenant_id for mm in managers if mm.tenant_id in unquote(path))
            with lock:
                first = key not in seen
                seen.add(key)
            if first:
                barrier.wait()

    proxy.intercept = intercept
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        responses = await asyncio.gather(
            *(client.delete(f"/{mm.tenant_id}/memories") for mm in managers)
        )
    assert [response.json() for response in responses] == [
        {"status": "cleared", "agent_name": "_user_memories"},
        {"status": "cleared", "agent_name": "_user_memories"},
    ]
    assert seen == {mm.tenant_id for mm in managers}
    proxy.intercept = None
    for mm in managers:
        assert _ids(mm, "_user_memories") == set()
        assert _ids(mm, "untouched") == {
            f"{mm.tenant_id}-keep-{i:03}" for i in range(2)
        }


@pytest.mark.asyncio
async def test_clear_refused_midway_fails_and_retry_removes_every_row(
    memory_store, memory_app
):
    managers, proxy, _ = memory_store
    mm = managers[0]
    seeded = _seed(mm, "_user_memories", 205)
    deleted = []

    def intercept(method, path, body):
        if method == "DELETE":
            if len(deleted) == 10:
                return 409, {"message": "injected memory deletion refusal"}
            deleted.append(unquote(path.rsplit("/", 1)[-1]))

    proxy.intercept = intercept
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app, raise_app_exceptions=False),
        base_url="http://runtime",
    ) as client:
        response = await client.delete(f"/{mm.tenant_id}/memories")
        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert len(deleted) == 10
        assert _ids(mm, "_user_memories") == seeded - set(deleted)
        proxy.intercept = None
        retried = await client.delete(f"/{mm.tenant_id}/memories")
    assert retried.json() == {"status": "cleared", "agent_name": "_user_memories"}
    assert _ids(mm, "_user_memories") == set()


def _seed_rows(mm, rows):
    """Seed ``(id, namespace)`` rows, none archived."""
    ids = [mid for mid, _ in rows]
    payloads = [
        {
            "data": f"stored content {mid}",
            "user_id": mm.tenant_id,
            "agent_id": namespace,
            "created_at": 1700000000,
            "archived": False,
        }
        for mid, namespace in rows
    ]
    mm.memory.vector_store.insert([[0.01] * 768] * len(ids), payloads, ids)


@pytest.mark.asyncio
async def test_delete_removes_only_a_memory_of_the_named_namespace(
    memory_store, memory_app
):
    managers, _, _ = memory_store
    target, peer = managers
    rows = {
        "user": f"{target.tenant_id}-user",
        "agent": f"{target.tenant_id}-agent",
        "strategy": f"{target.tenant_id}-strategy",
    }
    _seed_rows(
        target,
        [
            (rows["user"], "_user_memories"),
            (rows["agent"], "search_agent"),
            (rows["strategy"], "_strategy_store"),
        ],
    )
    peer_row = f"{peer.tenant_id}-user"
    _seed_rows(peer, [(peer_row, "_user_memories")])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        # The user namespace is the default; a strategy's ID is not in it.
        wrong_namespace = await client.delete(
            f"/{target.tenant_id}/memories/{rows['strategy']}"
        )
        system = await client.delete(
            f"/{target.tenant_id}/memories/{rows['strategy']}",
            params={"agent_name": "_strategy_store"},
        )
        other_tenant = await client.delete(f"/{target.tenant_id}/memories/{peer_row}")
        agent = await client.delete(
            f"/{target.tenant_id}/memories/{rows['agent']}",
            params={"agent_name": "search_agent"},
        )

    assert (wrong_namespace.status_code, wrong_namespace.json()) == (
        404,
        {
            "detail": f"Memory {rows['strategy']} not found among the memories "
            "of _user_memories"
        },
    )
    assert (system.status_code, system.json()) == (
        403,
        {
            "detail": "_strategy_store is a system memory namespace; "
            "the runtime manages it."
        },
    )
    assert (other_tenant.status_code, other_tenant.json()) == (
        404,
        {"detail": f"Memory {peer_row} not found among the memories of _user_memories"},
    )
    assert (agent.status_code, agent.json()) == (200, {"status": "deleted"})
    assert _ids(target, "_user_memories") == {rows["user"]}
    assert _ids(target, "search_agent") == set()
    assert _ids(target, "_strategy_store") == {rows["strategy"]}
    assert _ids(peer, "_user_memories") == {peer_row}


@pytest.mark.asyncio
async def test_clear_names_its_namespace_and_refuses_a_system_one(
    memory_store, memory_app
):
    managers, _, _ = memory_store
    target, _peer = managers
    users = _seed(target, "_user_memories", 3, prefix="user")
    _seed(target, "search_agent", 120, prefix="agent")
    strategies = _seed(target, "_strategy_store", 2, prefix="strategy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        system = await client.delete(
            f"/{target.tenant_id}/memories", params={"agent_name": "_strategy_store"}
        )
        cleared = await client.delete(
            f"/{target.tenant_id}/memories", params={"agent_name": "search_agent"}
        )

    assert (system.status_code, system.json()) == (
        403,
        {
            "detail": "_strategy_store is a system memory namespace; "
            "the runtime manages it."
        },
    )
    assert (cleared.status_code, cleared.json()) == (
        200,
        {"status": "cleared", "agent_name": "search_agent"},
    )
    assert _ids(target, "search_agent") == set()
    assert _ids(target, "_user_memories") == users
    assert _ids(target, "_strategy_store") == strategies


@pytest.mark.asyncio
async def test_stats_count_live_and_archived_rows_past_the_page(
    memory_store, memory_app
):
    managers, _, _ = memory_store
    target, peer = managers
    # _seed archives every third row: 69 of 205.
    _seed(target, "_user_memories", 205)
    _seed(target, "_strategy_store", 4, prefix="strategy")
    _seed(peer, "_user_memories", 7, prefix="peer")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        users = await client.get(f"/{target.tenant_id}/memories/stats")
        strategies = await client.get(
            f"/{target.tenant_id}/memories/stats",
            params={"agent_name": "_strategy_store"},
        )
        empty = await client.get(
            f"/{target.tenant_id}/memories/stats",
            params={"agent_name": "search_agent"},
        )

    assert users.json() == {
        "agent_name": "_user_memories",
        "user_id": target.tenant_id,
        "total": 136,
        "archived": 69,
        "writable": True,
    }
    assert strategies.json() == {
        "agent_name": "_strategy_store",
        "user_id": target.tenant_id,
        "total": 2,
        "archived": 2,
        "writable": False,
    }
    assert empty.json() == {
        "agent_name": "search_agent",
        "user_id": target.tenant_id,
        "total": 0,
        "archived": 0,
        "writable": True,
    }


@pytest.mark.asyncio
async def test_stats_during_a_store_outage_fail_rather_than_count_zero(
    memory_store, memory_app
):
    managers, proxy, _ = memory_store
    target, _peer = managers
    _seed(target, "_user_memories", 5)
    refused = []

    def intercept(method, path, body):
        if method == "POST" and path.startswith("/search/"):
            refused.append(path)
            return 503, {"message": "injected search outage"}

    proxy.intercept = intercept
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        response = await client.get(f"/{target.tenant_id}/memories/stats")
    proxy.intercept = None

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert {key: detail[key] for key in ("error", "message", "tenant_id")} == {
        "error": "memory_unavailable",
        "message": f"Could not read the memories of _user_memories for tenant "
        f"{target.tenant_id}.",
        "tenant_id": target.tenant_id,
    }
    assert len(refused) == 1


@pytest.mark.asyncio
async def test_two_deletes_racing_past_the_membership_check_leave_it_deleted(
    memory_store, memory_app
):
    managers, proxy, _ = memory_store
    target, _peer = managers
    mid = f"{target.tenant_id}-raced"
    kept = f"{target.tenant_id}-kept"
    _seed_rows(target, [(mid, "search_agent"), (kept, "search_agent")])
    barrier = threading.Barrier(2, timeout=30)
    reads = []
    lock = threading.Lock()

    def intercept(method, path, body):
        # The first two reads of the row are the two membership checks; each
        # completes only once both have arrived, so both requests pass the
        # check before either deletes.
        if method == "GET" and "/document/v1/" in path and mid in unquote(path):
            with lock:
                reads.append(path)
                checking = len(reads) <= 2
            if checking:
                barrier.wait()

    proxy.intercept = intercept
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.delete(
                    f"/{target.tenant_id}/memories/{mid}",
                    params={"agent_name": "search_agent"},
                )
                for _ in range(2)
            )
        )
    proxy.intercept = None

    deleted = (200, {"status": "deleted"})
    gone = (
        404,
        {"detail": f"Memory {mid} not found among the memories of search_agent"},
    )
    outcomes = sorted(
        ((response.status_code, response.json()) for response in responses),
        key=lambda outcome: outcome[0],
    )
    # Whichever reaches the store second finds the row deleted or deletes
    # nothing; neither fails.
    assert outcomes in ([deleted, deleted], [deleted, gone])
    assert _ids(target, "search_agent") == {kept}


@pytest.fixture
def retention_run(memory_store, memory_app, monkeypatch, tmp_path):
    managers, proxy, cm = memory_store
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1")
    monkeypatch.setenv("BACKEND_PORT", str(proxy.port))
    monkeypatch.setattr(tenant_manager, "_config_manager", cm)
    monkeypatch.setattr(tenant_manager, "_backend", managers[0]._resolve_backend())
    monkeypatch.setattr(
        tenant_manager,
        "_schema_loader",
        FilesystemSchemaLoader(Path("configs/schemas")),
    )
    log_dir, temp_dir = tmp_path / "logs", tmp_path / "scratch"
    log_dir.mkdir()
    temp_dir.mkdir()

    async def run():
        return await run_cleanup(
            tenant_id=managers[0].tenant_id,
            log_retention_days=7,
            memory_retention_days=30,
            log_dir=log_dir,
            temp_dir=temp_dir,
            temp_retention_days=1,
            schemas_dir="configs/schemas",
            config_keep_versions=10,
        )

    return run


def _seed_retention(mm, *, unpinned=True):
    now = int(time.time())
    records = {}
    for i in range(105):
        mid = f"pinned-{i:03}"
        kind = "learned_strategy" if i % 3 == 0 else "conversation_turn"
        records[mid] = {
            "data": f"preserve pinned content {i}",
            "agent_id": "retention",
            "kind": kind,
            "created_at": now - (40 if i % 2 else 20) * 86400 + i,
            "archived": i == 104,
        }
        records[f"pin-{i:03}"] = {
            "data": f"pin target {mid}",
            "agent_id": PIN_AGENT_NAME,
            "kind": "pin_record",
            "created_at": now - 80 * 86400 + i,
            "target_memory_id": mid,
            "target_kind": kind,
            "pinned_by": "org_admin",
            "pin_actor_id": "operator",
        }
    if unpinned:
        for name, days in (("soft", 20), ("hard", 40), ("fresh", 0)):
            records[name] = {
                "data": f"unpinned {name}",
                "agent_id": "retention",
                "kind": "conversation_turn",
                "created_at": now - days * 86400,
            }
    mm.memory.vector_store.insert(
        [[0.01] * 768] * len(records),
        [{"user_id": mm.tenant_id, **row} for row in records.values()],
        list(records),
    )
    return {
        row["id"]: row
        for row in mm.memory.get_all(user_id=mm.tenant_id, limit=None)["results"]
    }


@pytest.mark.asyncio
async def test_cleanup_preserves_every_pin_and_exhausts_expired_rows(
    memory_store, retention_run
):
    managers, _, _ = memory_store
    mm = managers[0]
    before = _seed_retention(mm)
    assert len(before) == 213
    result = await retention_run()
    assert result["memory_cleanup"] == {
        mm.tenant_id: {
            "status": "completed",
            "deleted_by_kind": {
                "conversation_turn": 1,
                "conversation_turn:archived": 1,
            },
        }
    }
    assert result["memory_cleanup_summary"] == {
        "completed": 1,
        "skipped": 0,
        "failed": 0,
    }
    after = {
        row["id"]: row
        for row in mm.memory.get_all(user_id=mm.tenant_id, limit=None)["results"]
    }
    assert set(after) == set(before) - {"hard"}
    assert {key: row for key, row in after.items() if key != "soft"} == {
        key: row for key, row in before.items() if key not in {"hard", "soft"}
    }
    assert after["soft"]["memory"] == before["soft"]["memory"]
    assert after["soft"]["metadata"]["archived"] is True
    assert {
        rec.target_memory_id
        for rec in PinService(mm, build_default_registry()).list_pins(mm.tenant_id)
    } == {f"pinned-{i:03}" for i in range(105)}


@pytest.mark.asyncio
async def test_runtime_and_cron_cleanup_interleave_preserving_pinned_rows(
    memory_store, retention_run
):
    managers, proxy, _ = memory_store
    mm = managers[0]
    before = _seed_retention(mm, unpinned=False)
    barrier = threading.Barrier(2, timeout=30)
    entered = 0
    lock = threading.Lock()

    def intercept(method, path, body):
        nonlocal entered
        query = (
            json.loads(body).get("yql", "")
            if body and method == "POST" and path.startswith("/search/")
            else ""
        )
        if 'user_id contains "stateclear:a"' in query and "agent_id" not in query:
            with lock:
                entered += 1
                first_page = entered <= 2
            if first_page:
                barrier.wait()

    registry = build_default_registry()
    scheduler = LifecycleScheduler(
        get_warm_managers=lambda: [mm],
        registry=registry,
        pin_lookup=build_pin_lookup(registry, lambda tid: {}),
    )
    proxy.intercept = intercept
    cron, runtime = await asyncio.gather(
        asyncio.to_thread(lambda: asyncio.run(retention_run())), scheduler.tick_once()
    )
    proxy.intercept = None
    assert cron["memory_cleanup"] == {
        mm.tenant_id: {"status": "completed", "deleted_by_kind": {}}
    }
    assert runtime == {"tenants": {mm.tenant_id: {}}, "total_deleted": 0}
    assert entered == 6
    after = {
        row["id"]: row
        for row in mm.memory.get_all(user_id=mm.tenant_id, limit=None)["results"]
    }
    assert after == before


@pytest.mark.asyncio
async def test_cleanup_pin_read_failure_mutates_nothing_and_fails_explicitly(
    memory_store, retention_run
):
    managers, proxy, _ = memory_store
    mm = managers[0]
    before = _seed_retention(mm)
    failures = 0

    def intercept(method, path, body):
        nonlocal failures
        if method == "POST" and path.startswith("/search/") and b"_pinning" in body:
            failures += 1
            return 409, {"message": "injected pin read refusal"}

    proxy.intercept = intercept
    result = await retention_run()
    proxy.intercept = None
    assert _run_failed(result) is True
    assert result["memory_cleanup_summary"] == {
        "completed": 0,
        "skipped": 0,
        "failed": 1,
    }
    assert result["memory_cleanup"][mm.tenant_id]["status"] == "failed"
    assert (
        "injected pin read refusal" in result["memory_cleanup"][mm.tenant_id]["error"]
    )
    assert failures == 1
    after = {
        row["id"]: row
        for row in mm.memory.get_all(user_id=mm.tenant_id, limit=None)["results"]
    }
    assert after == before


@pytest.mark.asyncio
async def test_health_reads_the_store_and_names_an_outage(memory_store, memory_app):
    managers, proxy, _ = memory_store
    target, peer = managers
    refused = []

    def intercept(method, path, body):
        if (
            method == "POST"
            and path.startswith("/search/")
            and (target.tenant_id.replace(":", "_").encode() in body)
        ):
            refused.append(path)
            return 503, {"message": "injected search outage"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=memory_app), base_url="http://runtime"
    ) as client:
        healthy = await client.get(f"/{target.tenant_id}/memories/health")
        proxy.intercept = intercept
        answers = await asyncio.gather(
            *(
                client.get(f"/{manager.tenant_id}/memories/health")
                for manager in (target, peer, target, peer)
            )
        )
    proxy.intercept = None

    assert (healthy.status_code, healthy.json()) == (
        200,
        {
            "tenant_id": target.tenant_id,
            "agent_name": "_user_memories",
            "healthy": True,
            "problem": None,
        },
    )
    assert [
        (a.status_code, a.json()["tenant_id"], a.json()["healthy"]) for a in answers
    ] == [
        (200, target.tenant_id, False),
        (200, peer.tenant_id, True),
        (200, target.tenant_id, False),
        (200, peer.tenant_id, True),
    ]
    assert answers[0].json()["problem"] == (
        "The memory store did not answer a read (VespaError)."
    )
    assert len(refused) == 2
