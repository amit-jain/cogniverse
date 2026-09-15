"""Complete memory deletion and pin-aware retention against Mem0 and Vespa."""

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
from cogniverse_core.memory.manager import Mem0MemoryManager
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
        endpoints = dict(shared_vespa, http_port=proxy.server.server_port)
        cm = make_config_manager(endpoints)
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
                backend_port=proxy.server.server_port,
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
@pytest.mark.parametrize("entry", ["tenant", "admin", "mixin", "dashboard"])
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
            {"status": "cleared"}
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
        # The dashboard's clear button calls this manager entrypoint directly.
        assert (
            await asyncio.to_thread(
                target.clear_agent_memory, target.tenant_id, "_user_memories"
            )
            is True
        )
    assert _ids(target, "_user_memories") == set()
    assert _ids(target, "untouched") == same_tenant
    assert _ids(peer, "_user_memories") == other_tenant


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
        {"status": "cleared"},
        {"status": "cleared"},
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
    assert retried.json() == {"status": "cleared"}
    assert _ids(mm, "_user_memories") == set()


@pytest.fixture
def retention_run(memory_store, memory_app, monkeypatch, tmp_path):
    managers, proxy, cm = memory_store
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1")
    monkeypatch.setenv("BACKEND_PORT", str(proxy.server.server_port))
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
