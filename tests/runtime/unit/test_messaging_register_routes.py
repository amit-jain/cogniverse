"""Registration and resolve routes for messaging users.

Drives the REAL admin router over ASGITransport with a real in-memory
ConfigStore and a partition-faithful Mem0 double. The contract under test:
validate → register → consume ordering (a failed registration never burns
the token), 404 for a bad token, 503 with the token intact on any backend
outage, null-only-when-genuinely-unregistered resolve, and single-use
under concurrent registration attempts.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.messaging_auth import InviteTokenManager, UserTenantMapper
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _PartitionedMemory:
    """Partition-faithful Mem0 double.

    Mirrors the real store contract that matters for user-tenant mapping:
    Mem0 caps an unfiltered ``get_all`` at 100 rows, and a promoted-field
    (``session_id``) filter is applied server-side. Without the filter a
    mapping past the 100th inserted row is invisible; with it the exact row
    comes back regardless of how full the partition is.
    """

    _DEFAULT_PAGE = 100

    def __init__(self):
        self.store = {}
        self.fail_writes = False
        self.fail_reads = False
        self.memory = object()

    def add_memory(self, content, tenant_id, agent_name, metadata=None, **kwargs):
        if self.fail_writes:
            raise ConnectionError("mem0 down")
        self.store.setdefault((tenant_id, agent_name), []).append(
            {"memory": content, "metadata": metadata or {}}
        )
        return "mem_1"

    def get_all_memories(
        self, tenant_id, agent_name, filters=None, limit=_DEFAULT_PAGE, **kwargs
    ):
        if self.fail_reads:
            raise ConnectionError("mem0 down")
        rows = list(self.store.get((tenant_id, agent_name), []))
        if filters and filters.get("session_id") is not None:
            wanted = filters["session_id"]
            rows = [
                r for r in rows if (r.get("metadata") or {}).get("session_id") == wanted
            ]
        if limit is not None:
            rows = rows[:limit]
        return rows


class _OutageStore(InMemoryConfigStore):
    def get_config(self, *args, **kwargs):
        raise ConnectionError("config store unreachable")


class _BarrierOnValidateStore(InMemoryConfigStore):
    """Blocks the two DIFFERENT-token validate reads on one 2-party barrier.

    Per-token locks let the two registrations run concurrently, so both
    validate threads reach the barrier and pass. A single process-global lock
    would serialize them: the second registration can't start its validate
    until the first fully finishes, so only one thread reaches the barrier and
    it trips ``BrokenBarrierError`` after the timeout — turning the parallel
    case into a 503. Armed only after minting so mint reads don't trip it.
    """

    def __init__(self, barrier: threading.Barrier):
        super().__init__()
        self._barrier = barrier
        self._seen: set = set()
        self.armed = False

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        if (
            self.armed
            and config_key.startswith("invite_token_")
            and config_key not in self._seen
        ):
            self._seen.add(config_key)
            self._barrier.wait(timeout=5)
        return super().get_config(tenant_id, scope, service, config_key, version)


@pytest.fixture
def harness():
    store = InMemoryConfigStore()
    store.initialize()
    cm = ConfigManager(store=store)
    memory = _PartitionedMemory()
    admin_router.set_system_memory_factory(lambda: memory)

    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: cm
    try:
        yield app, cm, memory
    finally:
        admin_router.set_system_memory_factory(None)


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )


async def _mint(client, tenant="acme:alice"):
    resp = await client.post("/admin/messaging/invite", json={"tenant_id": tenant})
    assert resp.status_code == 200
    return resp.json()["token"]


@pytest.mark.asyncio
async def test_register_then_resolve_round_trip(harness):
    app, cm, memory = harness
    async with _client(app) as client:
        token = await _mint(client)

        resp = await client.post(
            "/admin/messaging/register",
            json={"platform": "telegram", "external_user_id": "42", "token": token},
        )
        assert resp.status_code == 200
        assert resp.json() == {"tenant_id": "acme:alice"}

        resolved = await client.get(
            "/admin/messaging/resolve",
            params={"platform": "telegram", "external_user_id": "42"},
        )
        assert resolved.status_code == 200
        assert resolved.json() == {"tenant_id": "acme:alice"}

        # Token consumed: a second register of the same token is invalid.
        again = await client.post(
            "/admin/messaging/register",
            json={"platform": "telegram", "external_user_id": "43", "token": token},
        )
        assert again.status_code == 404


@pytest.mark.asyncio
async def test_invalid_token_is_404_and_writes_nothing(harness):
    app, cm, memory = harness
    async with _client(app) as client:
        resp = await client.post(
            "/admin/messaging/register",
            json={
                "platform": "telegram",
                "external_user_id": "42",
                "token": "bogus",
            },
        )
        assert resp.status_code == 404
        assert memory.store == {}


@pytest.mark.asyncio
async def test_mem0_outage_is_503_and_token_survives(harness):
    app, cm, memory = harness
    async with _client(app) as client:
        token = await _mint(client)
        memory.fail_writes = True

        resp = await client.post(
            "/admin/messaging/register",
            json={"platform": "telegram", "external_user_id": "42", "token": token},
        )
        assert resp.status_code == 503

        # Token intact: the same token registers once Mem0 recovers.
        memory.fail_writes = False
        retry = await client.post(
            "/admin/messaging/register",
            json={"platform": "telegram", "external_user_id": "42", "token": token},
        )
        assert retry.status_code == 200
        assert retry.json()["tenant_id"] == "acme:alice"


@pytest.mark.asyncio
async def test_config_outage_is_503(harness):
    app, cm, memory = harness
    outage_store = _OutageStore()
    outage_store.initialize()
    outage_cm = ConfigManager(store=outage_store)
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
        outage_cm
    )
    async with _client(app) as client:
        resp = await client.post(
            "/admin/messaging/register",
            json={"platform": "telegram", "external_user_id": "42", "token": "t"},
        )
        assert resp.status_code == 503
        assert memory.store == {}


@pytest.mark.asyncio
async def test_resolve_unregistered_is_null_but_outage_is_503(harness):
    app, cm, memory = harness
    async with _client(app) as client:
        resp = await client.get(
            "/admin/messaging/resolve",
            params={"platform": "telegram", "external_user_id": "99"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"tenant_id": None}

        memory.fail_reads = True
        outage = await client.get(
            "/admin/messaging/resolve",
            params={"platform": "telegram", "external_user_id": "99"},
        )
        assert outage.status_code == 503


@pytest.mark.asyncio
async def test_concurrent_registers_consume_the_token_once(harness):
    """Two racers, one token: exactly one 200, the loser gets 404, and
    exactly one mapping is stored."""
    app, cm, memory = harness
    async with _client(app) as client:
        token = await _mint(client)

        r1, r2 = await asyncio.gather(
            client.post(
                "/admin/messaging/register",
                json={
                    "platform": "telegram",
                    "external_user_id": "42",
                    "token": token,
                },
            ),
            client.post(
                "/admin/messaging/register",
                json={
                    "platform": "telegram",
                    "external_user_id": "43",
                    "token": token,
                },
            ),
        )
        statuses = sorted([r1.status_code, r2.status_code])
        assert statuses == [200, 404]

        mappings = [
            row
            for rows in memory.store.values()
            for row in rows
            if row["metadata"].get("type") == "user_mapping"
        ]
        assert len(mappings) == 1

    # The winner resolves; validate agrees the token is spent.
    assert InviteTokenManager(cm).validate_token(token) is None


@pytest.mark.asyncio
async def test_different_tokens_register_in_parallel():
    """Registrations for DIFFERENT tokens must not convoy behind one global
    lock. Both validate reads meet on a 2-party barrier; the per-token lock
    lets them run concurrently so both pass and return 200. A single global
    lock would serialize them and the barrier would trip (503)."""
    barrier = threading.Barrier(2)
    store = _BarrierOnValidateStore(barrier)
    store.initialize()
    cm = ConfigManager(store=store)
    memory = _PartitionedMemory()
    admin_router.set_system_memory_factory(lambda: memory)
    admin_router._register_locks.clear()

    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: cm
    try:
        async with _client(app) as client:
            token_a = await _mint(client, tenant="acme:alice")
            token_b = await _mint(client, tenant="beta:bob")
            store.armed = True  # only the two concurrent validates hit the barrier

            ra, rb = await asyncio.gather(
                client.post(
                    "/admin/messaging/register",
                    json={
                        "platform": "telegram",
                        "external_user_id": "1",
                        "token": token_a,
                    },
                ),
                client.post(
                    "/admin/messaging/register",
                    json={
                        "platform": "telegram",
                        "external_user_id": "2",
                        "token": token_b,
                    },
                ),
            )
        # Both cleared the barrier concurrently — different tokens ran in
        # parallel, not serialized behind one lock.
        assert ra.status_code == 200
        assert rb.status_code == 200
        assert ra.json()["tenant_id"] == "acme:alice"
        assert rb.json()["tenant_id"] == "beta:bob"
    finally:
        admin_router.set_system_memory_factory(None)
        admin_router._register_locks.clear()


def test_resolve_finds_mapping_beyond_the_hundred_row_page(harness):
    """A registered user past the 100th mapping still resolves.

    Mem0's unfiltered ``get_all`` returns a capped 100-row page, so a
    linear scan misses any mapping past it. ``get_tenant_id`` must narrow
    server-side by the stamped session key, not enumerate-and-scan.
    """
    _, _, memory = harness
    mapper = UserTenantMapper(memory)

    for i in range(1, 102):
        assert mapper.register_user("telegram", str(i), f"acme:t{i}") is True

    # The 101st mapping sits past Mem0's default page; it must still resolve.
    assert mapper.get_tenant_id("telegram", "101") == "acme:t101"
    # And a user well inside the first page resolves to its own tenant.
    assert mapper.get_tenant_id("telegram", "7") == "acme:t7"
    # An unregistered user is genuinely None, not a scan artifact.
    assert mapper.get_tenant_id("telegram", "9999") is None
