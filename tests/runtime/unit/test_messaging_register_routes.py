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

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.messaging_auth import InviteTokenManager
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _PartitionedMemory:
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

    def get_all_memories(self, tenant_id, agent_name):
        if self.fail_reads:
            raise ConnectionError("mem0 down")
        return self.store.get((tenant_id, agent_name), [])


class _OutageStore(InMemoryConfigStore):
    def get_config(self, *args, **kwargs):
        raise ConnectionError("config store unreachable")


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
