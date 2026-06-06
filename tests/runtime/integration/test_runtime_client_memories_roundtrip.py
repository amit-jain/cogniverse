"""Real-boundary round-trip for RuntimeClient memories CRUD.

The unit suite in tests/messaging/unit/test_runtime_client_crud.py mocks
RuntimeClient's httpx transport and asserts only the URL/params the client
builds. Here RuntimeClient drives the real tenant router over
httpx.ASGITransport against a real Mem0 backend (Vespa + denseon), so
list/clear must resolve to the real handlers and round-trip a persisted
user memory.
"""

import asyncio

import httpx
import pytest
from cogniverse_messaging.runtime_client import RuntimeClient
from fastapi import FastAPI

from cogniverse_runtime.routers import tenant as tenant_router

TENANT_ID = "test:unit"
USER_MEMORY_AGENT = "_user_memories"


@pytest.fixture
async def memories_client(memory_manager, config_manager):
    """RuntimeClient → ASGITransport → real tenant router → real Mem0.

    Depends on ``memory_manager`` so the per-tenant Mem0 singleton for
    ``test:unit`` is already initialised; the route's _get_memory_manager
    reuses it (lazy-init short-circuits).
    """
    original_cm = tenant_router._config_manager
    tenant_router.set_config_manager(config_manager)

    app = FastAPI()
    app.include_router(tenant_router.router, prefix="/admin/tenant")

    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://runtime",
        timeout=60.0,
    )
    try:
        yield rc, memory_manager
    finally:
        await rc._client.aclose()
        tenant_router._config_manager = original_cm


async def _retry(coro_factory, predicate, attempts: int = 10, delay: float = 1.0):
    result = None
    for _ in range(attempts):
        result = await coro_factory()
        if predicate(result):
            return result
        await asyncio.sleep(delay)
    return result


@pytest.mark.integration
class TestRuntimeClientMemoriesRoundTrip:
    @pytest.mark.asyncio
    async def test_seed_list_clear_memory(self, memories_client):
        rc, mm = memories_client
        marker = "User prefers concise bullet-point answers."

        # Clear first so _user_memories holds exactly the seeded entry,
        # independent of any state left by earlier cases in the module.
        await rc.clear_memories(tenant_id=TENANT_ID)

        mem_id = mm.add_memory(
            content=marker,
            tenant_id=TENANT_ID,
            agent_name=USER_MEMORY_AGENT,
            infer=False,
        )
        assert mem_id, "seed add_memory returned no id"

        try:
            listed = await _retry(
                lambda: rc.list_memories(tenant_id=TENANT_ID),
                lambda r: (
                    r.get("status") != "error"
                    and any(m["id"] == mem_id for m in r.get("memories", []))
                ),
            )
            assert listed.get("status") != "error", listed
            match = next((m for m in listed["memories"] if m["id"] == mem_id), None)
            assert match is not None, f"seeded memory {mem_id} not in {listed}"
            assert marker in match["memory"]
            assert match["type"] == "preference"
            assert match["owned"] is True

            cleared = await rc.clear_memories(tenant_id=TENANT_ID)
            assert cleared.get("status") == "cleared", cleared

            after = await _retry(
                lambda: rc.list_memories(tenant_id=TENANT_ID),
                lambda r: (
                    r.get("status") != "error"
                    and all(m["id"] != mem_id for m in r.get("memories", []))
                ),
            )
            ids_after = [m["id"] for m in after.get("memories", [])]
            assert mem_id not in ids_after, f"{mem_id} still present after clear"
        finally:
            mm.delete_memory(mem_id, tenant_id=TENANT_ID, agent_name=USER_MEMORY_AGENT)
