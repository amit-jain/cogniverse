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

    @pytest.mark.asyncio
    async def test_clear_by_category_preserves_other_categories(self, memories_client):
        """Clearing one category must not wipe the whole namespace. The client
        used to send ``agent_name`` (which the route drops), so any argument
        fell through to the clear-everything path."""
        rc, mm = memories_client
        await rc.clear_memories(tenant_id=TENANT_ID)

        mem_id = mm.add_memory(
            content="User prefers concise answers.",
            tenant_id=TENANT_ID,
            agent_name=USER_MEMORY_AGENT,
            metadata={"category": "alpha"},
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
            assert any(m["id"] == mem_id for m in listed.get("memories", [])), listed

            # Clear a DIFFERENT category: the 'alpha' memory must survive.
            cleared = await rc.clear_memories(tenant_id=TENANT_ID, category="beta")
            assert cleared.get("status") == "cleared", cleared
            assert cleared.get("deleted") == 0, cleared

            still = await rc.list_memories(tenant_id=TENANT_ID)
            assert any(m["id"] == mem_id for m in still.get("memories", [])), (
                "clearing category 'beta' wrongly deleted the 'alpha' memory"
            )

            # Clearing with no category removes it.
            await rc.clear_memories(tenant_id=TENANT_ID)
            gone = await _retry(
                lambda: rc.list_memories(tenant_id=TENANT_ID),
                lambda r: (
                    r.get("status") != "error"
                    and all(m["id"] != mem_id for m in r.get("memories", []))
                ),
            )
            assert all(m["id"] != mem_id for m in gone.get("memories", []))
        finally:
            try:
                mm.delete_memory(
                    mem_id, tenant_id=TENANT_ID, agent_name=USER_MEMORY_AGENT
                )
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_agent_name_scopes_listing_to_that_agent(self, memories_client):
        """agent_name scopes the listing to that agent's own mem0 store, not the
        default user/strategy namespaces. A memory saved under a named agent is
        visible under agent_name=<agent> but absent from the default listing;
        a user memory is the reverse. If the route ignored agent_name (falling
        through to the default namespaces) the agent memory would be missing
        from its own scoped listing."""
        rc, mm = memories_client
        agent = "video_search_agent"
        agent_marker = "Agent learned: prefer basketball clips."
        user_marker = "User prefers concise answers."

        await rc.clear_memories(tenant_id=TENANT_ID)

        agent_id = mm.add_memory(
            content=agent_marker, tenant_id=TENANT_ID, agent_name=agent, infer=False
        )
        user_id = mm.add_memory(
            content=user_marker,
            tenant_id=TENANT_ID,
            agent_name=USER_MEMORY_AGENT,
            infer=False,
        )
        assert agent_id and user_id, "seed add_memory returned no id"

        try:
            scoped = await _retry(
                lambda: rc.list_memories(tenant_id=TENANT_ID, agent_name=agent),
                lambda r: (
                    r.get("status") != "error"
                    and any(m["id"] == agent_id for m in r.get("memories", []))
                ),
            )
            scoped_ids = [m["id"] for m in scoped.get("memories", [])]
            assert agent_id in scoped_ids, (
                f"agent memory absent under agent_name={agent}: {scoped}"
            )
            assert user_id not in scoped_ids, (
                f"user memory leaked into agent_name={agent} listing: {scoped}"
            )

            default = await _retry(
                lambda: rc.list_memories(tenant_id=TENANT_ID),
                lambda r: (
                    r.get("status") != "error"
                    and any(m["id"] == user_id for m in r.get("memories", []))
                ),
            )
            default_ids = [m["id"] for m in default.get("memories", [])]
            assert user_id in default_ids, (
                f"user memory absent from the default listing: {default}"
            )
            assert agent_id not in default_ids, (
                f"agent-scoped memory leaked into the default listing: {default}"
            )
        finally:
            for mid, ns in ((agent_id, agent), (user_id, USER_MEMORY_AGENT)):
                try:
                    mm.delete_memory(mid, tenant_id=TENANT_ID, agent_name=ns)
                except Exception:
                    pass
