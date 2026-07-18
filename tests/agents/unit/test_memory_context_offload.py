"""Memory/instruction context injection runs off the event loop and reuses the
config manager.

inject_context_into_prompt does 3-6 blocking Vespa/mem0 round-trips (tenant
instructions + relevant context + strategies). Called inline from an async
_process_impl it stalled the whole event loop; and _get_tenant_instructions
built a FRESH VespaConfigStore + TCP session on every dispatch. These pin the
to_thread offload and the manager reuse.
"""

import asyncio
import time

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Mixin(MemoryAwareMixin):
    pass


async def _ticks_during(coro_factory) -> int:
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await coro_factory()
    stop.set()
    await t
    return ticks


@pytest.mark.asyncio
async def test_inject_context_async_offloads_blocking_work(monkeypatch):
    obj = _Mixin()

    def _blocking_inject(prompt, query):
        time.sleep(0.3)  # stand-in for the sync Vespa/mem0 round-trips
        return f"enriched:{prompt}"

    monkeypatch.setattr(obj, "inject_context_into_prompt", _blocking_inject)

    result_holder = {}

    async def _call():
        result_holder["v"] = await obj.inject_context_into_prompt_async("p", "q")

    ticks = await _ticks_during(_call)

    assert result_holder["v"] == "enriched:p"
    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s context injection — it ran on the loop"
    )


def _real_config_manager_with_instructions(text):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_sdk.interfaces.config_store import ConfigScope
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    cm = ConfigManager(store=store)
    cm.set_config_value(
        tenant_id="acme:acme",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": text, "updated_at": "2026-07-17T00:00:00+00:00"},
    )
    return cm, store


def test_get_tenant_instructions_reuses_injected_config_manager(monkeypatch):
    """_get_tenant_instructions must reuse the injected/singleton config manager,
    not build a fresh one per call."""
    import cogniverse_foundation.config.utils as cfg_utils

    # Fail loudly if the mixin builds a fresh manager instead of reusing one.
    def _boom():
        raise AssertionError("built a fresh config manager instead of reusing")

    monkeypatch.setattr(cfg_utils, "create_default_config_manager", _boom)

    cm, _ = _real_config_manager_with_instructions("be concise")
    obj = _Mixin()
    obj._memory_tenant_id = "acme:acme"
    obj._config_manager = cm

    assert obj._get_tenant_instructions() == "be concise"


def test_get_tenant_instructions_served_from_manager_ttl_cache():
    """Repeated enriching dispatches within the TTL must cost ONE store read.

    The mixin previously called cm.store.get_config directly on every
    dispatch, bypassing the manager's scoped TTL cache that sibling config
    scopes (routing, telemetry, agent) already route through."""
    cm, store = _real_config_manager_with_instructions("be concise")
    calls = {"get": 0}
    real_get = store.get_config

    def counting_get(*args, **kwargs):
        calls["get"] += 1
        return real_get(*args, **kwargs)

    store.get_config = counting_get

    obj = _Mixin()
    obj._memory_tenant_id = "acme:acme"
    obj._config_manager = cm

    assert [obj._get_tenant_instructions() for _ in range(3)] == ["be concise"] * 3
    assert calls["get"] == 1, "repeat reads within the TTL must hit the cache"
