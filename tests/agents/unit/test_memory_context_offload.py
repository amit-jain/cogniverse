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


def test_get_tenant_instructions_reuses_injected_config_manager(monkeypatch):
    """_get_tenant_instructions must reuse the injected/singleton config manager,
    not build a fresh one per call."""
    import cogniverse_foundation.config.utils as cfg_utils

    # Fail loudly if the mixin builds a fresh manager instead of reusing one.
    def _boom():
        raise AssertionError("built a fresh config manager instead of reusing")

    monkeypatch.setattr(cfg_utils, "create_default_config_manager", _boom)

    class _Entry:
        config_value = {"text": "be concise"}

    class _Store:
        def get_config(self, **kwargs):
            return _Entry()

    class _CM:
        store = _Store()

    obj = _Mixin()
    obj._memory_tenant_id = "acme:acme"
    obj._config_manager = _CM()

    assert obj._get_tenant_instructions() == "be concise"
