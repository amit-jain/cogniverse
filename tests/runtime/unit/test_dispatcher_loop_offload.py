"""A dispatch's blocking reads run off the serving event loop.

Each test holds one dependency of a dispatch until a heartbeat coroutine on the
loop releases it. On the loop, the held call keeps the heartbeat from running
until the hold times out; off the loop, the heartbeat runs and releases it.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:acme"
HOLD_S = 5.0


class _Held:
    """A blocking call released by a heartbeat on the loop."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.released = threading.Event()
        self.waits: list[bool] = []

    def hold(self) -> None:
        self.started.set()
        self.waits.append(self.released.wait(timeout=HOLD_S))

    async def heartbeat(self) -> list[bool]:
        """Whether the loop ran while the call was held."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30
        while not self.started.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        beat = [self.waits == []]
        self.released.set()
        return beat


class _Stop(BaseException):
    """Ends the dispatch once the held call returned; a ``BaseException`` so
    the dispatcher's degrade-and-log handlers do not swallow it."""


def _dispatcher(config_manager) -> AgentDispatcher:
    dispatcher = object.__new__(AgentDispatcher)
    dispatcher._config_manager = config_manager
    dispatcher._sandbox_manager = None
    dispatcher._registry = SimpleNamespace(get_agent=lambda name: None)
    return dispatcher


_EGRESS_SITES = {
    "search_agent": lambda d: d._execute_search_task("q", TENANT, 3),
    "routing_agent": lambda d: d._execute_gateway_task("q", {}, TENANT),
    "orchestrator_agent": lambda d: d._execute_orchestration_task("q", {}, TENANT),
    "summarizer_agent": lambda d: d._execute_summarization_task("q", TENANT),
    "coding_agent": lambda d: d._execute_coding_task("q", TENANT),
}


@pytest.mark.parametrize("policy", sorted(_EGRESS_SITES))
@pytest.mark.asyncio
async def test_the_egress_check_reads_config_off_the_loop(policy):
    """The egress check reads the system config, from the store when this
    process holds none within its staleness bound."""
    held = _Held()
    reads = []

    def get_system_config():
        reads.append(threading.current_thread() is threading.main_thread())
        held.hold()
        raise _Stop()

    dispatcher = _dispatcher(SimpleNamespace(get_system_config=get_system_config))

    async def dispatch():
        with pytest.raises(_Stop):
            await _EGRESS_SITES[policy](dispatcher)

    _, beat = await asyncio.gather(dispatch(), held.heartbeat())

    assert beat == [True]
    assert held.waits == [True]
    assert reads == [False]


@pytest.mark.asyncio
async def test_a_tenants_first_artefact_manager_is_built_off_the_loop():
    """A tenant's first dispatch builds its artefact manager and telemetry
    provider, which imports and constructs the provider's client."""
    held = _Held()
    built = []

    def factory(tenant_id):
        built.append(tenant_id)
        held.hold()
        raise RuntimeError("artefact store unreachable")

    dispatcher = object.__new__(AgentDispatcher)
    dispatcher._artifact_manager_factory = factory

    overlay, beat = await asyncio.gather(
        dispatcher.resolve_artefact_for_request("search_agent", TENANT, "seed"),
        held.heartbeat(),
    )

    assert beat == [True]
    assert held.waits == [True]
    assert built == [TENANT]
    assert (overlay["served_from"], overlay["error"]) == (
        "default",
        "RuntimeError: artefact store unreachable",
    )
