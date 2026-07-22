"""OrchestratorAgent must run synchronous Mem0 calls off the event loop.

``get_relevant_context`` (Mem0 vector search) and ``remember_success`` (Mem0
LLM fact-extraction add) are synchronous multi-second round trips; called inline
from ``_process_impl_locked`` they froze the loop. Each is now
``await asyncio.to_thread(...)``. The proof is deterministic: the offloaded call
records ``threading.get_ident()``, which must differ from the loop's thread.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorInput

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StopHere(Exception):
    pass


@pytest.mark.asyncio
async def test_get_relevant_context_runs_off_the_event_loop():
    recorded = {}

    def _rec_ctx(query):
        recorded["thread"] = threading.get_ident()
        raise _StopHere()  # halt right after the offloaded call

    d = object.__new__(OrchestratorAgent)
    d._ensure_memory_for_tenant = lambda *a, **k: None
    d.emit_progress = lambda *a, **k: None
    d.get_relevant_context = _rec_ctx

    inp = OrchestratorInput(query="q", tenant_id="acme:acme")
    loop_thread = threading.get_ident()

    with pytest.raises(_StopHere):
        await d._process_impl_locked(inp, "wf", "q", "acme:acme", None)

    assert recorded.get("thread") is not None
    # to_thread offload => get_relevant_context ran on a worker thread.
    assert recorded["thread"] != loop_thread


@pytest.mark.asyncio
async def test_ensure_memory_for_tenant_runs_off_the_event_loop():
    """First-touch Mem0 init (config reads + embedder + Vespa vector store
    construction) is a multi-second synchronous build; run inline it froze the
    loop on the first request per tenant. It is now offloaded via to_thread."""
    recorded = {}

    def _rec_ensure(tenant_id):
        recorded["thread"] = threading.get_ident()

    def _stop_ctx(query):
        raise _StopHere()  # halt right after the offloaded init

    d = object.__new__(OrchestratorAgent)
    d._ensure_memory_for_tenant = _rec_ensure
    d.emit_progress = lambda *a, **k: None
    d.get_relevant_context = _stop_ctx

    inp = OrchestratorInput(query="q", tenant_id="acme:acme")
    loop_thread = threading.get_ident()

    with pytest.raises(_StopHere):
        await d._process_impl_locked(inp, "wf", "q", "acme:acme", None)

    assert recorded.get("thread") is not None
    assert recorded["thread"] != loop_thread, (
        "the first-touch Mem0 init ran on the event loop"
    )


def test_ensure_memory_for_tenant_inits_once_under_concurrency(monkeypatch):
    """Offloaded to worker threads, concurrent first-touch requests for one
    tenant must initialize memory exactly once. A lock-less guard lets N threads
    each build a full Mem0 stack (embedder + Vespa vector store), leaking every
    loser's client."""
    orch = object.__new__(OrchestratorAgent)
    orch._memory_initialized_tenants = set()
    orch._memory_init_lock = threading.Lock()
    orch._config_manager = MagicMock()
    orch._config_manager.get_system_config.return_value = SimpleNamespace(
        inference_service_urls={"denseon": "http://denseon:1"}
    )

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda **k: SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(
                resolve=lambda name: SimpleNamespace(
                    model="ollama/x", api_base="http://lm:1"
                )
            )
        ),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.bootstrap.BootstrapConfig.from_environment",
        classmethod(
            lambda cls: SimpleNamespace(backend_url="http://vespa", backend_port=8080)
        ),
    )

    calls = []
    barrier = threading.Barrier(8)

    def _init(**kwargs):
        calls.append(kwargs["tenant_id"])
        time.sleep(0.05)  # widen the race window

    orch.initialize_memory = _init

    def worker():
        barrier.wait()
        orch._ensure_memory_for_tenant("acme:acme")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == ["acme:acme"], (
        f"memory initialized {len(calls)}x under concurrency (expected 1)"
    )
