"""OrchestratorAgent must run synchronous Mem0 calls off the event loop.

``get_relevant_context`` (Mem0 vector search) and ``remember_success`` (Mem0
LLM fact-extraction add) are synchronous multi-second round trips; called inline
from ``_process_impl_locked`` they froze the loop. Each is now
``await asyncio.to_thread(...)``. The proof is deterministic: the offloaded call
records ``threading.get_ident()``, which must differ from the loop's thread.
"""

from __future__ import annotations

import threading

import pytest

from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorInput


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
