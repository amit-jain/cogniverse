"""Knowledge agents must offload synchronous RLM work off the event loop.

Regression (PERF): multi_document_synthesis / temporal_reasoning /
federated_query called the blocking ``rlm.process(...)`` (a synchronous LM
round-trip) directly inside ``async def`` helpers, stalling the loop. Each is
now ``await asyncio.to_thread(rlm.process, ...)``. The proof is deterministic: a
``process`` that blocks on a threading.Event can only complete if a coroutine
scheduled alongside it gets to run and release it — impossible if it ran on the
loop.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from cogniverse_agents.federated_query_agent import FederatedQueryAgent
from cogniverse_agents.multi_document_synthesis_agent import (
    MultiDocumentSynthesisAgent,
)
from cogniverse_agents.temporal_reasoning_agent import TemporalReasoningAgent

# (agent class, method name, positional args for the 3-arg signature)
_CASES = [
    (FederatedQueryAgent, "_summarise_with_rlm", ("q", "block")),
    (MultiDocumentSynthesisAgent, "_synthesise_with_rlm", ("q", "docs")),
    (TemporalReasoningAgent, "_summarise_with_rlm", ("subject", "block")),
]


@pytest.mark.parametrize("cls, method_name, args", _CASES)
@pytest.mark.asyncio
async def test_with_rlm_offloads_blocking_process(cls, method_name, args, monkeypatch):
    release = threading.Event()

    class _FakeRLM:
        def process(self, **kwargs):
            # Only completes once the concurrent coroutine sets the event, which
            # requires this call to be OFF the event loop.
            assert release.wait(timeout=5), "event loop was blocked by rlm.process"
            return SimpleNamespace(answer="done")

    monkeypatch.setattr(
        "cogniverse_agents.inference.rlm_inference.build_rlm_from_options",
        lambda llm_config, rlm_options: _FakeRLM(),
    )

    agent = cls.__new__(cls)
    agent._llm_config = None
    method = getattr(agent, method_name)
    rlm_options = SimpleNamespace(include_trajectory=False, trajectory_max_entries=0)

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    result, _ = await asyncio.wait_for(
        asyncio.gather(method(*args, rlm_options), releaser()), timeout=5
    )
    assert result == "done"
