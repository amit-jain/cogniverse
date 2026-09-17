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
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cogniverse_agents.federated_query_agent import FederatedQueryAgent
from cogniverse_agents.knowledge_summarization_agent import (
    KnowledgeSummarizationAgent,
)
from cogniverse_agents.multi_document_synthesis_agent import (
    MultiDocumentSynthesisAgent,
)
from cogniverse_agents.temporal_reasoning_agent import TemporalReasoningAgent
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.parametrize("agent_kind", ["research", "coding"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.asyncio
async def test_optional_rlm_keeps_concurrent_requests_running(agent_kind, fail):
    from cogniverse_agents.coding_agent import CodingAgent, CodingInput
    from cogniverse_agents.deep_research_agent import (
        DeepResearchAgent,
        DeepResearchInput,
    )
    from cogniverse_agents.inference.rlm_inference import RLMResult
    from cogniverse_core.agents.rlm_options import RLMOptions

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def synthesis(**kwargs):
        calls.append((kwargs["query"], kwargs["tenant_id"]))
        entered.set()
        assert release.wait(3), "RLM held the serving event loop"
        if fail:
            raise ConnectionError("tenant-a RLM connection reset")
        return RLMResult("tenant-a synthesis", 1, 1, 17, 2.0)

    async def identity(query, _):
        return query

    async def context(*args):
        return "tenant-a context"

    async def questions(*args):
        return ["tenant-a question"]

    async def evidence(*args):
        return [{"question": "tenant-a question", "results": []}]

    async def evaluate(*args):
        return True, [], 0.9

    async def generate(*args):
        return "print(17)", "python solution.py"

    async def execute(*args):
        return {"stdout": "17\n", "stderr": "", "exit_code": 0}

    async def accepted(*args):
        return True, "accepted"

    if agent_kind == "research":
        agent = object.__new__(DeepResearchAgent)
        agent._decompose = questions
        agent._search_parallel = evidence
        agent._evaluate_evidence = evaluate
        agent._synthesize = context
        request = DeepResearchInput(
            query="tenant-a task", tenant_id="tenant:a", rlm=RLMOptions(enabled=True)
        )
        operation = agent._research
    else:

        class SandboxOwner:
            @asynccontextmanager
            async def task_session(self, agent_type, tenant_id):
                yield object()

        agent = object.__new__(CodingAgent)
        agent._sandbox_manager = SandboxOwner()
        agent._search_code_context = context
        agent._plan = context
        agent._generate_code = generate
        agent._execute_in_sandbox = execute
        agent._evaluate_output = accepted
        request = CodingInput(
            task="tenant-a task", tenant_id="tenant:a", rlm=RLMOptions(enabled=True)
        )
        operation = agent._process_impl
    agent.set_tenant_for_context = lambda tenant: None
    agent.inject_context_into_prompt_async = identity
    agent.emit_progress = lambda *args: None
    agent.process_with_rlm = synthesis

    async def concurrent_request():
        assert await asyncio.to_thread(entered.wait, 3)
        release.set()
        return {"tenant": "tenant:b", "status": "healthy"}

    output, independent = await asyncio.gather(operation(request), concurrent_request())
    assert independent == {"tenant": "tenant:b", "status": "healthy"}
    assert calls == [("tenant-a task", "tenant:a")]
    if fail:
        assert output.rlm_synthesis is None
        assert output.rlm_telemetry == {
            "rlm_enabled": False,
            "rlm_attempted": True,
            "rlm_error": "tenant-a RLM connection reset",
        }
    else:
        assert output.rlm_synthesis == "tenant-a synthesis"
        assert output.rlm_telemetry == {
            "rlm_enabled": True,
            "rlm_depth_reached": 1,
            "rlm_total_calls": 1,
            "rlm_tokens_used": 17,
            "rlm_latency_ms": 2.0,
            "rlm_was_fallback": False,
            "rlm_trajectory_length": 0,
            "context_size_chars": len(
                "## tenant-a question\n[]"
                if agent_kind == "research"
                else "tenant-a context"
            ),
        }


def _memory_config_manager():
    """The injected ConfigManager the agent reads, over an in-memory store."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


# (agent class, method name, positional args for the 3-arg signature)
_CASES = [
    (FederatedQueryAgent, "_summarise_with_rlm", ("q", "block")),
    (MultiDocumentSynthesisAgent, "_synthesise_with_rlm", ("q", "docs")),
    (TemporalReasoningAgent, "_summarise_with_rlm", ("subject", "block")),
    (KnowledgeSummarizationAgent, "_summarise_with_rlm", ("title", "block")),
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
        lambda llm_config, rlm_options, **kwargs: _FakeRLM(),
    )

    agent = cls.__new__(cls)
    agent._llm_config = None
    agent.bind_config_manager(_memory_config_manager())
    method = getattr(agent, method_name)
    rlm_options = SimpleNamespace(include_trajectory=False, trajectory_max_entries=0)

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    result, _ = await asyncio.wait_for(
        asyncio.gather(method(*args, rlm_options), releaser()), timeout=5
    )
    assert result == "done"


@pytest.mark.asyncio
async def test_detailed_report_process_impl_offloads_rlm():
    """DetailedReportAgent._process_impl must offload the sync RLM round-trip.

    The blocking process_with_rlm can only return if the concurrently scheduled
    releaser sets the event — impossible while it holds the loop thread.
    """
    from cogniverse_agents.detailed_report_agent import (
        DetailedReportAgent,
        DetailedReportInput,
        ReportResult,
        ThinkingPhase,
    )
    from cogniverse_agents.inference.rlm_inference import RLMResult
    from cogniverse_core.agents.rlm_options import RLMOptions

    release = threading.Event()
    rlm_tenants = []

    def blocking_process_with_rlm(query, context, rlm_options, tenant_id):
        rlm_tenants.append(tenant_id)
        assert release.wait(timeout=5), "event loop was blocked by process_with_rlm"
        return RLMResult(
            answer="synth",
            depth_reached=1,
            total_calls=1,
            tokens_used=1,
            latency_ms=1.0,
        )

    async def _fake_generate_report(request):
        return ReportResult(
            executive_summary="s",
            detailed_findings=[],
            visual_analysis=[],
            technical_details=[],
            recommendations=[],
            confidence_assessment={},
            thinking_phase=ThinkingPhase(
                content_analysis={"avg_relevance": 0.0, "total_results": 0},
                visual_assessment={},
                technical_findings=[],
                patterns_identified=[],
                gaps_and_limitations=[],
                reasoning="",
            ),
            metadata={},
        )

    agent = DetailedReportAgent.__new__(DetailedReportAgent)
    agent.emit_progress = lambda *a, **k: None
    agent._generate_report = _fake_generate_report
    agent.should_use_rlm_for_query = lambda *a, **k: True
    agent.process_with_rlm = blocking_process_with_rlm

    inp = DetailedReportInput(
        query="q",
        enhanced_query="q",
        tenant_id=None,
        search_results=[{"title": "t", "score": 0.9, "description": "d"}],
        rlm=RLMOptions(enabled=True),
    )

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    out, _ = await asyncio.wait_for(
        asyncio.gather(agent._process_impl(inp), releaser()), timeout=5
    )
    assert out.rlm_synthesis == "synth"
    assert out.rlm_telemetry["rlm_enabled"] is True
    assert out.rlm_telemetry["rlm_total_calls"] == 1
    # A report with no tenant runs its RLM as the system tenant, the tenant
    # its LM is routed for.
    assert rlm_tenants == [SYSTEM_TENANT_ID]


@pytest.mark.asyncio
async def test_knowledge_summarization_offloads_dspy():
    """KnowledgeSummarizationAgent._process_impl must offload the DSPy LM call.

    The non-RLM path runs self._dspy_module directly; a module that blocks on a
    threading.Event completes only when the loop stays free for the releaser.
    """
    from cogniverse_agents.knowledge_summarization_agent import (
        KnowledgeSummarizationAgent,
        KnowledgeSummarizationInput,
    )

    release = threading.Event()

    class _FakeModule:
        def __call__(self, **kw):
            assert release.wait(timeout=3), "event loop blocked by DSPy call"
            return SimpleNamespace(summary="done")

    agent = KnowledgeSummarizationAgent.__new__(KnowledgeSummarizationAgent)
    agent.deps = SimpleNamespace(tenant_id=None)
    agent._llm_config = None
    agent._graph_manager = None
    agent._dspy_module = _FakeModule()
    agent._fetch_filtered = lambda *a, **k: [
        {"id": "m1", "memory": "hello", "metadata": {}}
    ]

    inp = KnowledgeSummarizationInput(
        tenant_id="acme:acme", title="t", actor_id="admin"
    )

    async def releaser():
        await asyncio.sleep(0.05)
        release.set()

    out, _ = await asyncio.wait_for(
        asyncio.gather(agent._process_impl(inp), releaser()), timeout=15
    )
    assert out.summary == "done"
    assert out.source_count == 1
    assert out.used_rlm is False
