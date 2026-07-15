"""DeepSynthesisWorkflow's sub-agent dispatcher makes real HTTP calls.

the dispatcher in ``OrchestratorAgent._build_deep_synthesis_workflow``
returned the stub string ``f"(stub dispatch to {sub_agent_name}: …)"``
regardless of agent registration. The deep-synthesis loop ran end-to-end
but every round synthesised against placeholder content — no real
sub-agent output, no real reasoning across knowledge.

This test verifies the wire end-to-end:

  * a real registered sub-agent endpoint receives a POST when the
    DeepSynthesisWorkflow's RLM step emits ``ASK(<agent>: <q>)``;
  * the response's ``answer`` field is propagated back as the snippet
    the next RLM iteration sees as evidence;
  * unregistered sub-agents return a clear error string (not a
    silent placeholder), so the RLM step can decide to route elsewhere;
  * sub-agent HTTP failures don't crash the workflow — they're
    surfaced as a dispatch-failed snippet.

We use a real httpx-backed local server (FastAPI TestClient over
ASGI) as the sub-agent target so the dispatcher's POST actually
hits a process and returns real JSON. No mocking of the HTTP layer.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.routers.agents import AgentTask

pytestmark = pytest.mark.integration


@pytest.fixture
def fake_subagent_app() -> FastAPI:
    """A real FastAPI app standing in for a sub-agent's process endpoint.

    Records every POST so the test can assert the dispatcher actually
    delivered the right query.
    """
    app = FastAPI()
    app.state.calls: list[Dict[str, Any]] = []

    # Validate the REAL AgentTask contract — the same boundary the production
    # /agents/{name}/process route enforces. agent_name is required, so a
    # dispatch payload that omits it returns 422 exactly as production would.
    @app.post("/agents/{agent_name}/process")
    async def _process(agent_name: str, task: AgentTask):
        app.state.calls.append(task.model_dump())
        return {"answer": f"REAL_SUBAGENT_ANSWER for {task.query[:40]}"}

    return app


@pytest.fixture
def orchestrator_with_subagent(
    fake_subagent_app: FastAPI,
) -> tuple[OrchestratorAgent, FastAPI, AgentRegistry]:
    """OrchestratorAgent whose http_client routes to the FastAPI test app.

    httpx.AsyncClient supports `transport=ASGITransport(app=...)` so the
    dispatcher's POST actually executes the FastAPI handler in-process,
    no real socket — a true integration test of the dispatcher's HTTP
    contract without needing a separate process.
    """
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="f41_dispatcher", config_manager=cm)
    # Register the fake sub-agent so the dispatcher resolves it.
    ep = AgentEndpoint(
        name="fake_search",
        url="http://fake-subagent.test",
        capabilities=["video_search"],
        health_endpoint="/health",
        process_endpoint="/agents/fake_search/process",
        timeout=10,
    )
    registry.agents = {"fake_search": ep}

    transport = httpx.ASGITransport(app=fake_subagent_app)
    client = httpx.AsyncClient(
        transport=transport, base_url="http://fake-subagent.test"
    )

    orchestrator = OrchestratorAgent(
        deps=OrchestratorDeps(tenant_id="f41_test_tenant"),
        registry=registry,
        config_manager=cm,
        http_client=client,
    )
    return orchestrator, fake_subagent_app, registry


@pytest.mark.asyncio
class TestRealSubagentDispatch:
    async def test_known_subagent_receives_real_post(
        self, orchestrator_with_subagent, monkeypatch
    ):
        # Skip Deno requirement so RLMInference constructor doesn't blow up
        # when _build_deep_synthesis_workflow runs.
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        orchestrator, app, _ = orchestrator_with_subagent
        # Build with a REQUEST tenant distinct from any deps value — the
        # dispatch must carry THIS tenant, not deps.tenant_id / __system__.
        workflow = orchestrator._build_deep_synthesis_workflow("req_tenant_xyz")
        assert workflow is not None

        # Pull the inner dispatcher out of the constructed workflow.
        dispatcher_fn = workflow._dispatch
        snippet = await dispatcher_fn("What is the capital of France?", "fake_search")
        # Real HTTP roundtrip against the real AgentTask contract.
        assert len(app.state.calls) == 1
        recorded = app.state.calls[0]
        # agent_name is required by AgentTask; a payload missing it 422s.
        assert recorded["agent_name"] == "fake_search"
        assert recorded["query"] == "What is the capital of France?"
        # The REQUEST tenant reached the sub-agent (not deps / __system__).
        assert recorded["context"]["tenant_id"] == "req_tenant_xyz"
        # Answer field flows back as the snippet.
        assert "REAL_SUBAGENT_ANSWER" in snippet
        assert snippet.endswith("What is the capital of France?")

    async def test_unregistered_subagent_returns_error_string(
        self, orchestrator_with_subagent, monkeypatch
    ):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        orchestrator, app, _ = orchestrator_with_subagent
        workflow = orchestrator._build_deep_synthesis_workflow("t")
        snippet = await workflow._dispatch("anything", "agent_does_not_exist")
        # No HTTP call made for the unregistered agent.
        assert app.state.calls == []
        assert "not registered" in snippet
        # Specifically the agent name must appear so the RLM step knows
        # which one was missing.
        assert "agent_does_not_exist" in snippet

    async def test_subagent_5xx_surfaces_as_failure_snippet(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f41_5xx", config_manager=cm)
        registry.agents = {
            "broken_agent": AgentEndpoint(
                name="broken_agent",
                url="http://broken.test",
                capabilities=["video_search"],
                health_endpoint="/health",
                process_endpoint="/agents/broken_agent/process",
                timeout=5,
            )
        }

        # Build a FastAPI that always 500s.
        broken_app = FastAPI()

        @broken_app.post("/agents/broken_agent/process")
        async def _broken(payload: Dict[str, Any]):
            from fastapi import HTTPException

            raise HTTPException(500, "intentional")

        transport = httpx.ASGITransport(app=broken_app)
        client = httpx.AsyncClient(transport=transport, base_url="http://broken.test")
        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(tenant_id="f41_5xx_tenant"),
            registry=registry,
            config_manager=cm,
            http_client=client,
        )
        workflow = orchestrator._build_deep_synthesis_workflow("t")
        snippet = await workflow._dispatch("q", "broken_agent")
        # Workflow must continue (not raise); snippet says call failed.
        assert "broken_agent" in snippet
        assert "failed" in snippet.lower()

    async def test_dispatcher_response_without_known_answer_field_is_serialised(
        self, monkeypatch
    ):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f41_weird", config_manager=cm)
        registry.agents = {
            "weird_agent": AgentEndpoint(
                name="weird_agent",
                url="http://weird.test",
                capabilities=[],
                health_endpoint="/health",
                process_endpoint="/agents/weird_agent/process",
                timeout=5,
            )
        }

        weird_app = FastAPI()

        @weird_app.post("/agents/weird_agent/process")
        async def _weird(payload: Dict[str, Any]):
            # Returns a non-standard shape — none of {answer, output,
            # content, result, summary} present.
            return {"unexpected_key": "weird value", "items": [1, 2, 3]}

        transport = httpx.ASGITransport(app=weird_app)
        client = httpx.AsyncClient(transport=transport, base_url="http://weird.test")
        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(tenant_id="f41_weird_tenant"),
            registry=registry,
            config_manager=cm,
            http_client=client,
        )
        workflow = orchestrator._build_deep_synthesis_workflow("t")
        snippet = await workflow._dispatch("q", "weird_agent")
        # Snippet must contain the response payload (so the RLM step
        # can still try to make sense of it). Specifically the unique
        # marker word from the response.
        assert "weird value" in snippet


@pytest.mark.asyncio
class TestEndToEndWorkflowRunWithRealDispatch:
    async def test_workflow_run_ends_with_real_subagent_evidence(
        self, orchestrator_with_subagent, monkeypatch
    ):
        """Run the full workflow with a stubbed RLM so the loop terminates
        deterministically; assert the dispatched evidence in its trajectory
        comes from the REAL sub-agent, not a stub string."""
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        orchestrator, app, _ = orchestrator_with_subagent
        workflow = orchestrator._build_deep_synthesis_workflow("t")

        # Replace the RLM with one that emits SUBMIT() immediately so the
        # loop completes after the seed fan-out — we want to assert the
        # seed dispatch reached the real sub-agent, not test the RLM's
        # own behavior.
        from cogniverse_agents.deep_synthesis_workflow import (
            SUBMIT_TOKEN,
        )

        class _ImmediateSubmitRLM:
            def process(self, *, query, context, **_kw):
                from dataclasses import dataclass

                @dataclass
                class _R:
                    answer: str
                    tokens_used: int = 0
                    was_fallback: bool = False
                    iterations: int = 1
                    trajectory: list = None

                return _R(answer=f"DONE {SUBMIT_TOKEN}")

        workflow._rlm = _ImmediateSubmitRLM()

        result = await workflow.run(
            query="anything",
            tenant_id="f41_test_tenant",
            seed_subagents=["fake_search"],
        )
        # The seed sub-agent received a real POST.
        assert len(app.state.calls) == 1
        # The trajectory contains the real sub-agent's answer, not
        # any "(stub dispatch to ...)" placeholder.
        subagent_entries = [e for e in result.trajectory if e.get("kind") == "subagent"]
        assert len(subagent_entries) == 1
        assert "REAL_SUBAGENT_ANSWER" in subagent_entries[0]["snippet"]
        assert "stub dispatch" not in subagent_entries[0]["snippet"], (
            "the dispatcher's stub-string fallback must not appear in "
            "trajectories anymore — the deep-synthesis workflow now goes "
            "through real A2A HTTP dispatch"
        )
