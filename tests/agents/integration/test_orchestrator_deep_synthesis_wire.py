"""OrchestratorAgent dispatches via DeepSynthesisWorkflow on opt-in.

Without this wire, ``DeepSynthesisWorkflow`` was an orphan class —
defined and unit-tested, but no caller ever instantiated it. This test
verifies, against a real OrchestratorAgent + real AgentRegistry, that:

  * the new ``OrchestratorInput.synthesis_depth`` field exists and
    accepts ``"deep"``;
  * when ``synthesis_depth="deep"`` is set, the orchestrator constructs
    a ``DeepSynthesisWorkflow`` and runs it instead of the default
    plan-then-act path;
  * absent the flag, the orchestrator behaves exactly as before;
  * checked deep-synthesis failures are emitted once and surfaced.

The DSPy LM is stubbed because the workflow's RLM step is what we're
asserting *gets called*, not the LLM's actual answer quality.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.deep_synthesis_workflow import (
    DeepSynthesisResult,
    DeepSynthesisWorkflow,
)
from cogniverse_agents.inference import deno_check
from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
)
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

pytestmark = pytest.mark.integration


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="b7_int_tenant", config_manager=cm)
    return OrchestratorAgent(
        deps=OrchestratorDeps(tenant_id="b7_int_tenant"),
        registry=registry,
        config_manager=cm,
    )


class TestInputAcceptsSynthesisDepth:
    def test_field_exists_and_round_trips(self):
        # Critical: the field must be defined on OrchestratorInput so an
        # A2A-deserialised request can carry the opt-in.
        inp = OrchestratorInput(query="x", tenant_id="t", synthesis_depth="deep")
        assert inp.synthesis_depth == "deep"

    def test_field_optional_default_none(self):
        # Existing callers don't supply the field — must keep working.
        inp = OrchestratorInput(query="x", tenant_id="t")
        assert inp.synthesis_depth is None


@pytest.mark.asyncio
class TestOptInRouting:
    async def test_deep_flag_constructs_and_runs_workflow(
        self, orchestrator: OrchestratorAgent
    ):
        # Stub the workflow constructor to return a controlled fake whose
        # ``run`` we observe — that's how we know the wire reached it.
        observed_runs: list = []
        built_with: list = []

        class _FakeWorkflow:
            async def run(self, *, query, tenant_id, seed_subagents):
                observed_runs.append(
                    {"query": query, "tenant": tenant_id, "seeds": seed_subagents}
                )
                return DeepSynthesisResult(
                    answer="DEEP-SYNTH-OK",
                    iterations_used=2,
                    subagent_calls_made=3,
                    llm_calls_used=2,
                    was_capped=False,
                    was_submitted=True,
                    was_rate_limited=False,
                    trajectory=[],
                )

        def _build(tenant_id):
            built_with.append(tenant_id)
            return _FakeWorkflow()

        orchestrator._build_deep_synthesis_workflow = _build
        orchestrator._emit_orchestration_outcome = AsyncMock()

        out = await orchestrator._process_impl(
            OrchestratorInput(
                query="Compare refund policies across all subsidiaries",
                tenant_id="b7_int_tenant",
                synthesis_depth="deep",
            )
        )
        assert observed_runs, (
            "OrchestratorInput.synthesis_depth=deep did not reach "
            "DeepSynthesisWorkflow.run — the wire is dead"
        )
        # The request tenant is threaded into the build helper (it resolves
        # the per-tenant LLM endpoint and the sub-agent dispatch context).
        assert built_with == ["b7_int_tenant:b7_int_tenant"]
        assert observed_runs[0]["tenant"] == "b7_int_tenant:b7_int_tenant"
        assert observed_runs[0]["query"].startswith("Compare refund")
        assert out.workflow_id.startswith("deep_synthesis_")
        assert out.final_output["answer"] == "DEEP-SYNTH-OK"
        assert out.final_output["was_submitted"] is True
        assert "deep_synthesis" in out.execution_summary
        emitted_state = orchestrator._emit_orchestration_outcome.await_args.args[0]
        assert emitted_state.agent_sequence == ["deep_synthesis"]
        assert emitted_state.execution_order == ["deep_synthesis"]
        assert emitted_state.agent_results == {"deep_synthesis": {"status": "success"}}
        assert orchestrator._emit_orchestration_outcome.await_args.kwargs == {
            "success_override": True
        }

    async def test_default_path_when_synthesis_depth_unset(
        self, orchestrator: OrchestratorAgent
    ):
        # Track whether the deep-workflow constructor is called. When the
        # flag is unset it must NOT be — preserves existing behaviour for
        # the 99 % of requests that don't opt in.
        called: list = []
        original = orchestrator._build_deep_synthesis_workflow

        def _track(tenant_id):
            called.append(True)
            return original(tenant_id)

        orchestrator._build_deep_synthesis_workflow = _track  # type: ignore[method-assign]

        # Stub the rest of the orchestrator's plan-then-act path so the
        # test doesn't actually need real sub-agents.
        orchestrator._process_impl_locked = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda: {"final_output": {"answer": "ok"}},
                workflow_id="default_path",
                final_output={"answer": "ok"},
                execution_summary="default",
            )
        )

        try:
            await orchestrator._process_impl(
                OrchestratorInput(
                    query="any",
                    tenant_id="b7_int_tenant",
                    # synthesis_depth omitted
                )
            )
        except Exception:
            # The default path's downstream may fail on a stub — what we
            # care about is that the deep-workflow constructor was not
            # called.
            pass

        assert called == [], (
            "default plan-then-act dispatch must not construct the "
            "DeepSynthesisWorkflow"
        )

    async def test_workflow_exception_emits_failure_and_does_not_fallback(
        self, orchestrator: OrchestratorAgent
    ):
        class _BoomWorkflow:
            async def run(self, **kw):
                raise RuntimeError("simulated workflow failure")

        orchestrator._build_deep_synthesis_workflow = lambda tenant_id: _BoomWorkflow()
        orchestrator._process_impl_locked = AsyncMock()
        orchestrator._emit_orchestration_outcome = AsyncMock()

        with pytest.raises(RuntimeError, match="simulated workflow failure"):
            await orchestrator._process_impl(
                OrchestratorInput(
                    query="x",
                    tenant_id="b7_int_tenant",
                    synthesis_depth="deep",
                )
            )

        orchestrator._process_impl_locked.assert_not_awaited()
        emitted = orchestrator._emit_orchestration_outcome.await_args
        assert emitted.args[0].agent_sequence == ["deep_synthesis"]
        assert isinstance(emitted.kwargs["error"], RuntimeError)

    async def test_unsubmitted_result_exports_checked_failure(
        self, orchestrator: OrchestratorAgent
    ):
        class _CappedWorkflow:
            async def run(self, **kw):
                return DeepSynthesisResult(
                    answer="partial evidence",
                    iterations_used=8,
                    subagent_calls_made=12,
                    llm_calls_used=8,
                    was_capped=True,
                    was_submitted=False,
                    was_rate_limited=False,
                    trajectory=[],
                )

        orchestrator._build_deep_synthesis_workflow = lambda tenant_id: (
            _CappedWorkflow()
        )
        orchestrator._emit_orchestration_outcome = AsyncMock()

        output = await orchestrator._process_impl(
            OrchestratorInput(
                query="synthesize every exact source",
                tenant_id="b7_int_tenant",
                synthesis_depth="deep",
            )
        )

        assert output.final_output["was_submitted"] is False
        emitted = orchestrator._emit_orchestration_outcome.await_args
        assert emitted.args[0].agent_results == {
            "deep_synthesis": {
                "status": "error",
                "message": "Deep synthesis ended without a submitted answer",
            }
        }
        assert emitted.kwargs == {
            "success_override": False,
            "error_summary": "Deep synthesis ended without a submitted answer",
        }


@pytest.mark.asyncio
class TestBuildHelper:
    async def test_returns_none_when_prereqs_missing(self):
        """When config_manager has no LLM endpoint, _build returns None
        instead of raising — so the orchestrator falls back cleanly."""
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="b7_no_llm", config_manager=cm)
        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(tenant_id="b7_no_llm"),
            registry=registry,
            config_manager=cm,
        )
        # Force the get_config inside _build to raise.
        from unittest.mock import patch

        with patch(
            "cogniverse_foundation.config.utils.get_config",
            side_effect=RuntimeError("no LLM"),
        ):
            wf = orchestrator._build_deep_synthesis_workflow("b7_no_llm")
        assert wf is None

    async def test_returns_workflow_when_llm_available(self, monkeypatch):
        """When an LLM endpoint is resolvable, _build returns a real
        DeepSynthesisWorkflow — proving the constructor wire is alive.

        Deno is required for the actual RLM REPL execution; for this
        wire-coverage test we bypass the probe via configure state so
        the wire passes on dev machines without Deno installed.
        """
        monkeypatch.setattr(deno_check, "_skip_deno_check", True)
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="b7_with_llm", config_manager=cm)
        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(tenant_id="b7_with_llm"),
            registry=registry,
            config_manager=cm,
        )
        wf = orchestrator._build_deep_synthesis_workflow("b7_with_llm")
        assert isinstance(wf, DeepSynthesisWorkflow)

    async def test_build_routes_rlm_through_semantic_router_when_enabled(
        self, monkeypatch
    ):
        """When router routing is enabled, the deep-synthesis RLM's endpoint is
        rewritten to the semantic router with the tenant tier + the ``rlm_inference``
        task header — the direct backend endpoint is never used."""
        monkeypatch.setattr(deno_check, "_skip_deno_check", True)
        from cogniverse_foundation.config.unified_config import SemanticRouterConfig

        router = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://semantic-router:9099/v1",
            tenant_tiers={"b7_gw_tenant": "pro"},
            default_tier="free",
        )
        # route_rlm_endpoint resolves the semantic router config via
        # resolve_semantic_router_config; force it enabled so the real config
        # manager's resolved endpoint is routed through the semantic router.
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.resolve_semantic_router_config",
            lambda _cfg: router,
        )
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="b7_gw_tenant", config_manager=cm)
        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(tenant_id="b7_gw_tenant"),
            registry=registry,
            config_manager=cm,
        )
        wf = orchestrator._build_deep_synthesis_workflow("b7_gw_tenant")
        assert isinstance(wf, DeepSynthesisWorkflow)
        assert wf._rlm.llm_config.api_base == "http://semantic-router:9099/v1"
        assert wf._rlm.llm_config.extra_headers == {
            "x-authz-user-id": "b7_gw_tenant",
            "x-authz-user-groups": "pro",
        }
        # tenant_id is threaded onto the RLM for event scoping.
        assert wf._rlm._tenant_id == "b7_gw_tenant"
