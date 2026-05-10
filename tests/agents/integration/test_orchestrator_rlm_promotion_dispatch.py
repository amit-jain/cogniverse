"""Orchestrator stamps rlm.enabled=True on long sub-agent payloads:
real OrchestratorAgent → real _execute_plan → real A2A HTTP POST →
captured peer payload. Bypasses the DSPy planner; the promotion path
itself is production code (_maybe_promote_to_rlm in execute_step)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from cogniverse_agents._rlm_promotion import (
    _RLM_PROMOTION_DEFAULT_FRACTION,
    _RLM_PROMOTION_DEFAULT_THRESHOLD,
)
from cogniverse_agents.orchestrator_agent import (
    AgentStep,
    OrchestrationPlan,
    OrchestratorAgent,
    OrchestratorDeps,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

pytestmark = pytest.mark.integration


CUTOFF_CHARS = int(_RLM_PROMOTION_DEFAULT_THRESHOLD * _RLM_PROMOTION_DEFAULT_FRACTION)


class _RecordingPeer:
    def __init__(self) -> None:
        self.received_payloads: List[Dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def _handler(request: httpx.Request) -> httpx.Response:
            try:
                payload = json.loads(request.content.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {"_raw": request.content.decode("utf-8", errors="replace")}
            self.received_payloads.append({"url": str(request.url), "payload": payload})
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "agent": "search_agent",
                    "results": [{"id": "stub_result", "score": 0.5}],
                },
            )

        return httpx.MockTransport(_handler)


@pytest.fixture
def orchestrator_with_recording_peer():
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="orch_rlm_promotion", config_manager=cm)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://recording-peer.test:8002",
            capabilities=["search"],
            process_endpoint="/agents/search/process",
        )
    )

    peer = _RecordingPeer()
    transport = peer.transport()
    http_client = httpx.AsyncClient(transport=transport)

    orchestrator = OrchestratorAgent(
        deps=OrchestratorDeps(),
        registry=registry,
        config_manager=cm,
        port=8013,
        http_client=http_client,
    )
    yield orchestrator, peer, http_client


def _make_plan_with_payload_chars(target_chars: int) -> OrchestrationPlan:
    # _maybe_promote_to_rlm sums string lengths in agent_input;
    # one big "query" string trips the promotion.
    return OrchestrationPlan(
        query="long-document deep-research query",
        steps=[
            AgentStep(
                agent_name="search_agent",
                input_data={"query": "x" * target_chars},
                depends_on=[],
                reasoning="exercise RLM promotion",
            )
        ],
        parallel_groups=[[0]],
        reasoning="hand-built plan; bypasses DSPy",
    )


@pytest.mark.asyncio
async def test_long_payload_is_promoted_to_rlm_in_dispatched_payload(
    orchestrator_with_recording_peer,
):
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(CUTOFF_CHARS + 1_000)
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_pos"
        )

        assert len(peer.received_payloads) == 1, peer.received_payloads
        body = peer.received_payloads[0]["payload"]
        assert "rlm" in body, (
            f"orchestrator did not stamp rlm despite projected_chars > "
            f"cutoff={CUTOFF_CHARS}; body keys: {list(body.keys())}"
        )
        rlm = body["rlm"]
        assert rlm.get("enabled") is True, rlm
        assert rlm.get("auto_detect") is True, rlm
        assert rlm.get("context_threshold") == _RLM_PROMOTION_DEFAULT_THRESHOLD, rlm
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_short_payload_is_not_promoted(orchestrator_with_recording_peer):
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(100)
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_neg"
        )

        assert len(peer.received_payloads) == 1
        body = peer.received_payloads[0]["payload"]
        assert "rlm" not in body, (
            f"orchestrator stamped rlm on a small payload; must be silent "
            f"under cutoff={CUTOFF_CHARS}. body={body!r}"
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_caller_explicit_rlm_disable_wins_over_promotion(
    orchestrator_with_recording_peer,
):
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(CUTOFF_CHARS + 1_000)
        plan.steps[0].input_data["rlm"] = None
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_explicit"
        )

        body = peer.received_payloads[0]["payload"]
        assert body.get("rlm") is None, (
            f"orchestrator overrode explicit caller rlm=None and promoted; "
            f"got rlm={body.get('rlm')!r}"
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_non_promotable_agent_never_gets_rlm_stamp(
    orchestrator_with_recording_peer,
):
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    orchestrator.registry.register_agent(
        AgentEndpoint(
            name="entity_extraction_agent",
            url="http://recording-peer.test:8002",
            capabilities=["entity_extraction"],
            process_endpoint="/agents/entity_extraction/process",
        )
    )
    try:
        plan = OrchestrationPlan(
            query="x",
            steps=[
                AgentStep(
                    agent_name="entity_extraction_agent",
                    input_data={"query": "x" * (CUTOFF_CHARS + 1_000)},
                    depends_on=[],
                    reasoning="non-promotable agent gets a long payload",
                )
            ],
            parallel_groups=[[0]],
            reasoning="hand-built plan",
        )
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_nonprom"
        )

        body = peer.received_payloads[0]["payload"]
        assert "rlm" not in body, (
            "entity_extraction_agent is not on the RLM-promotable list; "
            f"promoting it would force RLM on agents that don't support it. "
            f"body={body!r}"
        )
    finally:
        await http_client.aclose()
