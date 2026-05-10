"""Integration: Orchestrator stamps rlm.enabled=True on long sub-agent payloads.

When the orchestrator's projected sub-agent payload exceeds the RLM
promotion cutoff (50,000 chars × 0.75 default fraction = 37,500), the
dispatch path stamps ``rlm.enabled=True`` onto the payload BEFORE the
A2A HTTP POST so the RLM-aware sub-agent (search, deep_research,
detailed_report, coding) recursively decomposes its context instead of
jamming everything into one prompt.

The promotion logic itself is unit-tested in
``tests/agents/unit/test_orch_rlm_promotion.py``. This file closes the
gap that audit flagged: a real-Orchestrator → real-AgentRegistry →
real-_execute_plan → real-A2A-HTTP test that captures the payload the
sub-agent actually received and asserts ``rlm.enabled=True``.

The sub-agent peer is a ``httpx.MockTransport`` that records every
inbound request and returns a stub success response. The orchestrator
runs end-to-end up to and including the HTTP POST; only the peer's
response is stubbed (the test is about what the orchestrator SENDS,
not what the peer DOES).

Bypasses the DSPy planner: builds an ``OrchestrationPlan`` by hand with
one large ``AgentStep`` and feeds it to ``_execute_plan``. That keeps
the test deterministic and fast, and the RLM promotion path is exactly
the same code as in production (``_maybe_promote_to_rlm`` is called
unconditionally inside ``execute_step``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from cogniverse_agents.orchestrator_agent import (
    _RLM_PROMOTION_DEFAULT_FRACTION,
    _RLM_PROMOTION_DEFAULT_THRESHOLD,
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
    """Captures every POST sent to an A2A peer and returns a stub response."""

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
    """Real OrchestratorAgent with the search_agent peer wired to a MockTransport."""
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
    """Build a one-step plan whose input_data sums to ``target_chars`` of strings.

    ``_maybe_promote_to_rlm`` projects the payload size by summing the
    lengths of every string-shaped value in ``agent_input``. Putting one
    big ``query`` string is the simplest path that the projection
    counts in full.
    """
    return OrchestrationPlan(
        query="long-document deep-research query",
        steps=[
            AgentStep(
                agent_name="search_agent",
                input_data={
                    # The orchestrator pops "query" off agent_input before the
                    # POST and rebuilds it into the top-level payload, but
                    # the projection reads input_data BEFORE the pop, so
                    # putting the bulk here is what trips the promotion.
                    "query": "x" * target_chars,
                },
                depends_on=[],
                reasoning="single-step plan to exercise RLM promotion",
            )
        ],
        parallel_groups=[[0]],
        reasoning="hand-built plan; bypasses DSPy",
    )


@pytest.mark.asyncio
async def test_long_payload_is_promoted_to_rlm_in_dispatched_payload(
    orchestrator_with_recording_peer,
):
    """Payload above the cutoff → dispatched POST carries rlm.enabled=True."""
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(CUTOFF_CHARS + 1_000)
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_pos"
        )

        assert len(peer.received_payloads) == 1, peer.received_payloads
        body = peer.received_payloads[0]["payload"]
        assert "rlm" in body, (
            "orchestrator did not stamp rlm on the dispatched payload "
            f"despite projected_chars > cutoff={CUTOFF_CHARS}; "
            f"received body keys: {list(body.keys())}"
        )
        rlm = body["rlm"]
        assert rlm.get("enabled") is True, rlm
        assert rlm.get("auto_detect") is True, rlm
        assert rlm.get("context_threshold") == _RLM_PROMOTION_DEFAULT_THRESHOLD, rlm
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_short_payload_is_not_promoted(orchestrator_with_recording_peer):
    """Payload below the cutoff → dispatched POST has no rlm field."""
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(100)
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_neg"
        )

        assert len(peer.received_payloads) == 1
        body = peer.received_payloads[0]["payload"]
        assert "rlm" not in body, (
            f"orchestrator stamped rlm on a small payload; promotion must be "
            f"silent under the cutoff. body={body!r}"
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_caller_explicit_rlm_disable_wins_over_promotion(
    orchestrator_with_recording_peer,
):
    """Explicit caller opt-out (rlm=None) is preserved even on long payloads."""
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    try:
        plan = _make_plan_with_payload_chars(CUTOFF_CHARS + 1_000)
        # Caller explicitly sets rlm=None — that means "do not promote".
        plan.steps[0].input_data["rlm"] = None
        await orchestrator._execute_plan(
            plan, tenant_id="orch_rlm_promotion", workflow_id="wf_rlm_explicit"
        )

        body = peer.received_payloads[0]["payload"]
        # rlm field is preserved (None or absent) — the orchestrator must
        # not overwrite an explicit caller choice.
        assert body.get("rlm") is None, (
            "orchestrator overrode explicit caller rlm=None and promoted "
            f"anyway; got rlm={body.get('rlm')!r}"
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_non_promotable_agent_never_gets_rlm_stamp(
    orchestrator_with_recording_peer,
):
    """Promotion only fires for agents in _RLM_PROMOTABLE_AGENTS."""
    orchestrator, peer, http_client = orchestrator_with_recording_peer
    # Register a non-promotable peer at the same MockTransport.
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
