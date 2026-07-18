"""Integration tests for the orchestrator's iterative retrieval loop.

Exercises ``OrchestratorAgent._iterative_retrieval_loop`` end-to-end against:
  - a real DSPy LM (temperature=0) wired through ``dspy.configure``
  - a fake ``search_agent`` peer that returns the pre-populated KG-shaped
    snippets the loop is expected to retrieve at each iteration
  - a fake ``kg_traversal_agent`` peer that surfaces the seg_2 snippet from
    ``curie_sorbonne_60s`` when the gate reports a missing "work location"
  - an OpenTelemetry ``InMemorySpanExporter`` wired through a fake
    ``telemetry_manager`` so the span-level tests can assert on the
    emitted spans

All output that the LM produces (gate decisions, reformulated queries,
final answer text) is byte-compared against goldens in ``goldens/`` —
re-record once with ``RECORD_GOLDEN=1`` after a hand review when DSPy /
LM drift is intentional.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pytest

from cogniverse_agents.orchestrator_agent import (
    AccumulatedEvidence,
    AgentStep,
    OrchestrationPlan,
    OrchestratorAgent,
    OrchestratorDeps,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from tests.agents.integration.conftest import is_llm_available

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# File-level skip: the loop needs a reachable LM endpoint. Without it every
# gate decision short-circuits to the safe default and the goldens cannot
# be exercised meaningfully.
# ---------------------------------------------------------------------------

if not is_llm_available():
    pytest.skip(
        "Configured LLM endpoint not reachable — iterative retrieval loop "
        "requires a real DSPy-backed LM for gate decisions",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Golden file helpers
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"


def assert_golden_json(actual, name: str) -> None:
    path = GOLDEN_DIR / name
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, f"Golden mismatch for {name}"


def assert_golden_gate(actual, name: str) -> None:
    """Golden assertion for a sufficiency-gate output: ``sufficient`` and
    ``missing_aspects`` are byte-locked; ``confidence`` is an LM-sampled
    float compared as a tight band; the free-prose ``rationale`` re-words
    across identical runs at temperature 0.1, so the contract is that it
    GROUNDS the verdict — quoting the evidence fact — not its byte stream."""
    path = GOLDEN_DIR / name
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    expected = json.loads(path.read_text())
    got = json.loads(actual_json)

    exp_conf = float(expected.pop("confidence"))
    got_conf = float(got.pop("confidence"))
    assert abs(got_conf - exp_conf) <= 0.05, (
        f"{name}: confidence {got_conf} vs {exp_conf} (band 0.05)"
    )

    expected.pop("rationale")
    rationale = got.pop("rationale")
    assert 50 <= len(rationale) <= 1000, (
        f"{name}: rationale length {len(rationale)} outside bounds: {rationale[:200]!r}"
    )
    assert "Marie Curie discovered radium" in rationale, (
        f"{name}: rationale never quotes the evidence fact:\n{rationale}"
    )
    assert "1898" in rationale, (
        f"{name}: rationale never cites the year the question asks for:\n{rationale}"
    )

    assert got == expected, (
        f"Golden mismatch for {name} (structural fields).\n"
        f"--- expected ---\n{json.dumps(expected, indent=2)}\n"
        f"--- actual ---\n{json.dumps(got, indent=2)}"
    )


def assert_golden_text(actual: str, name: str) -> None:
    path = GOLDEN_DIR / name
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
        return
    expected = path.read_text()
    assert actual == expected, f"Golden mismatch for {name}"


def assert_golden_npy(actual_array: np.ndarray, name: str) -> None:
    path = GOLDEN_DIR / name
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, actual_array)
        return
    expected = np.load(path)
    assert np.array_equal(actual_array, expected), f"Golden mismatch for {name}"


# ---------------------------------------------------------------------------
# Pre-populated KG fixture shared across the loop tests
# ---------------------------------------------------------------------------


def _marie_curie_30s_seg3() -> Dict[str, Any]:
    return {
        "source_doc_id": "marie_curie_30s",
        "segment_id": "seg_3",
        "ts_start": 12.0,
        "ts_end": 18.5,
        "text": "Marie Curie discovered radium in 1898 at the Sorbonne.",
        "score": 0.94,
    }


def _marie_curie_30s_seg4() -> Dict[str, Any]:
    return {
        "source_doc_id": "marie_curie_30s",
        "segment_id": "seg_4",
        "ts_start": 18.5,
        "ts_end": 25.0,
        "text": "She later won the Nobel Prize in Physics.",
        "score": 0.88,
    }


def _curie_sorbonne_60s_seg2() -> Dict[str, Any]:
    return {
        "source_doc_id": "curie_sorbonne_60s",
        "segment_id": "seg_2",
        "ts_start": 10.0,
        "ts_end": 20.0,
        "text": "Marie Curie was a professor at the Sorbonne in Paris.",
        "score": 0.87,
    }


# ---------------------------------------------------------------------------
# In-memory OTEL exporter wired through a minimal fake telemetry_manager
# ---------------------------------------------------------------------------


class _FakeTelemetryManager:
    """Minimal stand-in for ``TelemetryManager`` that exposes ``.span()``.

    The orchestrator only calls ``telemetry_manager.span(...)`` from the
    retrieval-iteration emit path; everything else is best-effort. By
    pointing every tenant at the same ``InMemorySpanExporter`` we can
    inspect ``retrieval_iteration``, ``InstrumentedRLM.run``, and the
    ``KnowledgeGraphTraversalAgent.traverse`` child spans the span-level
    tests require.
    """

    def __init__(self, exporter, tracer):
        self.exporter = exporter
        self._tracer = tracer

    def span(self, name, tenant_id, project_name=None, attributes=None):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with self._tracer.start_as_current_span(name) as span:
                span.set_attribute("tenant.id", tenant_id)
                for k, v in (attributes or {}).items():
                    if v is not None:
                        span.set_attribute(k, v)
                yield span

        return _ctx()


@pytest.fixture
def captured_spans():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return _FakeTelemetryManager(exporter, tracer)


# ---------------------------------------------------------------------------
# Fake A2A peers — return the snippets the iter loop is supposed to find
# ---------------------------------------------------------------------------


class _IterRetrievalPeer:
    """Per-iteration scripted peer for the search_agent + kg_traversal_agent.

    The orchestrator runs each iteration through ``_execute_plan`` then
    ``_expand_via_kg_traversal``. We script per-call responses for both
    endpoints so the loop sees exactly the snippets the goldens expect.

    The ``preload`` constructor knob lets the RLM-promotion test prime
    evidence to >6000 chars so the gate promotes to ``InstrumentedRLM``.
    """

    def __init__(self, preload: Optional[List[Dict[str, Any]]] = None):
        self.preload = preload or []
        self.search_calls = 0
        self.kg_calls = 0

    def transport(self) -> httpx.MockTransport:
        def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "search" in url:
                self.search_calls += 1
                if self.search_calls == 1:
                    results = [_marie_curie_30s_seg3()] + self.preload
                elif self.search_calls == 2:
                    # Second iteration receives reformulated query that
                    # locates the work-location and year corroboration
                    results = [
                        _marie_curie_30s_seg3(),
                        _marie_curie_30s_seg4(),
                        _curie_sorbonne_60s_seg2(),
                    ]
                else:
                    results = []
                return httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "agent": "search_agent",
                        "results": results,
                    },
                )

            if "kg_traversal" in url or "traverse" in url:
                self.kg_calls += 1
                return httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "agent": "kg_traversal_agent",
                        "results": [_curie_sorbonne_60s_seg2()],
                        "nodes": [
                            {"name": "1898"},
                            {"name": "radium"},
                            {"name": "sorbonne"},
                        ],
                    },
                )

            return httpx.Response(404, json={"error": f"no route for {url}"})

        return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# Orchestrator fixture shared across the loop tests
# ---------------------------------------------------------------------------


def _build_orchestrator(
    *,
    telemetry_manager,
    peer: _IterRetrievalPeer,
) -> OrchestratorAgent:
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="iter_retrieval_test", config_manager=cm)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://search-peer.test:8002",
            capabilities=["search"],
            process_endpoint="/agents/search/process",
        )
    )
    registry.register_agent(
        AgentEndpoint(
            name="kg_traversal_agent",
            url="http://kg-traversal-peer.test:8014",
            capabilities=["kg_traversal"],
            process_endpoint="/agents/kg_traversal/process",
        )
    )

    http_client = httpx.AsyncClient(transport=peer.transport())
    orchestrator = OrchestratorAgent(
        deps=OrchestratorDeps(),
        registry=registry,
        config_manager=cm,
        port=8013,
        http_client=http_client,
    )
    # Inject the fake telemetry_manager so the loop emits spans into
    # the InMemorySpanExporter we control.
    orchestrator.telemetry_manager = telemetry_manager
    return orchestrator


def _make_search_plan(query: str) -> OrchestrationPlan:
    return OrchestrationPlan(
        query=query,
        steps=[
            AgentStep(
                agent_name="search_agent",
                input_data={"query": query},
                depends_on=[],
                reasoning="exercise the iterative retrieval loop",
            )
        ],
        parallel_groups=[[0]],
        reasoning="hand-built plan; bypasses DSPy planner",
    )


# ---------------------------------------------------------------------------
# Query string equality (literal)
# ---------------------------------------------------------------------------


CANONICAL_QUERY = "What did Marie Curie discover and where did she work in 1898?"


def test_query_string_literal():
    """The test fixture's query string is byte-equal to the canonical literal."""
    assert (
        CANONICAL_QUERY
        == "What did Marie Curie discover and where did she work in 1898?"
    )


# ---------------------------------------------------------------------------
# Shared test driver — runs the loop and returns AccumulatedEvidence
# ---------------------------------------------------------------------------


async def _run_loop(
    orchestrator: OrchestratorAgent,
    *,
    query: str = CANONICAL_QUERY,
) -> tuple[AccumulatedEvidence, Dict[str, Any]]:
    plan = _make_search_plan(query)
    agent_results: Dict[str, Any] = {}
    loop_result = await orchestrator._iterative_retrieval_loop(
        query=query,
        plan=plan,
        tenant_id="iter_retrieval_test",
        workflow_id="wf_iter_test",
        session_id=None,
        agent_results_sink=agent_results,
    )
    return loop_result, agent_results


# ---------------------------------------------------------------------------
# Iter 1 evidence byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter1_evidence_byte_equal_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    loop_result, _ = await _run_loop(orchestrator)

    # Pull the evidence the loop accumulated through iter 1 by re-running
    # the first iteration in isolation. _execute_plan + _extract +
    # _deduplicate are pure given the same plan / peer state.
    iter1_peer = _IterRetrievalPeer()
    iter1_orch = _build_orchestrator(telemetry_manager=captured_spans, peer=iter1_peer)
    plan = _make_search_plan(CANONICAL_QUERY)
    iter1_results = await iter1_orch._execute_plan(
        plan,
        workflow_id="wf_iter1_iso",
        tenant_id="iter_retrieval_test",
        session_id=None,
    )
    snippets = iter1_orch._extract_evidence_from_results(iter1_results)
    iter1_evidence = iter1_orch._deduplicate_evidence(snippets)

    sorted_evidence = sorted(
        iter1_evidence, key=lambda s: (s["source_doc_id"], s["segment_id"])
    )
    assert_golden_json(sorted_evidence, "iter_loop_trajectory_iter1.json")
    assert loop_result.iterations_executed >= 1


# ---------------------------------------------------------------------------
# Gate 1 output byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate1_output_byte_equal_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    evidence = [_marie_curie_30s_seg3()]
    gate_output = await orchestrator._run_sufficiency_gate(
        original_query=CANONICAL_QUERY,
        accumulated_evidence=evidence,
        iteration_idx=0,
    )
    assert_golden_json(gate_output, "iter_loop_gate1.json")


# ---------------------------------------------------------------------------
# Iter 2 evidence byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter2_evidence_byte_equal_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    loop_result, _ = await _run_loop(orchestrator)

    iter2_evidence = sorted(
        loop_result.evidence, key=lambda s: (s["source_doc_id"], s["segment_id"])
    )
    assert_golden_json(iter2_evidence, "iter_loop_trajectory_iter2.json")


# ---------------------------------------------------------------------------
# Gate 2 output byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate2_output_locked_against_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    loop_result, _ = await _run_loop(orchestrator)
    assert_golden_gate(loop_result.final_gate_output, "iter_loop_gate2.json")


# ---------------------------------------------------------------------------
# iterations_executed == 2, exit_reason == "sufficient"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_terminates_at_sufficient_after_two_iterations(
    captured_spans, dspy_lm
):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    loop_result, _ = await _run_loop(orchestrator)
    assert loop_result.iterations_executed == 2
    assert loop_result.exit_reason == "sufficient"


# ---------------------------------------------------------------------------
# Final answer text byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_answer_text_byte_equal_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    loop_result, agent_results = await _run_loop(orchestrator)
    aggregated = orchestrator._aggregate_results(CANONICAL_QUERY, agent_results)
    answer_text = aggregated.get("aggregated_result") or aggregated.get("answer") or ""
    assert_golden_text(answer_text, "iter_loop_answer.txt")
    # Sanity: loop emitted the joint-trace evidence the answer is grounded on
    assert len(loop_result.evidence) >= 1


# ---------------------------------------------------------------------------
# Phoenix span tree: retrieval_iteration spans with iteration_idx and
# sufficiency_score from goldens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_iteration_spans_match_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    await _run_loop(orchestrator)

    spans = [
        s
        for s in captured_spans.exporter.get_finished_spans()
        if s.name == "retrieval_iteration"
    ]
    spans.sort(key=lambda s: s.start_time)
    assert len(spans) == 2, [
        s.name for s in captured_spans.exporter.get_finished_spans()
    ]

    span_summary = [
        {
            "iteration_idx": int(s.attributes.get("iteration_idx", -1)),
            "sufficiency_score": float(s.attributes.get("sufficiency_score", 0.0)),
        }
        for s in spans
    ]
    assert [s["iteration_idx"] for s in span_summary] == [1, 2]
    assert_golden_json(span_summary, "iter_loop_spans_d8.json")


# ---------------------------------------------------------------------------
# Token budget breach: exit at iter 1, partial_due_to_budget == True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_budget_breach_exits_at_iter1(captured_spans, dspy_lm, monkeypatch):
    # Override the cap so the cumulative-token check fires after iter 1.
    monkeypatch.setattr(
        "cogniverse_agents.orchestrator_agent._ITER_RETRIEVAL_TOKEN_BUDGET",
        500,
    )
    # Force the gate to NOT report sufficient so the loop attempts iter 2,
    # at which point the post-iter cumulative-token check trips the cap.
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)

    async def _force_insufficient_gate(
        *, original_query, accumulated_evidence, iteration_idx
    ):
        return {
            "sufficient": False,
            "missing_aspects": ["work location"],
            "confidence": 0.42,
            "rationale": "forced-insufficient for budget breach test",
        }

    orchestrator._run_sufficiency_gate = _force_insufficient_gate  # type: ignore[method-assign]

    loop_result, agent_results = await _run_loop(orchestrator)
    assert loop_result.exit_reason == "token_budget"
    assert loop_result.partial_due_to_budget is True
    assert loop_result.iterations_executed == 1

    aggregated = orchestrator._aggregate_results(CANONICAL_QUERY, agent_results)
    answer_text = aggregated.get("aggregated_result") or aggregated.get("answer") or ""
    assert_golden_text(answer_text, "iter_loop_answer_budget_breach.txt")


# ---------------------------------------------------------------------------
# RLM promotion: pre-load evidence > 6000 chars; InstrumentedRLM.run
# child span emitted with attribute from golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rlm_promotion_emits_instrumented_rlm_child_span(
    captured_spans, dspy_lm, monkeypatch
):
    # Both gate iterations must run to completion regardless of LM latency:
    # on a contended LM the capped RLM sub-loops outlast the default wall
    # clock, iteration 2 is cut, and its span never exists. The subject
    # here is span emission on promotion — wall-clock behavior has its own
    # test below — so give the loop latency headroom.
    monkeypatch.setattr(
        "cogniverse_agents.orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS",
        30 * 60 * 1000,
    )
    # 100 copies of the seg_3 snippet pushes the JSON serialization above
    # _ITER_GATE_RLM_PROMOTION_CHARS (6000 chars).
    preload = [_marie_curie_30s_seg3() for _ in range(100)]
    peer = _IterRetrievalPeer(preload=preload)
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)
    await _run_loop(orchestrator)

    rlm_spans = [
        s
        for s in captured_spans.exporter.get_finished_spans()
        if s.name.startswith("InstrumentedRLM")
    ]
    assert rlm_spans, "expected at least one InstrumentedRLM.* child span"
    rlm_iterations = [
        s.attributes.get("rlm_iterations")
        for s in rlm_spans
        if s.attributes.get("rlm_iterations") is not None
    ]
    assert_golden_json(
        {"rlm_iterations": rlm_iterations},
        "iter_loop_rlm_promotion_d10.json",
    )


# ---------------------------------------------------------------------------
# Wall-clock cap: exit_reason == "wall_clock", duration_ms < 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wall_clock_cap_exits_promptly(captured_spans, dspy_lm, monkeypatch):
    monkeypatch.setattr(
        "cogniverse_agents.orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS",
        100,
    )
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)

    async def _force_insufficient_gate(
        *, original_query, accumulated_evidence, iteration_idx
    ):
        # Force a small sleep so the wall-clock check fires.
        await asyncio.sleep(0.15)
        return {
            "sufficient": False,
            "missing_aspects": ["work location"],
            "confidence": 0.30,
            "rationale": "forced-insufficient for wall-clock test",
        }

    orchestrator._run_sufficiency_gate = _force_insufficient_gate  # type: ignore[method-assign]

    started = time.monotonic()
    loop_result, _ = await _run_loop(orchestrator)
    duration_ms = (time.monotonic() - started) * 1000.0

    assert loop_result.exit_reason == "wall_clock"
    assert loop_result.partial_due_to_timeout is True
    assert duration_ms < 2000.0, (
        f"loop reported wall_clock but ran for {duration_ms:.0f}ms; "
        f"the 2000ms ceiling exists only to catch totally broken sleep behaviour"
    )


# ---------------------------------------------------------------------------
# KG expansion call: exactly one KnowledgeGraphTraversalAgent.traverse
# span with attributes byte-equal golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kg_traversal_span_attributes_match_golden(captured_spans, dspy_lm):
    peer = _IterRetrievalPeer()
    orchestrator = _build_orchestrator(telemetry_manager=captured_spans, peer=peer)

    # Force the loop to traverse the KG between iterations.
    real_gate = orchestrator._run_sufficiency_gate
    call_count = {"n": 0}

    async def _gate_then_sufficient(
        *, original_query, accumulated_evidence, iteration_idx
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "sufficient": False,
                "missing_aspects": ["work location"],
                "confidence": 0.42,
                "rationale": "forced-insufficient to trigger KG traversal",
            }
        return await real_gate(
            original_query=original_query,
            accumulated_evidence=accumulated_evidence,
            iteration_idx=iteration_idx,
        )

    orchestrator._run_sufficiency_gate = _gate_then_sufficient  # type: ignore[method-assign]

    await _run_loop(orchestrator)

    kg_spans = [
        s
        for s in captured_spans.exporter.get_finished_spans()
        if s.name == "KnowledgeGraphTraversalAgent.traverse"
    ]
    assert len(kg_spans) == 1, [
        s.name for s in captured_spans.exporter.get_finished_spans()
    ]
    span = kg_spans[0]
    # Production emits the real traversal: the seed subject (the missing
    # aspect that triggered traversal), the evidence time window, and the
    # node ids the peer returned.
    assert span.attributes.get("node_name") == "work location"
    assert sorted(json.loads(span.attributes.get("result_node_ids", "[]"))) == [
        "1898",
        "radium",
        "sorbonne",
    ]
    ts_start = float(span.attributes.get("filter_ts_start"))
    ts_end = float(span.attributes.get("filter_ts_end"))
    assert ts_start <= ts_end
