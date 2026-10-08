"""The routing decision routes against real Phoenix.

Decisions are recorded with the output slot ``GatewayAgent._emit_routing_span``
writes (or through it), LLM labels by ``AnnotationStorage.store_llm_annotation``.
The runtime reads Phoenix through a forwarding proxy, so a test can hold or
fail its reads and writes.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.gateway_agent import GatewayAgent, _RoutingThresholds
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    AutoAnnotation,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ROUTING,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.routers import routing_decisions
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import record_routing

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    """The global telemetry manager: spans export to Phoenix, reads and
    annotation writes go through ``phoenix_proxy``."""
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_proxy.url,
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture()
def app(phoenix_proxy):
    app = FastAPI()
    app.include_router(routing_decisions.router, prefix="/admin/tenant")
    yield app
    phoenix_proxy.intercept = None


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=300,
    )


def _tenant(prefix="routing"):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


async def _until(client, path, predicate, timeout=90.0):
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(path)
        assert response.status_code == 200, response.text
        body = response.json()
        if predicate(body) or time.monotonic() > deadline:
            return body
        await asyncio.sleep(2)


def _routing_path(tenant):
    return f"/admin/tenant/{tenant}/routing-decisions?lookback_hours=1"


def _routed(expected):
    return lambda body: body["total"] + body["unreadable"] >= expected


async def test_routing_decisions_are_evaluated_per_agent(telemetry, app):
    tenant = _tenant("routing")
    recorded = [
        ("search_agent", 0.9, 100, {}),
        ("search_agent", 0.8, 200, {}),
        ("search_agent", 0.3, 300, {"failed": True}),
        ("summarizer_agent", 0.6, 400, {"within_request": False}),
    ]
    span_ids = [
        record_routing(
            telemetry, tenant, agent, confidence, duration, minutes_ago=age, **kwargs
        )
        for age, (agent, confidence, duration, kwargs) in enumerate(recorded, start=1)
    ]
    # A routing span that records no confidence cannot be evaluated.
    tracer = telemetry._get_tracer_for_project(tenant, None)
    with tracer.start_as_current_span(SPAN_NAME_ROUTING) as span:
        span.set_attribute("output.value", '{"chosen_agent": "search_agent"}')
    record_routing(
        telemetry, _tenant("otherrouting"), "search_agent", 0.9, 5, minutes_ago=1
    )
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _routing_path(tenant), _routed(5))

    outcomes = [
        ("success", "completed_successfully"),
        ("success", "completed_successfully"),
        ("failure", "routing_error"),
        ("ambiguous", "no_parent_span"),
    ]
    assert [
        {key: d[key] for key in d if key not in ("trace_id", "start_time")}
        for d in body["decisions"]
    ] == [
        {
            "span_id": span_id,
            "query": f"a query for {agent}",
            "chosen_agent": agent,
            "confidence": confidence,
            "outcome": outcome,
            "reason": reason,
            "latency_ms": pytest.approx(float(duration), abs=1e-6),
            "entity_extraction_failed": False,
            "label": None,
        }
        for span_id, (agent, confidence, duration, _), (outcome, reason) in zip(
            span_ids, recorded, outcomes, strict=True
        )
    ]
    assert {k: body[k] for k in body if k != "decisions"} == {
        "total": 4,
        "successes": 2,
        "failures": 1,
        "ambiguous": 1,
        "unreadable": 1,
        "accuracy": 0.5,
        "confidence_calibration": pytest.approx(
            statistics.correlation([0.9, 0.8, 0.3, 0.6], [1.0, 1.0, 0.0, 0.0])
        ),
        "latency_ms": {
            "mean": pytest.approx(250.0),
            "p50": pytest.approx(250.0),
            "p95": pytest.approx(385.0),
        },
        "per_agent": [
            {
                "agent": "search_agent",
                "decisions": 3,
                "successes": 2,
                "failures": 1,
                "ambiguous": 0,
                "success_rate": pytest.approx(2 / 3),
                "mean_confidence": pytest.approx(2 / 3),
                "mean_latency_ms": pytest.approx(200.0),
            },
            {
                "agent": "summarizer_agent",
                "decisions": 1,
                "successes": 0,
                "failures": 0,
                "ambiguous": 1,
                "success_rate": 0.0,
                "mean_confidence": 0.6,
                "mean_latency_ms": pytest.approx(400.0),
            },
        ],
    }


async def test_a_decision_recorded_by_the_gateway_is_evaluated(telemetry, app):
    tenant = _tenant("gatewayrouting")
    tracer = telemetry._get_tracer_for_project(tenant, None)
    with tracer.start_as_current_span("a2a.request"):
        GatewayAgent._emit_routing_span(
            SimpleNamespace(telemetry_manager=telemetry),
            tenant_id=tenant,
            query="summarize the keynote",
            complexity="simple",
            modality="video",
            generation_type="summary",
            routed_to="summarizer_agent",
            confidence=0.7,
            reasoning="a summary request",
            thresholds=_RoutingThresholds(0.8, 0.3),
            entity_extraction_failed=True,
        )
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _routing_path(tenant), _routed(1))

    [decision] = body["decisions"]
    assert {
        key: decision[key]
        for key in (
            "query",
            "chosen_agent",
            "confidence",
            "outcome",
            "reason",
            "entity_extraction_failed",
        )
    } == {
        "query": "summarize the keynote",
        "chosen_agent": "summarizer_agent",
        "confidence": 0.7,
        "outcome": "success",
        "reason": "completed_successfully",
        "entity_extraction_failed": True,
    }
    # One decision has no defined calibration.
    assert (
        body["total"],
        body["successes"],
        body["unreadable"],
        body["confidence_calibration"],
    ) == (1, 1, 0, None)


LLM_LABEL = AutoAnnotation(
    span_id="",
    label=AnnotationLabel.WRONG_ROUTING,
    confidence=0.8,
    reasoning="a summary request belongs to the summarizer",
    suggested_correct_agent="summarizer_agent",
    requires_human_review=True,
)


def _store_llm_label(tenant, span_id):
    run_in_own_loop(
        AnnotationStorage(tenant_id=tenant).store_llm_annotation(span_id, LLM_LABEL)
    )


def _labelled(span_id, predicate):
    def check(body):
        return any(
            d["span_id"] == span_id and d["label"] and predicate(d["label"])
            for d in body["decisions"]
        )

    return check


def _decision(body, span_id):
    [decision] = [d for d in body["decisions"] if d["span_id"] == span_id]
    return decision


async def _recorded_decision(client, telemetry, tenant, *, llm_label=True):
    """One routing decision of ``tenant`` as the list serves it, labelled
    by the LLM unless ``llm_label`` is false."""
    span_id = record_routing(telemetry, tenant, "search_agent", 0.4, 50, minutes_ago=1)
    telemetry.force_flush(timeout_millis=10000)
    if not llm_label:
        body = await _until(client, _routing_path(tenant), _routed(1))
        return _decision(body, span_id)
    _store_llm_label(tenant, span_id)
    body = await _until(
        client, _routing_path(tenant), _labelled(span_id, lambda label: True)
    )
    return _decision(body, span_id)


def _decision_path(tenant, decision, action):
    return f"/admin/tenant/{tenant}/routing-decisions/{decision['span_id']}/{action}"


LLM_LABEL_FIELDS = {
    "label": "wrong_routing",
    "confidence": 0.8,
    "reasoning": "a summary request belongs to the summarizer",
    "suggested_agent": "summarizer_agent",
    "annotator": "llm",
}


async def test_approving_an_llm_label_keeps_it(telemetry, app):
    tenant = _tenant("approve")
    async with _client(app) as client:
        decision = await _recorded_decision(client, telemetry, tenant)
        assert decision["label"] == {
            **LLM_LABEL_FIELDS,
            "human_reviewed": False,
            "requires_review": True,
            "approved_by": None,
        }
        response = await client.post(
            _decision_path(tenant, decision, "approve"),
            json={"start_time": decision["start_time"], "reviewer": "dana"},
        )
        approved = {
            **LLM_LABEL_FIELDS,
            "human_reviewed": True,
            "requires_review": False,
            "approved_by": "dana",
        }
        assert response.status_code == 200, response.text
        assert response.json() == {**decision, "label": approved}
        body = await _until(
            client,
            _routing_path(tenant),
            _labelled(decision["span_id"], lambda label: label["human_reviewed"]),
        )
    assert _decision(body, decision["span_id"])["label"] == approved


async def test_a_reviewer_label_replaces_the_llm_label(telemetry, app):
    tenant = _tenant("relabel")
    async with _client(app) as client:
        decision = await _recorded_decision(client, telemetry, tenant)
        response = await client.put(
            _decision_path(tenant, decision, "label"),
            json={
                "start_time": decision["start_time"],
                "reviewer": "dana",
                "label": "correct",
                "reasoning": "search was right",
            },
        )
        relabelled = {
            "label": "correct",
            "confidence": 1.0,
            "reasoning": "search was right",
            "suggested_agent": None,
            "annotator": "dana",
            "human_reviewed": True,
            "requires_review": False,
            "approved_by": None,
        }
        assert response.status_code == 200, response.text
        assert response.json() == {**decision, "label": relabelled}
        body = await _until(
            client,
            _routing_path(tenant),
            _labelled(decision["span_id"], lambda label: label["annotator"] == "dana"),
        )
        # A person's label is not the LLM's to approve.
        refused = await client.post(
            _decision_path(tenant, decision, "approve"),
            json={"start_time": decision["start_time"], "reviewer": "erin"},
        )
    assert _decision(body, decision["span_id"])["label"] == relabelled
    assert (refused.status_code, refused.json()["detail"]) == (
        409,
        f"Routing decision {decision['span_id']} is labelled by dana, not the LLM.",
    )


async def test_review_refuses_what_it_cannot_apply(telemetry, app):
    tenant = _tenant("refuse")
    async with _client(app) as client:
        decision = await _recorded_decision(client, telemetry, tenant, llm_label=False)
        unlabelled = await client.post(
            _decision_path(tenant, decision, "approve"),
            json={"start_time": decision["start_time"], "reviewer": "dana"},
        )
        other = _tenant("refuseother")
        elsewhere = await client.put(
            _decision_path(other, decision, "label"),
            json={
                "start_time": decision["start_time"],
                "reviewer": "dana",
                "label": "wrong",
            },
        )
        unknown_label = await client.put(
            _decision_path(tenant, decision, "label"),
            json={
                "start_time": decision["start_time"],
                "reviewer": "dana",
                "label": "wrong_routing",
            },
        )
        naive = await client.put(
            _decision_path(tenant, decision, "label"),
            json={
                "start_time": decision["start_time"].replace("+00:00", ""),
                "reviewer": "dana",
                "label": "wrong",
            },
        )
    assert (unlabelled.status_code, unlabelled.json()["detail"]) == (
        404,
        f"Routing decision {decision['span_id']} has no LLM label to approve.",
    )
    started = datetime.fromisoformat(decision["start_time"]).isoformat()
    assert (elsewhere.status_code, elsewhere.json()["detail"]) == (
        404,
        f"No routing decision {decision['span_id']} started at {started} "
        f"for tenant {other}.",
    )
    assert unknown_label.status_code == 422
    assert [error["loc"] for error in unknown_label.json()["detail"]] == [
        ["body", "label"]
    ]
    assert (naive.status_code, naive.json()["detail"]) == (
        422,
        "start_time must include a timezone.",
    )


async def test_concurrent_approvals_and_reads_hold_their_own_state(
    telemetry, app, phoenix_proxy
):
    tenants = [_tenant("concurrentrouting"), _tenant("concurrentrouting")]
    async with _client(app) as client:
        decision = await _recorded_decision(client, telemetry, tenants[0])
        record_routing(telemetry, tenants[1], "summarizer_agent", 0.9, 5, minutes_ago=2)
        record_routing(telemetry, tenants[1], "summarizer_agent", 0.9, 5, minutes_ago=3)
        telemetry.force_flush(timeout_millis=10000)
        await _until(client, _routing_path(tenants[1]), _routed(2))
        approvals = 4
        reads = approvals + 2
        barrier = threading.Barrier(reads, timeout=60)
        held = []

        def hold_span_reads(method, path, body):
            # Every request reaches Phoenix before any is answered.
            if "/spans" in path and len(held) < reads:
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_span_reads
        responses = await asyncio.gather(
            *(
                client.post(
                    _decision_path(tenants[0], decision, "approve"),
                    json={"start_time": decision["start_time"], "reviewer": f"r{i}"},
                )
                for i in range(approvals)
            ),
            *(client.get(_routing_path(tenant)) for tenant in tenants),
        )
        phoenix_proxy.intercept = None
        body = await _until(
            client,
            _routing_path(tenants[0]),
            _labelled(decision["span_id"], lambda label: label["human_reviewed"]),
        )

    assert len(held) == reads
    assert [
        (r.status_code, r.json()["label"]["label"], r.json()["label"]["approved_by"])
        for r in responses[:approvals]
    ] == [(200, "wrong_routing", f"r{i}") for i in range(approvals)]
    assert [
        (r.json()["total"], {d["chosen_agent"] for d in r.json()["decisions"]})
        for r in responses[approvals:]
    ] == [(1, {"search_agent"}), (2, {"summarizer_agent"})]
    final = _decision(body, decision["span_id"])["label"]
    assert ({k: final[k] for k in LLM_LABEL_FIELDS}, final["human_reviewed"]) == (
        LLM_LABEL_FIELDS,
        True,
    )
    assert final["approved_by"] in {f"r{i}" for i in range(approvals)}


async def test_an_unreadable_or_unwritable_telemetry_backend_answers_502(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant("routingoutage")
    async with _client(app) as client:
        decision = await _recorded_decision(client, telemetry, tenant)
        span_id = decision["span_id"]
        approve = {"start_time": decision["start_time"], "reviewer": "dana"}

        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        listed = await client.get(_routing_path(tenant))
        read_back = await client.post(
            _decision_path(tenant, decision, "approve"), json=approve
        )

        def fail_annotation_writes(method, path, body):
            if method == "POST" and "span_annotations" in path:
                return (503, {"detail": "down"})
            return None

        phoenix_proxy.intercept = fail_annotation_writes
        unwritten = await client.post(
            _decision_path(tenant, decision, "approve"), json=approve
        )
        unlabelled = await client.put(
            _decision_path(tenant, decision, "label"),
            json={**approve, "label": "wrong"},
        )
        phoenix_proxy.intercept = None
        body = await _until(client, _routing_path(tenant), _routed(1))

    assert [
        (r.status_code, r.json()["detail"]["error"], r.json()["detail"]["message"])
        for r in (listed, read_back, unwritten, unlabelled)
    ] == [
        (
            502,
            "telemetry_unavailable",
            f"Could not read the routing decisions of tenant {tenant}.",
        ),
        (
            502,
            "telemetry_unavailable",
            f"Could not read routing decision {span_id} of tenant {tenant}.",
        ),
        (
            502,
            "annotation_not_stored",
            f"The label of routing decision {span_id} was not stored.",
        ),
        (
            502,
            "annotation_not_stored",
            f"The label of routing decision {span_id} was not stored.",
        ),
    ]
    # The failed writes left the LLM's label as it was.
    assert _decision(body, span_id)["label"] == decision["label"]
