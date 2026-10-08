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
from cogniverse_agents.gateway_agent import (
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    _RoutingThresholds,
)
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    AutoAnnotation,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
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
from tests.utils.memory_store import InMemoryConfigStore
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
        "project": telemetry.config.get_project_name(tenant),
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
                # Two of three succeeded; with no false negatives recorded,
                # recall is 1.0 and F1 is 2 * (2/3) / (2/3 + 1).
                "precision": pytest.approx(2 / 3),
                "recall": 1.0,
                "f1": pytest.approx(0.8),
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
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
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
            started_ns=time.time_ns(),
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
        phoenix_proxy.intercept = lambda method, path, body: (400, {"detail": "bad"})
        refused = await client.get(_routing_path(tenant))

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

    did_not_answer = (
        ": the telemetry store did not answer (HTTPStatusError). It may be "
        "starting rather than misconfigured; refresh to try again."
    )
    assert [
        (r.status_code, r.json()["detail"].get("transient"))
        for r in (listed, read_back, refused)
    ] == [(502, True), (502, True), (502, False)]
    assert [
        (r.status_code, r.json()["detail"]["error"], r.json()["detail"]["message"])
        for r in (listed, read_back, refused, unwritten, unlabelled)
    ] == [
        (
            502,
            "telemetry_unavailable",
            f"Could not read the routing decisions of tenant {tenant}" + did_not_answer,
        ),
        (
            502,
            "telemetry_unavailable",
            f"Could not read routing decision {span_id} of tenant {tenant}"
            + did_not_answer,
        ),
        (
            502,
            "telemetry_unavailable",
            f"Could not read the routing decisions of tenant {tenant}: the "
            "telemetry store refused the query (RuntimeError). Check the "
            "telemetry configuration of this tenant.",
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


def test_only_failures_a_retry_can_clear_read_as_transient():
    refused = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("GET", "http://phoenix/v1/spans"),
        response=httpx.Response(400),
    )
    unavailable = httpx.HTTPStatusError(
        "unavailable",
        request=httpx.Request("GET", "http://phoenix/v1/spans"),
        response=httpx.Response(503),
    )

    def wrapped(cause):
        try:
            raise RuntimeError("Failed to query every span") from cause
        except RuntimeError as exc:
            return exc

    timeout = TimeoutError("read timed out")
    reset = ConnectionResetError("reset")
    transport = httpx.ConnectError("refused")
    assert [
        routing_decisions._transient_cause(exc)
        for exc in (
            wrapped(unavailable),
            wrapped(timeout),
            reset,
            wrapped(transport),
            wrapped(refused),
            ValueError("no such project"),
        )
    ] == [unavailable, timeout, reset, transport, None, None]


class _SlowEntityModel:
    """Stands in for the GLiNER service: answers after ``seconds``."""

    def __init__(self, seconds):
        self.seconds = seconds

    def predict_entities(self, query, labels, threshold):
        time.sleep(self.seconds)
        return [{"text": "video", "label": "video_content", "score": 0.92}]


async def test_a_gateway_decision_lasts_as_long_as_its_classification(telemetry, app):
    tenant = _tenant("decisionlength")
    store = InMemoryConfigStore()
    store.initialize()
    gateway = GatewayAgent(deps=GatewayDeps())
    gateway.bind_config_manager(ConfigManager(store=store))
    gateway._gliner_model = _SlowEntityModel(0.3)
    gateway.telemetry_manager = telemetry
    started = time.monotonic()
    decided = await gateway._process_impl(
        GatewayInput(query="cooking videos", tenant_id=tenant)
    )
    took_ms = (time.monotonic() - started) * 1000
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _routing_path(tenant), _routed(1))

    [decision] = body["decisions"]
    assert (decision["query"], decision["chosen_agent"]) == (
        "cooking videos",
        decided.routed_to,
    )
    # The span covers the classification, not only the write of its record.
    assert 300 <= decision["latency_ms"] <= took_ms
    assert body["latency_ms"]["mean"] == decision["latency_ms"]


def _candidates_path(tenant, **params):
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"/admin/tenant/{tenant}/routing-decisions/annotation-candidates?{query}"


async def _decisions_of(client, telemetry, tenant, recorded):
    """Record ``recorded`` (agent, confidence, extra) decisions a minute
    apart, oldest first; the list's decisions by span ID once it has them."""
    span_ids = [
        record_routing(
            telemetry, tenant, agent, confidence, 50, minutes_ago=age, **extra
        )
        for age, (agent, confidence, extra) in zip(
            range(len(recorded), 0, -1), recorded, strict=True
        )
    ]
    telemetry.force_flush(timeout_millis=10000)
    body = await _until(client, _routing_path(tenant), _routed(len(recorded)))
    by_span = {d["span_id"]: d for d in body["decisions"]}
    return [by_span[span_id] for span_id in span_ids]


def _candidate(decision, priority, reason):
    return {
        "span_id": decision["span_id"],
        "start_time": decision["start_time"],
        "query": decision["query"],
        "chosen_agent": decision["chosen_agent"],
        "confidence": decision["confidence"],
        "outcome": decision["outcome"],
        "priority": priority,
        "reason": reason,
    }


async def test_annotation_candidates_follow_the_threshold_and_the_cap(telemetry, app):
    tenant = _tenant("candidates")
    async with _client(app) as client:
        failed, unsure, boundary, confident, unparented = await _decisions_of(
            client,
            telemetry,
            tenant,
            [
                ("search_agent", 0.3, {"failed": True}),
                ("search_agent", 0.5, {}),
                ("summarizer_agent", 0.7, {}),
                ("search_agent", 0.9, {}),
                ("summarizer_agent", 0.8, {"within_request": False}),
            ],
        )
        at_default = await client.get(_candidates_path(tenant, lookback_hours=1))
        at_low = await client.get(
            _candidates_path(tenant, lookback_hours=1, confidence_threshold=0.4)
        )
        capped = await client.get(
            _candidates_path(tenant, lookback_hours=1, max_annotations=2)
        )
        out_of_range = await client.get(
            _candidates_path(tenant, confidence_threshold=1.5, max_annotations=0)
        )

    high = _candidate(failed, "high", "Failure with low confidence (0.30)")
    verify = _candidate(
        unsure,
        "medium",
        "Success but low confidence (0.50) - verify correctness",
    )
    ambiguous = _candidate(
        unparented, "medium", "Ambiguous outcome - unclear if routing was correct"
    )
    diverse = _candidate(
        boundary,
        "low",
        "Near decision boundary (0.70) - training data diversity",
    )
    assert at_default.status_code == 200, at_default.text
    # Highest priority first, then the older decision within a priority; the
    # confident success needs no review.
    assert at_default.json() == {"candidates": [high, verify, ambiguous, diverse]}
    assert confident["span_id"] not in {
        c["span_id"] for c in at_default.json()["candidates"]
    }
    # At 0.4 the 0.30 failure is still below the threshold, and the 0.50
    # success is no longer low-confidence.
    assert at_low.json() == {"candidates": [high, ambiguous, diverse]}
    assert capped.json() == {"candidates": [high, verify]}
    assert out_of_range.status_code == 422
    assert sorted(error["loc"] for error in out_of_range.json()["detail"]) == [
        ["query", "confidence_threshold"],
        ["query", "max_annotations"],
    ]


async def test_label_statistics_count_the_stored_labels(telemetry, app):
    tenant = _tenant("labelstats")
    statistics_path = f"/admin/tenant/{tenant}/routing-decisions/label-statistics"
    async with _client(app) as client:
        empty = await client.get(statistics_path)
        llm, reviewed, _ = await _decisions_of(
            client,
            telemetry,
            tenant,
            [
                ("search_agent", 0.4, {}),
                ("search_agent", 0.5, {}),
                ("search_agent", 0.9, {}),
            ],
        )
        _store_llm_label(tenant, llm["span_id"])
        response = await client.put(
            _decision_path(tenant, reviewed, "label"),
            json={
                "start_time": reviewed["start_time"],
                "reviewer": "dana",
                "label": "correct",
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 90
        while True:
            counted = await client.get(statistics_path)
            assert counted.status_code == 200, counted.text
            if counted.json()["total"] == 2 or time.monotonic() > deadline:
                break
            await asyncio.sleep(2)

    assert empty.json() == {
        "total": 0,
        "human_reviewed": 0,
        "pending_review": 0,
        "by_label": {},
    }
    assert counted.json() == {
        "total": 2,
        "human_reviewed": 1,
        "pending_review": 1,
        "by_label": {"wrong_routing": 1, "correct": 1},
    }


async def test_concurrent_candidate_searches_keep_their_own_tenant_and_threshold(
    telemetry, app, phoenix_proxy
):
    tenants = [_tenant("candidatesa"), _tenant("candidatesb")]
    async with _client(app) as client:
        [low_a] = await _decisions_of(
            client, telemetry, tenants[0], [("search_agent", 0.5, {})]
        )
        [low_b] = await _decisions_of(
            client, telemetry, tenants[1], [("summarizer_agent", 0.55, {})]
        )
        searches = [
            (tenants[0], 0.6),
            (tenants[0], 0.4),
            (tenants[1], 0.6),
            (tenants[1], 0.4),
        ]
        barrier = threading.Barrier(len(searches), timeout=60)
        held = []

        def hold_span_reads(method, path, body):
            # Every search reaches Phoenix before any is answered.
            if "/spans" in path and len(held) < len(searches):
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_span_reads
        responses = await asyncio.gather(
            *(
                client.get(
                    _candidates_path(
                        tenant, lookback_hours=1, confidence_threshold=threshold
                    )
                )
                for tenant, threshold in searches
            )
        )
        phoenix_proxy.intercept = None

    assert len(held) == len(searches)
    assert [r.json() for r in responses] == [
        {
            "candidates": [
                _candidate(
                    low_a,
                    "medium",
                    "Success but low confidence (0.50) - verify correctness",
                )
            ]
        },
        {"candidates": []},
        {
            "candidates": [
                _candidate(
                    low_b,
                    "medium",
                    "Success but low confidence (0.55) - verify correctness",
                )
            ]
        },
        {"candidates": []},
    ]


async def test_candidates_and_label_statistics_report_an_outage(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant("candidatesoutage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    async with _client(app) as client:
        candidates = await client.get(_candidates_path(tenant, lookback_hours=1))
        statistics_read = await client.get(
            f"/admin/tenant/{tenant}/routing-decisions/label-statistics"
        )
    phoenix_proxy.intercept = None
    assert [
        (r.status_code, r.json()["detail"]["message"], r.json()["detail"]["transient"])
        for r in (candidates, statistics_read)
    ] == [
        (
            502,
            f"Could not read the routing decisions needing annotation of tenant "
            f"{tenant}: the telemetry store did not answer (HTTPStatusError). It "
            "may be starting rather than misconfigured; refresh to try again.",
            True,
        ),
        (
            502,
            f"Could not read the stored routing labels of tenant {tenant}: the "
            "telemetry store did not answer (HTTPStatusError). It may be starting "
            "rather than misconfigured; refresh to try again.",
            True,
        ),
    ]
