"""The web client's Annotation queue view, driven in Chromium against the
runtime's annotation queue on real Redis and real Phoenix.

Each test emits real routing spans for a fresh tenant, enqueues review
requests for them through the runtime's enqueue route, takes every review
action through the page, and reads the outcome back: the queue the route
serves and the human annotations Phoenix stores on the spans.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
import redis
import redis.asyncio as aioredis
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import REVIEW_LABELS
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_foundation.telemetry.span_contract import record_span_io
from cogniverse_runtime.routers import agents
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

QUEUE_PREFIX = f"web-ops-test:annotation-view:{uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def runtime_url(phoenix_container, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        review_config_manager(phoenix_container, workflow_state_redis_url),
        schema_loader,
        workflow_state_redis_url,
        annotation_queue_prefix=QUEUE_PREFIX,
    ) as url:
        yield url


@pytest.fixture()
def web_url(built_client, runtime_url):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture()
def queue_cleared(workflow_state_redis_url, runtime_url):
    """An empty annotation queue before and after the test."""

    def clear():
        client = redis.Redis.from_url(workflow_state_redis_url)
        try:
            keys = list(client.scan_iter(f"{QUEUE_PREFIX}*"))
            if keys:
                client.delete(*keys)
        finally:
            client.close()

    clear()
    yield
    clear()


@pytest.fixture()
def tenant_spans(real_telemetry, queue_cleared):
    """A fresh tenant and a factory emitting one of its routing spans,
    returning the span ID once Phoenix serves it."""
    tenant = f"webann{uuid4().hex[:8]}:main"
    storage = AnnotationStorage(tenant_id=tenant)

    def emit(query: str, chosen_agent: str, confidence: float) -> str:
        with real_telemetry.span(name="cogniverse.routing", tenant_id=tenant) as span:
            record_span_io(
                span,
                input_value=query,
                output={"chosen_agent": chosen_agent, "confidence": confidence},
                operation="routing",
            )
            span_id = format(span.get_span_context().span_id, "016x")
        real_telemetry.force_flush(timeout_millis=10000)
        return span_id

    return tenant, storage, emit


def _request(span_id, tenant, query, agent, confidence, priority, reason):
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime.now(timezone.utc),
        query=query,
        chosen_agent=agent,
        routing_confidence=confidence,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason=reason,
        context={},
        tenant_id=tenant,
    )


def _enqueue(runtime_url, requests):
    response = httpx.post(
        f"{runtime_url}/agents/annotations/queue/enqueue",
        json={"requests": [request.to_dict() for request in requests]},
        timeout=30,
    )
    assert (response.status_code, response.json()["enqueued"]) == (
        200,
        len(requests),
    )


def _served(runtime_url):
    response = httpx.get(f"{runtime_url}/agents/annotations/queue", timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


def _labels_until(storage, span_ids, timeout=60.0):
    """The human annotations Phoenix serves for ``span_ids`` once each has
    one, as ``{span_id: (label, reasoning)}``."""
    deadline = time.monotonic() + timeout
    while True:
        end = datetime.now(timezone.utc)
        rows = run_in_own_loop(
            storage.query_annotated_spans(
                start_time=end - timedelta(hours=1),
                end_time=end,
                only_human_reviewed=True,
            )
        )
        found = {
            row["span_id"]: (row["annotation_label"], row["annotation_reasoning"])
            for row in rows
        }
        if set(found) == set(span_ids) or time.monotonic() > deadline:
            return found
        time.sleep(2)


def _open(page: Page, web_url: str):
    page.goto(f"{web_url}/#/ops/annotations")
    expect(
        page.get_by_role("heading", name="Annotation queue", level=1)
    ).to_be_visible()


def _summary(page: Page):
    return page.get_by_role("region", name="Annotation queue").locator("p.muted")


def _cells(page: Page, title: str):
    return page.get_by_role("table", name=f"{title} requests").locator("tbody td")


def _row(request, tenant, assignee=None, due=None, actions=None):
    cells = [
        request.span_id,
        tenant,
        request.query,
        request.chosen_agent,
        f"{request.routing_confidence:.2f}",
        "ambiguous",
        request.priority.value,
        request.reason,
    ]
    if assignee is not None:
        cells += [assignee, due]
    if actions is not None:
        cells.append(actions)
    return cells


class TestAnnotationsView:
    def test_a_reviewer_assigns_and_labels_requests_through_the_page(
        self, page, web_url, runtime_url, tenant_spans
    ):
        tenant, storage, emit = tenant_spans
        high = _request(
            emit("play something relaxing", "search_agent", 0.2),
            tenant,
            "play something relaxing",
            "search_agent",
            0.2,
            AnnotationPriority.HIGH,
            "very low routing confidence",
        )
        low = _request(
            emit("summarize the lecture", "summarizer_agent", 0.6),
            tenant,
            "summarize the lecture",
            "summarizer_agent",
            0.6,
            AnnotationPriority.LOW,
            "edge case for training diversity",
        )
        overdue = _request(
            emit("find the intro clip", "search_agent", 0.4),
            tenant,
            "find the intro clip",
            "search_agent",
            0.4,
            AnnotationPriority.MEDIUM,
            "ambiguous outcome",
        )
        # Enqueued low priority first: the queue serves by priority.
        _enqueue(runtime_url, [low, high, overdue])
        assigned = httpx.post(
            f"{runtime_url}/agents/annotations/queue/{overdue.span_id}/assign",
            json={"reviewer": "away@example.com", "sla_hours": 0},
            timeout=30,
        )
        assert assigned.status_code == 200, assigned.text
        due = assigned.json()["annotation"]["sla_deadline"]

        _open(page, web_url)
        expect(_summary(page)).to_have_text(
            "2 pending, 0 assigned, 1 expired, 0 completed."
        )
        expect(_cells(page, "Pending")).to_have_text(
            _row(high, tenant, actions="Assign to meAnnotate")
            + _row(low, tenant, actions="Assign to meAnnotate")
        )
        expect(_cells(page, "Expired")).to_have_text(
            _row(overdue, tenant, "away@example.com", due)
        )
        expect(
            page.get_by_role("region", name="Assigned").get_by_text(
                "No assigned requests."
            )
        ).to_be_visible()

        pending = page.get_by_role("region", name="Pending")
        pending.get_by_role("button", name=f"Assign {high.span_id} to me").click()
        expect(pending.get_by_role("alert")).to_have_text(
            "Enter your name as the reviewer first."
        )
        page.get_by_label("Reviewer").fill("reviewer@example.com")
        pending.get_by_role("button", name=f"Assign {high.span_id} to me").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Assigned {high.span_id} to reviewer@example.com."
        )
        served = _served(runtime_url)
        assert [
            (request["span_id"], request["assigned_to"])
            for request in served["assigned"]
        ] == [(high.span_id, "reviewer@example.com")]
        expect(_summary(page)).to_have_text(
            "1 pending, 1 assigned, 1 expired, 0 completed."
        )
        expect(_cells(page, "Assigned")).to_have_text(
            _row(
                high,
                tenant,
                "reviewer@example.com",
                served["assigned"][0]["sla_deadline"],
                actions="Annotate",
            )
        )

        assigned_list = page.get_by_role("region", name="Assigned")
        assigned_list.get_by_role("button", name=f"Annotate {high.span_id}").click()
        form = page.get_by_role("form", name=f"Annotate {high.span_id}")
        expect(form.get_by_label("Label").locator("option")).to_have_text(
            ["Choose a label", *(label.value for label in REVIEW_LABELS)]
        )
        form.get_by_label("Label").select_option("wrong")
        form.get_by_label("Reasoning").fill("should have gone to the music agent")
        form.get_by_role("button", name="Save label").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Labelled {high.span_id} wrong."
        )
        expect(_summary(page)).to_have_text(
            "1 pending, 0 assigned, 1 expired, 1 completed."
        )
        expect(_cells(page, "Pending")).to_have_text(
            _row(low, tenant, actions="Assign to meAnnotate")
        )

        pending = page.get_by_role("region", name="Pending")
        pending.get_by_role("button", name=f"Annotate {low.span_id}").click()
        form = page.get_by_role("form", name=f"Annotate {low.span_id}")
        form.get_by_label("Label").select_option("correct")
        form.get_by_role("button", name="Save label").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Labelled {low.span_id} correct."
        )
        expect(_summary(page)).to_have_text(
            "0 pending, 0 assigned, 1 expired, 2 completed."
        )

        assert _labels_until(storage, {high.span_id, low.span_id}) == {
            high.span_id: ("wrong", "should have gone to the music agent"),
            low.span_id: ("correct", ""),
        }


class TestConcurrency:
    def test_a_request_another_reviewer_labelled_first_is_refused_not_relabelled(
        self, browser, web_url, runtime_url, tenant_spans
    ):
        tenant, storage, emit = tenant_spans
        request = _request(
            emit("play the keynote", "search_agent", 0.3),
            tenant,
            "play the keynote",
            "search_agent",
            0.3,
            AnnotationPriority.HIGH,
            "low routing confidence",
        )
        _enqueue(runtime_url, [request])
        contexts = [browser.new_context() for _ in range(2)]
        first, second = (context.new_page() for context in contexts)
        try:
            for page, reviewer in (
                (first, "first@example.com"),
                (second, "second@example.com"),
            ):
                _open(page, web_url)
                page.get_by_label("Reviewer").fill(reviewer)
                page.get_by_role("region", name="Pending").get_by_role(
                    "button", name=f"Annotate {request.span_id}"
                ).click()
            for page, label in ((first, "correct"), (second, "wrong")):
                page.get_by_role(
                    "form", name=f"Annotate {request.span_id}"
                ).get_by_label("Label").select_option(label)
            first.get_by_role("form", name=f"Annotate {request.span_id}").get_by_role(
                "button", name="Save label"
            ).click()
            expect(first.get_by_role("status")).to_have_text(
                f"Labelled {request.span_id} correct."
            )
            # The second page still shows the request the first just labelled.
            form = second.get_by_role("form", name=f"Annotate {request.span_id}")
            form.get_by_role("button", name="Save label").click()
            expect(form.get_by_role("alert")).to_have_text(
                f"Cannot complete span {request.span_id}: status is completed"
            )
        finally:
            for context in contexts:
                context.close()
        assert _served(runtime_url)["statistics"]["by_status"] == {"completed": 1}
        assert _labels_until(storage, {request.span_id}) == {
            request.span_id: ("correct", "")
        }


class TestFaults:
    def test_an_unreachable_queue_shows_an_outage_not_an_empty_queue(
        self, page, web_url, queue_cleared
    ):
        working = agents._annotation_queue
        agents.set_annotation_queue(
            AnnotationQueue(
                aioredis.from_url(
                    "redis://127.0.0.1:9", socket_connect_timeout=1, socket_timeout=1
                ),
                key_prefix=QUEUE_PREFIX,
            )
        )
        try:
            _open(page, web_url)
            queue = page.get_by_role("region", name="Annotation queue")
            expect(queue.get_by_role("alert")).to_have_text(
                "The annotation queue did not answer; retry."
            )
            expect(_summary(page)).to_have_count(0)
            expect(page.get_by_role("region", name="Pending")).to_have_count(0)
        finally:
            agents.set_annotation_queue(working)


def test_review_labels_route_serves_the_reviewer_labels(runtime_url):
    response = httpx.get(f"{runtime_url}/agents/annotations/labels", timeout=30)
    assert (response.status_code, response.json()) == (
        200,
        {"labels": [label.value for label in REVIEW_LABELS]},
    )
