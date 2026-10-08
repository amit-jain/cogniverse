"""The web client's Workflow reviews view, driven in Chromium against
orchestration spans in real Phoenix.

Workflows are emitted by the orchestrator's own span writer. The runtime reads
and annotates them through a forwarding proxy in front of Phoenix's HTTP API,
so a test can hold or fail the annotation writes. Every review is read back
from Phoenix.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotationStorage,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

KEY = "web-ops-harness-key"
LECTURE_REPORT = {
    "workflow_id": "wf-lecture-report",
    "query": "summarize the lecture and write a report",
    "agent_sequence": ["search_agent", "summarizer_agent"],
    "execution_order": ["search_agent", "summarizer_agent"],
    "execution_time": 3.5,
    "success": True,
    "tasks_completed": 2,
    "pattern": "sequential",
}
MISSING_CLIP = {
    "workflow_id": "wf-missing-clip",
    "query": "find the keynote intro clip",
    "agent_sequence": ["search_agent"],
    "execution_order": ["search_agent"],
    "execution_time": 1.25,
    "success": False,
    "tasks_completed": 0,
    "pattern": "parallel",
    "error_summary": "search_agent timed out",
}
EARLIER_DIGEST = {
    "workflow_id": "wf-earlier-digest",
    "query": "digest yesterday's standups",
    "agent_sequence": ["summarizer_agent"],
    "execution_order": ["summarizer_agent"],
    "execution_time": 2.0,
    "success": True,
    "tasks_completed": 1,
    "pattern": "sequential",
}


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


@pytest.fixture(scope="module")
def runtime_url(phoenix_container, schema_loader, workflow_state_redis_url, telemetry):
    with serve_ops_runtime(
        review_config_manager(phoenix_container, workflow_state_redis_url),
        schema_loader,
        workflow_state_redis_url,
    ) as url:
        yield url


@pytest.fixture()
def web_url(built_client, runtime_url, phoenix_proxy):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []
    phoenix_proxy.intercept = None


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
def tenant(telemetry):
    """A fresh tenant with LECTURE_REPORT then MISSING_CLIP recorded."""
    tenant_id = canonical_tenant_id(f"webwf{uuid4().hex[:8]}")
    emitter = SimpleNamespace(telemetry_manager=telemetry)
    for workflow in (LECTURE_REPORT, MISSING_CLIP):
        run_in_own_loop(
            OrchestratorAgent._emit_orchestration_span(
                emitter, tenant_id=tenant_id, **workflow
            )
        )
    telemetry.force_flush(timeout_millis=10000)
    return tenant_id


def _show(page: Page, web_url: str, tenant: str):
    page.goto(f"{web_url}/#/ops/workflows")
    expect(
        page.get_by_role("heading", name="Workflow reviews", level=1)
    ).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show workflows").click()


def _panel(page: Page, tenant: str):
    return page.get_by_role("region", name=f"Workflows of {tenant}")


def _column(page: Page, tenant: str, index: int):
    return (
        _panel(page, tenant)
        .get_by_role("table", name="Workflows")
        .locator(f"tbody tr td:nth-child({index})")
    )


def _refresh_until(page: Page, tenant: str, index: int, want: list[str], timeout=60.0):
    """Refresh the list until column ``index`` reads ``want`` (Phoenix serves
    spans and annotations after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        expect(
            _panel(page, tenant).locator("table, p.muted, .alert").first
        ).to_be_visible()
        shown = _column(page, tenant, index).all_inner_texts()
        if shown == want or time.monotonic() > deadline:
            return shown
        _panel(page, tenant).get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


def _stored_reviews(tenant: str, want: int, timeout=60.0):
    """``{workflow_id: (label, score, metadata)}`` of the human reviews Phoenix
    stores for the tenant, once there are ``want`` of them."""
    storage = OrchestrationAnnotationStorage(tenant_id=tenant)
    deadline = time.monotonic() + timeout
    while True:
        end = datetime.now(timezone.utc)
        rows = run_in_own_loop(
            storage.query_annotated_spans(
                start_time=end - timedelta(hours=1), end_time=end
            )
        )
        reviews = {
            row["annotations"][0]["metadata"]["workflow_id"]: (
                row["annotations"][0]["result"]["label"],
                row["annotations"][0]["result"]["score"],
                row["annotations"][0]["metadata"],
            )
            for row in rows
        }
        if len(reviews) == want or time.monotonic() > deadline:
            return reviews
        time.sleep(2)


class TestWorkflowReviewsView:
    def test_a_reviewer_rates_a_workflow_and_the_review_lands_on_its_span(
        self, page, web_url, tenant, phoenix_proxy
    ):
        _show(page, web_url, tenant)
        expect(page.locator(".ops-view > p.muted").first).to_have_text(
            "Review orchestration workflows to improve future routing and "
            "orchestration decisions. Your reviews become ground truth for "
            "optimization."
        )
        assert _refresh_until(
            page, tenant, 2, [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
        ) == [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
        expect(
            _panel(page, tenant).get_by_text("Found 2 workflows.", exact=True)
        ).to_be_visible()
        rows = _panel(page, tenant).locator("tbody tr")
        expect(rows.nth(0).locator("td").nth(1)).to_have_text(MISSING_CLIP["query"])
        assert [row.locator("td").all_inner_texts()[1:7] for row in rows.all()] == [
            [
                "find the keynote intro clip",
                "parallel",
                "search_agent",
                "1.25s",
                "failed: search_agent timed out",
                "—",
            ],
            [
                "summarize the lecture and write a report",
                "sequential",
                "search_agent → summarizer_agent",
                "3.50s",
                "succeeded",
                "—",
            ],
        ]

        _panel(page, tenant).get_by_role(
            "button", name="Review wf-lecture-report"
        ).click()
        review = page.get_by_role("region", name="Review wf-lecture-report")
        terms = review.locator("dl.facts > dt").all_inner_texts()
        values = review.locator("dl.facts > dd").all_inner_texts()
        assert dict(zip(terms, values, strict=True)) == {
            "Query": "summarize the lecture and write a report",
            "Pattern": "sequential",
            "Agent sequence": "search_agent, summarizer_agent",
            "Execution order": "search_agent, summarizer_agent",
            "Tasks completed": "2",
            "Outcome": "succeeded",
        }
        form = review.get_by_role("form", name="Review of wf-lecture-report")
        form.get_by_label("Quality").select_option("poor")
        form.get_by_label("Score (0–1)").fill("0.3")
        form.get_by_role("button", name="Save review").click()
        expect(form.get_by_role("alert")).to_have_text(
            "Enter your name as the reviewer first."
        )

        _panel(page, tenant).get_by_label("Reviewer").fill("reviewer@example.com")
        form.get_by_role("group", name="Was the pattern optimal?").get_by_label(
            "No"
        ).check()
        form.get_by_label("Suggested pattern").select_option("parallel")
        form.get_by_label("Why that pattern").fill("the two steps are independent")
        form.get_by_label("The right agents were used").uncheck()
        form.get_by_label("Missing agents").fill("report_agent")
        form.get_by_label("Unnecessary agents").fill("summarizer_agent")
        form.get_by_label("The execution order was optimal").uncheck()
        form.get_by_label("Suggested order").fill("report_agent\nsearch_agent")
        form.get_by_label("Why that order").fill("outline the report first")
        form.get_by_label("What went well").fill("the search found the lecture")
        form.get_by_label("What went wrong").fill("no report was written")
        form.get_by_label("Improvement notes").fill("use the report agent for reports")
        written = threading.Event()

        def slow_reads_after_the_write(method, path, body):
            # Phoenix serves a new annotation after an indexing delay; hold
            # every read after the write so the page cannot lean on one.
            if method == "POST" and "annotations" in path:
                written.set()
            elif written.is_set():
                time.sleep(8)
            return None

        phoenix_proxy.intercept = slow_reads_after_the_write
        form.get_by_role("button", name="Save review").click()
        expect(page.get_by_role("status")).to_have_text(
            "Saved the review of wf-lecture-report: poor."
        )
        # The saved review shows at once, from the save's own answer.
        expect(_column(page, tenant, 7)).to_have_text(
            ["—", "poor (0.30) by reviewer@example.com"], timeout=4000
        )
        phoenix_proxy.intercept = None
        assert _refresh_until(
            page, tenant, 7, ["—", "poor (0.30) by reviewer@example.com"]
        ) == ["—", "poor (0.30) by reviewer@example.com"]

        reviews = _stored_reviews(tenant, 1)
        label, score, metadata = reviews["wf-lecture-report"]
        assert (
            set(reviews),
            label,
            score,
            {
                key: metadata[key]
                for key in (
                    "pattern_is_optimal",
                    "suggested_pattern",
                    "pattern_feedback",
                    "agents_are_correct",
                    "missing_agents",
                    "unnecessary_agents",
                    "suggested_agents",
                    "execution_order_is_optimal",
                    "suggested_execution_order",
                    "execution_order_feedback",
                    "what_went_well",
                    "what_went_wrong",
                    "improvement_notes",
                    "annotator_id",
                )
            },
        ) == (
            {"wf-lecture-report"},
            "poor",
            0.3,
            {
                "pattern_is_optimal": False,
                "suggested_pattern": "parallel",
                "pattern_feedback": "the two steps are independent",
                "agents_are_correct": False,
                "missing_agents": "report_agent",
                "unnecessary_agents": "summarizer_agent",
                "suggested_agents": "search_agent,report_agent",
                "execution_order_is_optimal": False,
                "suggested_execution_order": "report_agent,search_agent",
                "execution_order_feedback": "outline the report first",
                "what_went_well": "the search found the lecture",
                "what_went_wrong": "no report was written",
                "improvement_notes": "use the report agent for reports",
                "annotator_id": "reviewer@example.com",
            },
        )

    def test_the_window_and_the_cap_choose_the_workflows_and_unsure_is_not_optimal(
        self, page, web_url, tenant, telemetry
    ):
        three_hours_ago = time.time_ns() - 3 * 3600 * 10**9

        class Earlier:
            """The telemetry manager, starting each span three hours ago."""

            def span(self, name, tenant_id):
                return telemetry.span(
                    name, tenant_id=tenant_id, start_time=three_hours_ago
                )

        run_in_own_loop(
            OrchestratorAgent._emit_orchestration_span(
                SimpleNamespace(telemetry_manager=Earlier()),
                tenant_id=tenant,
                **EARLIER_DIGEST,
            )
        )
        telemetry.force_flush(timeout_millis=10000)
        queries = [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
        _show(page, web_url, tenant)
        panel = _panel(page, tenant)
        assert _refresh_until(page, tenant, 2, [*queries, EARLIER_DIGEST["query"]]) == [
            *queries,
            EARLIER_DIGEST["query"],
        ]
        expect(panel.get_by_text("Found 3 workflows.", exact=True)).to_be_visible()

        panel.get_by_label("Window").select_option("Last hour")
        assert _refresh_until(page, tenant, 2, queries) == queries
        expect(panel.get_by_text("Found 2 workflows.", exact=True)).to_be_visible()

        with page.expect_response(
            lambda r: "orchestration-workflows?lookback_hours=1&limit=1" in r.url
        ):
            panel.get_by_label("Max workflows").select_option("1")
        expect(
            panel.get_by_text("Found 1 workflow, the newest 1.", exact=True)
        ).to_be_visible()
        assert _column(page, tenant, 2).all_inner_texts() == [MISSING_CLIP["query"]]

        panel.get_by_label("Reviewer").fill("reviewer@example.com")
        panel.get_by_role("button", name="Review wf-missing-clip").click()
        form = page.get_by_role("form", name="Review of wf-missing-clip")
        verdict = form.get_by_role("group", name="Was the pattern optimal?")
        expect(verdict.get_by_role("radio")).to_have_count(3)
        expect(verdict.get_by_label("Yes")).to_be_checked()
        verdict.get_by_label("No").check()
        expect(form.get_by_label("Suggested pattern")).to_be_visible()
        verdict.get_by_label("Unsure").check()
        expect(form.get_by_label("Suggested pattern")).to_have_count(0)
        form.get_by_label("Quality").select_option("acceptable")
        form.get_by_label("Score (0–1)").fill("0.5")
        form.get_by_role("button", name="Save review").click()
        expect(page.get_by_role("status")).to_have_text(
            "Saved the review of wf-missing-clip: acceptable."
        )
        label, score, metadata = _stored_reviews(tenant, 1)["wf-missing-clip"]
        assert (
            label,
            score,
            metadata["pattern_is_optimal"],
            metadata.get("suggested_pattern"),
            metadata.get("pattern_feedback"),
        ) == ("acceptable", 0.5, False, None, None)


class TestConcurrency:
    def test_two_reviewers_saving_at_once_each_land_on_their_own_workflow(
        self, browser, web_url, tenant, phoenix_proxy
    ):
        barrier = threading.Barrier(2, timeout=30)
        held = []
        lock = threading.Lock()

        def hold_annotation_writes(method, path, body):
            # Both annotation writes reach Phoenix together.
            if method == "POST" and "annotations" in path:
                with lock:
                    held.append(json.loads(body))
                barrier.wait()
            return None

        contexts = [browser.new_context() for _ in range(2)]
        pages = [context.new_page() for context in contexts]
        reviews = (
            ("wf-lecture-report", "first@example.com", "excellent"),
            ("wf-missing-clip", "second@example.com", "failed"),
        )
        try:
            for page, (workflow, reviewer, label) in zip(pages, reviews, strict=True):
                _show(page, web_url, tenant)
                _refresh_until(
                    page, tenant, 2, [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
                )
                _panel(page, tenant).get_by_label("Reviewer").fill(reviewer)
                _panel(page, tenant).get_by_role(
                    "button", name=f"Review {workflow}"
                ).click()
                form = page.get_by_role("form", name=f"Review of {workflow}")
                form.get_by_label("Quality").select_option(label)
                form.get_by_label("Score (0–1)").fill(
                    "0.9" if label == "excellent" else "0"
                )
            phoenix_proxy.intercept = hold_annotation_writes
            for page, (workflow, _, _) in zip(pages, reviews, strict=True):
                page.get_by_role("form", name=f"Review of {workflow}").get_by_role(
                    "button", name="Save review"
                ).click(no_wait_after=True)
            for page, (workflow, _, label) in zip(pages, reviews, strict=True):
                expect(page.get_by_role("status")).to_have_text(
                    f"Saved the review of {workflow}: {label}."
                )
            phoenix_proxy.intercept = None
        finally:
            phoenix_proxy.intercept = None
            for context in contexts:
                context.close()
        assert len(held) == 2
        stored = _stored_reviews(tenant, 2)
        assert {
            workflow: (label, score, metadata["annotator_id"])
            for workflow, (label, score, metadata) in stored.items()
        } == {
            "wf-lecture-report": ("excellent", 0.9, "first@example.com"),
            "wf-missing-clip": ("failed", 0.0, "second@example.com"),
        }


class TestFaults:
    def test_an_unreachable_telemetry_backend_shows_an_outage_not_an_empty_list(
        self, page, web_url, tenant, phoenix_proxy
    ):
        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        _show(page, web_url, tenant)
        panel = _panel(page, tenant)
        expect(panel.get_by_role("alert")).to_have_text(
            f"Could not read the orchestration workflows of tenant {tenant}."
        )
        expect(
            panel.get_by_text("No orchestration workflows in this window.")
        ).to_have_count(0)
        expect(panel.get_by_role("table", name="Workflows")).to_have_count(0)

    def test_a_review_the_backend_does_not_store_shows_why(
        self, page, web_url, runtime_url, tenant, phoenix_proxy
    ):
        _show(page, web_url, tenant)
        _refresh_until(
            page, tenant, 2, [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
        )
        _panel(page, tenant).get_by_label("Reviewer").fill("reviewer@example.com")
        _panel(page, tenant).get_by_role(
            "button", name="Review wf-lecture-report"
        ).click()
        form = page.get_by_role("form", name="Review of wf-lecture-report")
        form.get_by_label("Quality").select_option("good")
        form.get_by_label("Score (0–1)").fill("0.8")
        listed = httpx.get(
            f"{runtime_url}/admin/tenant/{tenant}/orchestration-workflows",
            params={"lookback_hours": 1},
            timeout=60,
        )
        [span_id] = [
            w["span_id"]
            for w in listed.json()["workflows"]
            if w["workflow_id"] == "wf-lecture-report"
        ]

        def fail_annotation_writes(method, path, body):
            if method == "POST" and "annotations" in path:
                return (503, {"detail": "down"})
            return None

        phoenix_proxy.intercept = fail_annotation_writes
        form.get_by_role("button", name="Save review").click()
        expect(form.get_by_role("alert")).to_have_text(
            f"The review of workflow span {span_id} was not stored."
        )
        expect(form.get_by_role("group", name="Failure details")).to_have_count(1)
        expect(form.locator("details pre")).to_have_text(
            "error annotation_not_stored, failure HTTPStatusError, HTTP 502"
        )
        expect(page.get_by_role("status")).to_have_count(0)
