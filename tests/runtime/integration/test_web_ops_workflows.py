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
    build_web_client,
    install_web_client,
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
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("web_ops")))


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
        self, page, web_url, tenant
    ):
        _show(page, web_url, tenant)
        assert _refresh_until(
            page, tenant, 2, [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
        ) == [MISSING_CLIP["query"], LECTURE_REPORT["query"]]
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
        form.get_by_label("The pattern was optimal").uncheck()
        form.get_by_label("Suggested pattern").select_option("parallel")
        form.get_by_label("Why that pattern").fill("the two steps are independent")
        form.get_by_label("The right agents were used").uncheck()
        form.get_by_label("Missing agents").fill("report_agent")
        form.get_by_label("Unnecessary agents").fill("summarizer_agent")
        form.get_by_label("Improvement notes").fill("use the report agent for reports")
        form.get_by_role("button", name="Save review").click()
        expect(page.get_by_role("status")).to_have_text(
            "Saved the review of wf-lecture-report: poor."
        )
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
                "execution_order_is_optimal": True,
                "improvement_notes": "use the report agent for reports",
                "annotator_id": "reviewer@example.com",
            },
        )


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
