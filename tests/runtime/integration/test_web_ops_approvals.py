"""The web client's Approvals view, driven in Chromium against the approval
store on real Phoenix and Redis.

The built client's Node server forwards to the runtime's approval routes on a
real uvicorn socket. Each test saves a review batch through the production
store, takes every decision through the page, and reads the outcome back from
the store: the queue the routes serve and the tenant's approved training
dataset.
"""

from __future__ import annotations

import json
import re
import time
from uuid import uuid4

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.optimizer.entity_self_consistency import (
    SELF_CONSISTENCY_METADATA_KEY,
)
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_runtime.routers import approvals
from tests.utils.approval_review import (
    ROUTING,
    SELF_CONSISTENCY,
    WORKFLOW,
    approved_rows,
    approved_rows_until,
    reject_without_regenerating,
    review_config_manager,
    run_in_own_loop,
    save_review_batch,
)
from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]


@pytest.fixture(scope="module")
def review_config(phoenix_container, workflow_state_redis_url):
    return review_config_manager(phoenix_container, workflow_state_redis_url)


@pytest.fixture(scope="module")
def runtime_url(review_config, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        review_config, schema_loader, workflow_state_redis_url
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
def review(review_config, telemetry_manager_with_phoenix, runtime_url):
    """A fresh tenant whose saved batch holds ``{batch}_routing`` and
    ``{batch}_workflow`` awaiting review and ``{batch}_confident``
    auto-approved."""
    tenant = f"webapprv{uuid4().hex[:8]}:main"
    batch = f"batch_{uuid4().hex[:8]}"
    storage = ApprovalStorageImpl.from_system_config(
        review_config, telemetry_manager_with_phoenix, tenant
    )
    save_review_batch(storage, batch)
    yield {
        "tenant": tenant,
        "batch": batch,
        "storage": storage,
        "routing": f"{batch}_routing",
        "workflow": f"{batch}_workflow",
    }
    approvals.set_config_manager(review_config)


def _served_queue(runtime_url, tenant):
    """The items the approval route serves, by ID."""
    response = httpx.get(f"{runtime_url}/admin/tenant/{tenant}/approvals", timeout=60)
    assert response.status_code == 200, response.text
    return {item["item_id"]: item for item in response.json()["items"]}


def _served_queue_until(runtime_url, tenant, want, timeout=60.0):
    """The served queue once its IDs equal ``want`` (Phoenix serves spans and
    annotations after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while set(items := _served_queue(runtime_url, tenant)) != want:
        if time.monotonic() > deadline:
            break
        time.sleep(2)
    return items


def _show(page: Page, web_url: str, tenant: str):
    page.goto(f"{web_url}/#/ops/approvals")
    expect(page.get_by_role("heading", name="Approvals", level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show review queue").click()


def _queue(page: Page, tenant: str):
    return page.get_by_role("region", name=f"Awaiting review in {tenant}")


def _item(page: Page, item_id: str):
    return page.get_by_role("region", name=f"Review {item_id}")


def _review_titles(page: Page):
    return page.locator("section.panel h2").filter(has_text=re.compile(r"^Review "))


def _shown_queue_until(page: Page, tenant: str, want: set[str], timeout=60.0):
    """Refresh the page's queue until it shows exactly ``want``."""
    deadline = time.monotonic() + timeout
    while True:
        expect(_queue(page, tenant).locator("p.muted")).to_be_visible()
        shown = {
            title.removeprefix("Review ")
            for title in _review_titles(page).all_inner_texts()
        }
        if shown == want or time.monotonic() > deadline:
            return shown
        _queue(page, tenant).get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


def _facts(panel) -> dict[str, str]:
    terms = panel.locator("dl.facts > dt").all_inner_texts()
    values = panel.locator("dl.facts > dd").all_inner_texts()
    return dict(zip(terms, values, strict=True))


def _decide(page: Page, item_id: str, button: str):
    _item(page, item_id).get_by_role("button", name=button, exact=True).click()


def _rejection(page: Page, item_id: str):
    return _item(page, item_id).get_by_role("form", name=f"Reject {item_id}")


def _section(page: Page, name: str):
    page.get_by_role("navigation", name="Approval sections").get_by_role(
        "button", name=name, exact=True
    ).click()


def _history_until(runtime_url, tenant, done, timeout=60.0):
    """The served review history once ``done(history)`` holds (Phoenix
    serves spans and annotations after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        response = httpx.get(
            f"{runtime_url}/admin/tenant/{tenant}/approvals/history", timeout=60
        )
        assert response.status_code == 200, response.text
        history = response.json()
        if done(history) or time.monotonic() > deadline:
            return history
        time.sleep(2)


def _ids(entries):
    return [entry["item_id"] for entry in entries]


def _rows(table, count: int) -> list[list[str]]:
    """The cells of each of the table's ``count`` body rows, once it has them."""
    expect(table.locator("tbody tr")).to_have_count(count)
    return [
        row.get_by_role("cell").all_inner_texts()
        for row in table.locator("tbody tr").all()
    ]


class TestApprovalsView:
    def test_a_reviewer_corrects_rejects_and_approves_through_the_page(
        self, page, web_url, runtime_url, review
    ):
        tenant, batch = review["tenant"], review["batch"]
        routing, workflow = review["routing"], review["workflow"]
        served = _served_queue_until(runtime_url, tenant, {routing, workflow})

        _show(page, web_url, tenant)
        assert _shown_queue_until(page, tenant, {routing, workflow}) == {
            routing,
            workflow,
        }
        expect(
            _queue(page, tenant).get_by_text("2 items await review.")
        ).to_be_visible()
        facts = _facts(_item(page, routing))
        assert facts == {
            "Batch": batch,
            "Schema": "RoutingExperienceSchema",
            "Status": "pending_review",
            "Confidence": "0.40",
            "Band": "Very low",
            "Retry count": "0",
            "Created": served[routing]["created_at"],
            "Query": ROUTING["query"],
            **(
                {"Reasoning": served[routing]["reasoning"]}
                if served[routing]["reasoning"]
                else {}
            ),
            "Entities": "gradient descent (CONCEPT)",
            "Self-consistency": (
                "Agreement (5 samples): gradient descent (CONCEPT) 0.60 — needs review\n"
                "Agreement (5 samples): lecture (MEDIA) 1.00"
            ),
        }
        # The band reads in its tone, as on the synthetic results.
        expect(_item(page, routing).locator("dd.band.error")).to_have_text("Very low")
        assert served[routing]["metadata"][SELF_CONSISTENCY_METADATA_KEY] == (
            SELF_CONSISTENCY
        )
        assert (
            json.loads(
                _item(page, routing).get_by_label("Example", exact=True).text_content()
            )
            == ROUTING
        )
        expect(_item(page, routing).get_by_label("Generation metadata")).to_have_count(
            0
        )

        _decide(page, routing, "Approve")
        expect(_item(page, routing).get_by_role("alert")).to_have_text(
            "Enter your name as the reviewer first."
        )
        _queue(page, tenant).get_by_label("Reviewer").fill("reviewer@example.com")

        # Each correctable field has its own editor, prefilled with its value.
        template = served[workflow]["correction_template"]
        _decide(page, workflow, "Reject with corrections")
        form = _rejection(page, workflow)
        assert form.locator("fieldset > label").evaluate_all(
            "labels => labels.map(label => [...label.childNodes]"
            ".filter(node => node.nodeType === Node.TEXT_NODE)"
            ".map(node => node.textContent).join(''))"
        ) == list(template)
        for name, value in template.items():
            editor = form.get_by_label(name, exact=True)
            if isinstance(value, bool):
                assert editor.is_checked() is value
            elif isinstance(value, (list, dict)):
                assert json.loads(editor.input_value()) == value
            else:
                assert editor.input_value() == str(value)
        expect(_item(page, routing).get_by_role("button")).to_have_text(
            ["Approve", "Reject and regenerate"]
        )

        form.get_by_role("button", name="Submit rejection").click()
        expect(_item(page, workflow).get_by_role("alert")).to_have_text(
            "A WorkflowExecutionSchema rejection needs at least one correction."
        )
        form.get_by_label("task_count", exact=True).fill("three")
        form.get_by_role("button", name="Submit rejection").click()
        expect(form.get_by_role("alert")).to_have_text("task_count must be a number.")
        form.get_by_role("button", name="Cancel").click()
        expect(_rejection(page, workflow)).to_have_count(0)
        expect(_item(page, workflow).get_by_role("button")).to_have_text(
            ["Approve", "Reject with corrections"]
        )
        assert {
            key: item["data"]
            for key, item in _served_queue(runtime_url, tenant).items()
        } == {routing: ROUTING, workflow: WORKFLOW}
        assert approved_rows(review["storage"], routing) == 0

        corrected = ["video_search_agent", "detailed_report_agent"]
        _decide(page, workflow, "Reject with corrections")
        form = _rejection(page, workflow)
        expect(form.get_by_label("task_count", exact=True)).to_have_value(
            str(WORKFLOW["task_count"])
        )
        form.get_by_label("Feedback").fill("a report needs the report agent")
        form.get_by_label("agent_sequence", exact=True).fill(json.dumps(corrected))
        form.get_by_role("button", name="Submit rejection").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(
            f"Rejected {workflow}; its corrected replacement "
        )
        replacement = (
            notice.inner_text()
            .removeprefix(f"Rejected {workflow}; its corrected replacement ")
            .removesuffix(" awaits review.")
        )
        served = _served_queue_until(runtime_url, tenant, {routing, replacement})
        assert (
            served[replacement]["status"],
            served[replacement]["data"],
            served[replacement]["metadata"]["original_item_id"],
            served[replacement]["metadata"]["regeneration_feedback"],
        ) == (
            "regenerated",
            dict(WORKFLOW, agent_sequence=corrected),
            workflow,
            "a report needs the report agent",
        )
        assert _shown_queue_until(page, tenant, {routing, replacement}) == {
            routing,
            replacement,
        }

        _decide(page, routing, "Approve")
        expect(page.get_by_role("status")).to_have_text(
            f"Approved {routing} into the training dataset."
        )
        assert approved_rows_until(review["storage"], routing, 1) == 1
        assert set(_served_queue_until(runtime_url, tenant, {replacement})) == {
            replacement
        }
        assert _shown_queue_until(page, tenant, {replacement}) == {replacement}
        expect(
            _queue(page, tenant).get_by_text("1 item awaits review.")
        ).to_be_visible()

        # The history tabs read the decisions back from the store.
        history = _history_until(
            runtime_url,
            tenant,
            lambda body: (
                _ids(body["approved"]) == [routing, f"{batch}_confident"]
                and _ids(body["rejected"]) == [workflow]
            ),
        )
        _section(page, "Approved")
        approved = page.get_by_role("region", name=f"Approved items of {tenant}")
        expect(approved.get_by_text("2 items approved.")).to_be_visible()
        assert _rows(approved.get_by_role("table", name="Approved items"), 2) == [
            [
                routing,
                ROUTING["query"],
                "0.40",
                "approved",
                "reviewer@example.com",
                history["approved"][0]["reviewed_at"],
            ],
            [
                f"{batch}_confident",
                "play the intro clip",
                "0.95",
                "auto_approved",
                "—",
                "—",
            ],
        ]
        _section(page, "Rejected")
        rejected = page.get_by_role("region", name=f"Rejected items of {tenant}")
        expect(rejected.get_by_text("1 item rejected.")).to_be_visible()
        assert _rows(rejected.get_by_role("table", name="Rejected items"), 1) == [
            [
                workflow,
                WORKFLOW["query"],
                "a report needs the report agent",
                json.dumps({"agent_sequence": corrected}, separators=(",", ":")),
                "reviewer@example.com",
                f"{replacement} (regenerated)",
                "",
            ]
        ]
        _section(page, "Statistics")
        stats = page.get_by_role("region", name=f"Review statistics of {tenant}")
        totals = stats.locator("dl.facts > dd")
        expect(stats.locator("dl.facts > dt")).to_have_text(
            [
                "Total items",
                "Awaiting review",
                "Auto-approved",
                "Approved",
                "Rejected",
                "Approval rate",
            ]
        )
        expect(totals).to_have_text(["4", "1", "1", "1", "1", "50.0%"])
        bars = stats.get_by_role("figure", name="Average confidence by status")
        expect(bars.locator(".bar-label")).to_have_text(
            ["Awaiting review", "Auto-approved", "Approved", "Rejected"]
        )
        expect(bars.locator(".bar-value")).to_have_text(
            [f"{served[replacement]['confidence']:.2f}", "0.95", "0.40", "0.30"]
        )

    def test_a_rejected_item_nothing_replaced_is_regenerated_from_the_page(
        self, page, web_url, runtime_url, review
    ):
        tenant, batch = review["tenant"], review["batch"]
        routing, workflow = review["routing"], review["workflow"]
        corrected = ["video_search_agent", "detailed_report_agent"]
        _served_queue_until(runtime_url, tenant, {routing, workflow})
        run_in_own_loop(
            reject_without_regenerating(
                review["storage"],
                batch,
                workflow,
                feedback="wrong agents",
                corrections={"agent_sequence": corrected},
                reviewer="earlier@example.com",
            )
        )
        _history_until(
            runtime_url, tenant, lambda body: _ids(body["rejected"]) == [workflow]
        )
        _show(page, web_url, tenant)
        _section(page, "Rejected")
        rejected = page.get_by_role("region", name=f"Rejected items of {tenant}")
        table = rejected.get_by_role("table", name="Rejected items")
        assert _rows(table, 1) == [
            [
                workflow,
                WORKFLOW["query"],
                "wrong agents",
                json.dumps({"agent_sequence": corrected}, separators=(",", ":")),
                "earlier@example.com",
                "—",
                "Regenerate",
            ]
        ]
        table.get_by_role("button", name="Regenerate").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(f"Regenerated {workflow} as ")
        replacement = (
            notice.inner_text()
            .removeprefix(f"Regenerated {workflow} as ")
            .removesuffix("; it awaits review.")
        )
        served = _served_queue_until(runtime_url, tenant, {routing, replacement})
        assert (served[replacement]["data"], served[replacement]["status"]) == (
            dict(WORKFLOW, agent_sequence=corrected),
            "regenerated",
        )
        _history_until(
            runtime_url,
            tenant,
            lambda body: body["rejected"][0]["replacement_id"] == replacement,
        )
        rejected.get_by_role("button", name="Refresh").click()
        expect(table.locator("tbody tr").get_by_role("cell")).to_have_text(
            [
                workflow,
                WORKFLOW["query"],
                "wrong agents",
                json.dumps({"agent_sequence": corrected}, separators=(",", ":")),
                "earlier@example.com",
                f"{replacement} (regenerated)",
                "",
            ]
        )
        _section(page, "Pending")
        assert _shown_queue_until(page, tenant, {routing, replacement}) == {
            routing,
            replacement,
        }

    def test_an_item_no_schema_describes_is_rejected_without_feedback(
        self, page, web_url, runtime_url, review
    ):
        tenant = review["tenant"]
        batch = f"batch_{uuid4().hex[:8]}"
        note = f"{batch}_note"
        run_in_own_loop(
            review["storage"].save_batch(
                ApprovalBatch(
                    batch_id=batch,
                    items=[
                        ReviewItem(
                            item_id=note,
                            data={"query": "what is this", "note": "free-form"},
                            metadata={"agent_type": "routing"},
                            confidence=0.2,
                            status=ApprovalStatus.PENDING_REVIEW,
                        )
                    ],
                    context={"tenant_id": tenant, "optimizer": "routing"},
                )
            )
        )
        pending = {review["routing"], review["workflow"], note}
        _served_queue_until(runtime_url, tenant, pending)
        _show(page, web_url, tenant)
        assert _shown_queue_until(page, tenant, pending) == pending
        _queue(page, tenant).get_by_label("Reviewer").fill("reviewer@example.com")
        item = _item(page, note)
        expect(
            item.get_by_text(
                "No example schema describes this item, so it can be approved "
                "or rejected but not corrected."
            )
        ).to_be_visible()
        _decide(page, note, "Reject")
        form = _rejection(page, note)
        expect(form.locator("fieldset")).to_have_count(0)
        form.get_by_role("button", name="Submit rejection").click()
        expect(page.get_by_role("status")).to_have_text(f"Rejected {note}.")
        _history_until(
            runtime_url, tenant, lambda body: _ids(body["rejected"]) == [note]
        )
        _section(page, "Rejected")
        rejected = page.get_by_role("region", name=f"Rejected items of {tenant}")
        assert _rows(rejected.get_by_role("table", name="Rejected items"), 1) == [
            [note, "what is this", "—", "—", "reviewer@example.com", "—", ""]
        ]

    def test_a_tenant_with_nothing_to_review_says_so_on_every_tab(
        self, page, web_url, runtime_url, telemetry_manager_with_phoenix
    ):
        tenant = f"webapprv{uuid4().hex[:8]}:main"
        assert _served_queue(runtime_url, tenant) == {}
        _show(page, web_url, tenant)
        expect(
            _queue(page, tenant).get_by_text(f"Nothing awaits review in {tenant}.")
        ).to_be_visible()
        for section, region, text in (
            (
                "Approved",
                f"Approved items of {tenant}",
                f"No approved items in {tenant}.",
            ),
            (
                "Rejected",
                f"Rejected items of {tenant}",
                f"No rejected items in {tenant}.",
            ),
            (
                "Statistics",
                f"Review statistics of {tenant}",
                f"No items in {tenant} yet.",
            ),
        ):
            _section(page, section)
            panel = page.get_by_role("region", name=region)
            expect(panel.get_by_text(text)).to_be_visible()
            expect(panel.get_by_role("table")).to_have_count(0)


class TestConcurrency:
    def test_a_decision_another_reviewer_made_first_is_refused_not_overwritten(
        self, browser, web_url, runtime_url, review
    ):
        tenant = review["tenant"]
        routing, workflow = review["routing"], review["workflow"]
        _served_queue_until(runtime_url, tenant, {routing, workflow})
        contexts = [browser.new_context() for _ in range(2)]
        first, second = (context.new_page() for context in contexts)
        try:
            for page, reviewer in (
                (first, "first@example.com"),
                (second, "second@example.com"),
            ):
                _show(page, web_url, tenant)
                assert _shown_queue_until(page, tenant, {routing, workflow}) == {
                    routing,
                    workflow,
                }
                _queue(page, tenant).get_by_label("Reviewer").fill(reviewer)

            # Another reviewer's rejection of the workflow item is elected in
            # Redis while both pages still show it.
            run_in_own_loop(
                review["storage"].select_review_decision(
                    batch_id=review["batch"],
                    original_item_id=workflow,
                    decision=ReviewDecision(
                        item_id=workflow,
                        approved=False,
                        feedback="wrong agent",
                        reviewer="rival@example.com",
                    ),
                )
            )
            _decide(first, workflow, "Approve")
            expect(_item(first, workflow).get_by_role("alert")).to_have_text(
                f"Item {workflow} was already decided by another reviewer."
            )

            _decide(first, routing, "Approve")
            expect(first.get_by_role("status")).to_have_text(
                f"Approved {routing} into the training dataset."
            )
            # The second page still shows the item the first approved, after
            # the store stopped serving it as pending.
            assert set(_served_queue_until(runtime_url, tenant, {workflow})) == {
                workflow
            }
            _decide(second, routing, "Approve")
            expect(_item(second, routing).get_by_role("alert")).to_have_text(
                f"Item {routing} of batch {review['batch']} is not awaiting review."
            )
        finally:
            for context in contexts:
                context.close()
        assert approved_rows_until(review["storage"], routing, 1) == 1
        assert approved_rows(review["storage"], workflow) == 0
        served = _served_queue(runtime_url, tenant)
        assert {key: item["data"] for key, item in served.items()} == {
            workflow: WORKFLOW
        }


class TestFaults:
    def test_an_unreachable_store_shows_an_outage_not_an_empty_queue(
        self, page, web_url, phoenix_container, workflow_state_redis_url, review
    ):
        tenant = review["tenant"]
        approvals.set_config_manager(
            review_config_manager(
                phoenix_container,
                workflow_state_redis_url,
                telemetry_url="http://127.0.0.1:9",
            )
        )
        _show(page, web_url, tenant)
        queue = _queue(page, tenant)
        expect(queue.get_by_role("alert")).to_have_text(
            f"Could not read the items awaiting review for tenant {tenant}."
        )
        expect(queue.get_by_text(f"Nothing awaits review in {tenant}.")).to_have_count(
            0
        )
        expect(_review_titles(page)).to_have_count(0)
        for section, region, empty in (
            (
                "Approved",
                f"Approved items of {tenant}",
                f"No approved items in {tenant}.",
            ),
            (
                "Rejected",
                f"Rejected items of {tenant}",
                f"No rejected items in {tenant}.",
            ),
            (
                "Statistics",
                f"Review statistics of {tenant}",
                f"No items in {tenant} yet.",
            ),
        ):
            _section(page, section)
            panel = page.get_by_role("region", name=region)
            expect(panel.get_by_role("alert")).to_have_text(
                f"Could not read the review history of tenant {tenant}."
            )
            expect(panel.get_by_text(empty)).to_have_count(0)
            expect(panel.get_by_role("table")).to_have_count(0)
