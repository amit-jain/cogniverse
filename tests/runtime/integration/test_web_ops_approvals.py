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
from cogniverse_core.approval.interfaces import ReviewDecision
from cogniverse_runtime.routers import approvals
from tests.utils.approval_review import (
    ROUTING,
    SELF_CONSISTENCY,
    WORKFLOW,
    approved_rows,
    approved_rows_until,
    review_config_manager,
    run_in_own_loop,
    save_review_batch,
)
from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

KEY = "web-ops-harness-key"


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
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
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
    _item(page, item_id).get_by_role("button", name=button).click()


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
        assert {key: value for key, value in facts.items() if key != "Example"} == {
            "Batch": batch,
            "Schema": "RoutingExperienceSchema",
            "Status": "pending_review",
            "Confidence": "0.40",
            "Created": served[routing]["created_at"],
            **(
                {"Reasoning": served[routing]["reasoning"]}
                if served[routing]["reasoning"]
                else {}
            ),
            "Self-consistency": (
                "Agreement (5 samples): gradient descent (CONCEPT) 0.60 — needs review\n"
                "Agreement (5 samples): lecture (MEDIA) 1.00"
            ),
        }
        assert served[routing]["metadata"][SELF_CONSISTENCY_METADATA_KEY] == (
            SELF_CONSISTENCY
        )
        assert json.loads(facts["Example"]) == ROUTING
        template = served[workflow]["correction_template"]
        assert (
            json.loads(
                _item(page, workflow).get_by_label("Corrections (JSON)").input_value()
            )
            == template
        )

        _decide(page, routing, "Approve")
        expect(_item(page, routing).get_by_role("alert")).to_have_text(
            "Enter your name as the reviewer first."
        )
        _queue(page, tenant).get_by_label("Reviewer").fill("reviewer@example.com")

        _decide(page, workflow, "Reject with corrections")
        expect(_item(page, workflow).get_by_role("alert")).to_have_text(
            "A rejection needs feedback."
        )
        form = _item(page, workflow)
        form.get_by_label("Feedback").fill("a report needs the report agent")
        _decide(page, workflow, "Reject with corrections")
        expect(form.get_by_role("alert")).to_have_text(
            "A WorkflowExecutionSchema rejection needs at least one correction."
        )
        expect(_item(page, routing).get_by_role("button")).to_have_text(
            ["Approve", "Reject and regenerate"]
        )
        form.get_by_label("Corrections (JSON)").fill('{"bogus": 1}')
        _decide(page, workflow, "Reject with corrections")
        expect(form.get_by_role("alert")).to_have_text(
            "WorkflowExecutionSchema unsupported correction fields: bogus"
        )
        assert {
            key: item["data"]
            for key, item in _served_queue(runtime_url, tenant).items()
        } == {routing: ROUTING, workflow: WORKFLOW}
        assert approved_rows(review["storage"], routing) == 0

        corrected = ["video_search_agent", "detailed_report_agent"]
        form.get_by_label("Corrections (JSON)").fill(
            json.dumps(dict(template, agent_sequence=corrected))
        )
        _decide(page, workflow, "Reject with corrections")
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
