"""The approval review routes against real Phoenix and Redis.

Batches are saved through the production approval store; every decision is
taken through the routes and read back from the store: the pending queue, the
replacement it serves, and the tenant's approved training dataset.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_core.approval.interfaces import ReviewDecision
from cogniverse_runtime.routers import approvals
from cogniverse_synthetic.approval.corrections import SCHEMA_CORRECTION_FIELDS
from cogniverse_synthetic.schemas import (
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)
from tests.utils.approval_review import (
    ROUTING,
    WORKFLOW,
    approved_rows,
    approved_rows_until,
    reject_without_regenerating,
    review_config_manager,
    save_review_batch,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]


@pytest.fixture
def review(phoenix_container, telemetry_manager_with_phoenix, workflow_state_redis_url):
    """A tenant with one saved batch: a routing and a workflow item awaiting
    review and one auto-approved item."""
    config_manager = review_config_manager(phoenix_container, workflow_state_redis_url)
    approvals.set_config_manager(config_manager)
    tenant_id = f"apprv{uuid4().hex[:8]}:main"
    batch_id = f"batch_{uuid4().hex[:8]}"
    storage = ApprovalStorageImpl.from_system_config(
        config_manager, telemetry_manager_with_phoenix, tenant_id
    )
    save_review_batch(storage, batch_id)
    app = FastAPI()
    app.include_router(approvals.router, prefix="/admin/tenant")
    yield {
        "app": app,
        "tenant": tenant_id,
        "batch": batch_id,
        "storage": storage,
        "config_manager": config_manager,
    }
    approvals.set_config_manager(None)


async def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=120,
    )


async def _pending_until(client, tenant, want, timeout=60.0):
    """The pending item IDs once they equal ``want`` (Phoenix serves spans
    and annotations after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(f"/admin/tenant/{tenant}/approvals")
        assert response.status_code == 200, response.text
        items = {item["item_id"]: item for item in response.json()["items"]}
        if set(items) == want or time.monotonic() > deadline:
            return items
        await asyncio.sleep(2)


async def test_pending_items_carry_their_schema_and_correctable_fields(review):
    batch = review["batch"]
    async with await _client(review["app"]) as client:
        items = await _pending_until(
            client, review["tenant"], {f"{batch}_routing", f"{batch}_workflow"}
        )

    routing, workflow = items[f"{batch}_routing"], items[f"{batch}_workflow"]
    assert {
        key: routing[key]
        for key in ("batch_id", "status", "confidence", "schema_name", "data")
    } == {
        "batch_id": batch,
        "status": "pending_review",
        "confidence": 0.4,
        "schema_name": "RoutingExperienceSchema",
        "data": ROUTING,
    }
    assert routing["correction_template"] == {
        "entities": ROUTING["entities"],
        "relationships": [],
        "chosen_agent": "video_search_agent",
    }
    assert set(routing["correction_template"]) == set(
        SCHEMA_CORRECTION_FIELDS[RoutingExperienceSchema]
    )
    assert (workflow["schema_name"], workflow["correction_template"]) == (
        "WorkflowExecutionSchema",
        {
            field: WORKFLOW[field]
            for field in SCHEMA_CORRECTION_FIELDS[WorkflowExecutionSchema]
            if field in WORKFLOW
        },
    )
    assert (routing["corrections_required"], workflow["corrections_required"]) == (
        False,
        True,
    )


async def test_an_approval_leaves_the_queue_for_the_training_dataset(review):
    batch, tenant = review["batch"], review["tenant"]
    item_id = f"{batch}_routing"
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {item_id, f"{batch}_workflow"})
        response = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{item_id}",
            json={"approved": True, "reviewer": "reviewer@example.com"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["status"], body["item"]["item_id"], body["item"]["status"]) == (
            "approved",
            item_id,
            "approved",
        )
        remaining = await _pending_until(client, tenant, {f"{batch}_workflow"})
    assert set(remaining) == {f"{batch}_workflow"}
    assert (
        await asyncio.to_thread(approved_rows_until, review["storage"], item_id, 1) == 1
    )


async def test_a_rejection_with_corrections_queues_the_corrected_replacement(review):
    batch, tenant = review["batch"], review["tenant"]
    item_id = f"{batch}_workflow"
    corrected = ["video_search_agent", "detailed_report_agent"]
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {item_id, f"{batch}_routing"})
        response = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{item_id}",
            json={
                "approved": False,
                "reviewer": "reviewer@example.com",
                "feedback": "a report needs the report agent",
                "corrections": {"agent_sequence": corrected},
            },
        )
        assert response.status_code == 200, response.text
        replacement = response.json()["item"]
        assert response.json()["status"] == "regenerated"
        assert (
            replacement["status"],
            replacement["data"],
            replacement["metadata"]["original_item_id"],
            replacement["metadata"]["regeneration_feedback"],
        ) == (
            "regenerated",
            dict(WORKFLOW, agent_sequence=corrected),
            item_id,
            "a report needs the report agent",
        )
        items = await _pending_until(
            client, tenant, {replacement["item_id"], f"{batch}_routing"}
        )
    assert set(items) == {replacement["item_id"], f"{batch}_routing"}
    assert items[replacement["item_id"]]["data"]["agent_sequence"] == corrected


async def test_a_decision_the_item_cannot_take_is_refused_and_changes_nothing(
    review,
):
    batch, tenant = review["batch"], review["tenant"]
    workflow = f"{batch}_workflow"
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {workflow, f"{batch}_routing"})
        url = f"/admin/tenant/{tenant}/approvals/{batch}"
        no_feedback = await client.post(
            f"{url}/{batch}_routing",
            json={
                "approved": False,
                "reviewer": "r",
                "corrections": {"chosen_agent": "summarizer_agent"},
            },
        )
        bad_field = await client.post(
            f"{url}/{workflow}",
            json={
                "approved": False,
                "reviewer": "r",
                "feedback": "wrong",
                "corrections": {"bogus": 1},
            },
        )
        no_corrections = await client.post(
            f"{url}/{workflow}",
            json={"approved": False, "reviewer": "r", "feedback": "wrong"},
        )
        unknown = await client.post(
            f"{url}/{batch}_missing", json={"approved": True, "reviewer": "r"}
        )
        auto = await client.post(
            f"{url}/{batch}_confident", json={"approved": True, "reviewer": "r"}
        )
        after = await _pending_until(client, tenant, {workflow, f"{batch}_routing"})

    assert (no_feedback.status_code, no_feedback.json()) == (
        400,
        {"detail": "Regenerating a RoutingExperienceSchema item needs feedback."},
    )
    assert (bad_field.status_code, bad_field.json()) == (
        400,
        {"detail": "WorkflowExecutionSchema unsupported correction fields: bogus"},
    )
    assert (no_corrections.status_code, no_corrections.json()) == (
        400,
        {
            "detail": "A WorkflowExecutionSchema rejection needs at least one correction."
        },
    )
    for response, item in ((unknown, "missing"), (auto, "confident")):
        assert (response.status_code, response.json()) == (
            404,
            {"detail": f"Item {batch}_{item} of batch {batch} is not awaiting review."},
        )
    assert {key: item["data"] for key, item in after.items()} == {
        workflow: WORKFLOW,
        f"{batch}_routing": ROUTING,
    }


async def test_two_reviewers_approving_one_item_at_once_add_one_training_row(
    review,
):
    batch, tenant = review["batch"], review["tenant"]
    item_id = f"{batch}_routing"
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {item_id, f"{batch}_workflow"})
        responses = await asyncio.gather(
            *(
                client.post(
                    f"/admin/tenant/{tenant}/approvals/{batch}/{item_id}",
                    json={"approved": True, "reviewer": reviewer},
                )
                for reviewer in ("first@example.com", "second@example.com")
            )
        )
        await _pending_until(client, tenant, {f"{batch}_workflow"})
    first, late = sorted(responses, key=lambda response: response.status_code)
    assert (first.status_code, first.json()["status"]) == (200, "approved")
    # Redis elects one decision: the later reviewer either loses that election
    # or finds the item already decided.
    late_detail = late.json()["detail"]
    if late.status_code == 409:
        assert {key: late_detail[key] for key in ("error", "message", "tenant_id")} == {
            "error": "approval_decision_conflict",
            "message": f"Item {item_id} was already decided by another reviewer.",
            "tenant_id": tenant,
        }
    else:
        assert (late.status_code, late_detail) == (
            404,
            f"Item {item_id} of batch {batch} is not awaiting review.",
        )
    assert (
        await asyncio.to_thread(approved_rows_until, review["storage"], item_id, 1) == 1
    )


async def test_a_decision_redis_already_elected_for_another_reviewer_is_a_conflict(
    review,
):
    batch, tenant = review["batch"], review["tenant"]
    item_id = f"{batch}_routing"
    await review["storage"].select_review_decision(
        batch_id=batch,
        original_item_id=item_id,
        decision=ReviewDecision(
            item_id=item_id, approved=False, feedback="wrong agent", reviewer="rival"
        ),
    )
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {item_id, f"{batch}_workflow"})
        response = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{item_id}",
            json={"approved": True, "reviewer": "late@example.com"},
        )
        after = await _pending_until(client, tenant, {item_id, f"{batch}_workflow"})
    detail = response.json()["detail"]
    assert (
        response.status_code,
        {key: detail[key] for key in ("error", "message", "tenant_id")},
    ) == (
        409,
        {
            "error": "approval_decision_conflict",
            "message": f"Item {item_id} was already decided by another reviewer.",
            "tenant_id": tenant,
        },
    )
    assert after[item_id]["status"] == "pending_review"
    assert await asyncio.to_thread(approved_rows, review["storage"], item_id) == 0


async def test_an_unreachable_store_reads_as_an_outage_not_an_empty_queue(
    phoenix_container, telemetry_manager_with_phoenix, workflow_state_redis_url
):
    approvals.set_config_manager(
        review_config_manager(
            phoenix_container,
            workflow_state_redis_url,
            telemetry_url="http://127.0.0.1:9",
        )
    )
    app = FastAPI()
    app.include_router(approvals.router, prefix="/admin/tenant")
    tenant = f"apprv{uuid4().hex[:8]}:main"
    try:
        async with await _client(app) as client:
            response = await client.get(f"/admin/tenant/{tenant}/approvals")
    finally:
        approvals.set_config_manager(None)
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert {key: detail[key] for key in ("error", "message", "tenant_id")} == {
        "error": "approval_store_unavailable",
        "message": f"Could not read the items awaiting review for tenant {tenant}.",
        "tenant_id": tenant,
    }


CORRECTED = ["video_search_agent", "detailed_report_agent"]


async def _get_until(client, path, done, timeout=60.0):
    """The JSON body of ``path`` once ``done(body)`` holds (Phoenix serves
    spans and annotations after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(path)
        assert response.status_code == 200, response.text
        body = response.json()
        if done(body) or time.monotonic() > deadline:
            return body
        await asyncio.sleep(2)


def _ids(entries):
    return [entry["item_id"] for entry in entries]


async def test_history_and_stats_report_every_decision(review):
    batch, tenant = review["batch"], review["tenant"]
    routing, workflow = f"{batch}_routing", f"{batch}_workflow"
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {routing, workflow})
        approved = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{routing}",
            json={"approved": True, "reviewer": "first@example.com"},
        )
        assert approved.status_code == 200, approved.text
        # A correction-only rejection takes no feedback.
        rejected = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{workflow}",
            json={
                "approved": False,
                "reviewer": "second@example.com",
                "corrections": {"agent_sequence": CORRECTED},
            },
        )
        assert rejected.status_code == 200, rejected.text
        replacement = rejected.json()["item"]
        assert (
            rejected.json()["status"],
            replacement["metadata"]["regeneration_feedback"],
        ) == ("regenerated", "")
        history = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: (
                _ids(body["approved"]) == [routing, f"{batch}_confident"]
                and _ids(body["rejected"]) == [workflow]
            ),
        )
        stats = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/stats",
            lambda body: body["total"] == 4 and body["approved"] == 1,
        )
        pending = await _pending_until(client, tenant, {replacement["item_id"]})

    approved_entry, auto_entry = history["approved"]
    assert {
        key: approved_entry[key]
        for key in (
            "batch_id",
            "status",
            "confidence",
            "query",
            "schema_name",
            "reviewer",
            "feedback",
            "corrections",
            "replacement_id",
        )
    } == {
        "batch_id": batch,
        "status": "approved",
        "confidence": 0.4,
        "query": ROUTING["query"],
        "schema_name": "RoutingExperienceSchema",
        "reviewer": "first@example.com",
        "feedback": None,
        "corrections": {},
        "replacement_id": None,
    }
    assert (
        approved_entry["reviewed_at"]
        == approved.json()["item"]["metadata"]["decision"]["timestamp"]
    )
    assert (
        auto_entry["status"],
        auto_entry["query"],
        auto_entry["confidence"],
        auto_entry["reviewer"],
    ) == ("auto_approved", "play the intro clip", 0.95, None)
    (rejected_entry,) = history["rejected"]
    assert (
        rejected_entry["reviewed_at"]
        == rejected.json()["item"]["metadata"]["decision"]["timestamp"]
    )
    assert {
        key: rejected_entry[key]
        for key in (
            "status",
            "query",
            "data",
            "schema_name",
            "reviewer",
            "feedback",
            "corrections",
            "replacement_id",
            "replacement_status",
        )
    } == {
        "status": "rejected",
        "query": WORKFLOW["query"],
        "data": WORKFLOW,
        "schema_name": "WorkflowExecutionSchema",
        "reviewer": "second@example.com",
        "feedback": "",
        "corrections": {"agent_sequence": CORRECTED},
        "replacement_id": replacement["item_id"],
        "replacement_status": "regenerated",
    }
    replacement_confidence = pending[replacement["item_id"]]["confidence"]
    assert stats == {
        "total": 4,
        "pending": 1,
        "auto_approved": 1,
        "approved": 1,
        "rejected": 1,
        "approval_rate": 0.5,
        "average_confidence": {
            "pending": replacement_confidence,
            "auto_approved": 0.95,
            "approved": 0.4,
            "rejected": 0.3,
        },
    }


async def test_the_rejected_list_puts_the_latest_decision_first(review):
    """An item rejected and replaced was reviewed when its rejection was
    decided, so it sorts by that time against items rejected outright."""
    batch, tenant, storage = review["batch"], review["tenant"], review["storage"]
    routing, workflow = f"{batch}_routing", f"{batch}_workflow"
    await reject_without_regenerating(
        storage,
        batch,
        routing,
        feedback="the gateway chose the wrong agent",
        reviewer="first@example.com",
    )
    async with await _client(review["app"]) as client:
        rejected = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{workflow}",
            json={
                "approved": False,
                "reviewer": "second@example.com",
                "corrections": {"agent_sequence": CORRECTED},
            },
        )
        assert rejected.status_code == 200, rejected.text
        history = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: set(_ids(body["rejected"])) == {routing, workflow},
        )

    first, second = history["rejected"]
    decided = rejected.json()["item"]["metadata"]["decision"]["timestamp"]
    assert (first["item_id"], first["reviewed_at"], second["item_id"]) == (
        workflow,
        decided,
        routing,
    )
    assert second["reviewed_at"] < decided


async def test_two_reviewers_rejecting_at_once_leave_the_elected_decision_time(
    review,
):
    """Of two rejections at once one decision is elected; the rejected
    original reads that decision's reviewer and time, never the loser's."""
    batch, tenant = review["batch"], review["tenant"]
    workflow = f"{batch}_workflow"
    url = f"/admin/tenant/{tenant}/approvals/{batch}/{workflow}"
    async with await _client(review["app"]) as client:
        await _pending_until(client, tenant, {f"{batch}_routing", workflow})
        responses = await asyncio.gather(
            *(
                client.post(
                    url,
                    json={
                        "approved": False,
                        "reviewer": reviewer,
                        "corrections": {"agent_sequence": sequence},
                    },
                )
                for reviewer, sequence in (
                    ("first@example.com", CORRECTED),
                    ("second@example.com", ["video_search_agent"]),
                )
            )
        )
        [won] = [r for r in responses if r.status_code == 200]
        [lost] = [r for r in responses if r.status_code != 200]
        decision = won.json()["item"]["metadata"]["decision"]
        history = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: _ids(body["rejected"]) == [workflow],
        )

    assert (lost.status_code, lost.json()["detail"]["error"]) == (
        409,
        "approval_decision_conflict",
    )
    (entry,) = history["rejected"]
    assert (
        entry["reviewer"],
        entry["reviewed_at"],
        entry["replacement_id"],
    ) == (decision["reviewer"], decision["timestamp"], won.json()["item"]["item_id"])


async def test_a_rejected_item_nothing_replaced_is_regenerated_once(review):
    batch, tenant, storage = review["batch"], review["tenant"], review["storage"]
    routing, workflow = f"{batch}_routing", f"{batch}_workflow"
    await reject_without_regenerating(
        storage,
        batch,
        workflow,
        feedback="a report needs the report agent",
        corrections={"agent_sequence": CORRECTED},
        reviewer="reviewer@example.com",
    )
    url = f"/admin/tenant/{tenant}/approvals/{batch}"
    async with await _client(review["app"]) as client:
        history = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: _ids(body["rejected"]) == [workflow],
        )
        assert (
            history["rejected"][0]["replacement_id"],
            history["rejected"][0]["feedback"],
            history["rejected"][0]["corrections"],
        ) == (None, "a report needs the report agent", {"agent_sequence": CORRECTED})
        await _pending_until(client, tenant, {routing})

        pending_item = await client.post(f"{url}/{routing}/regenerate")
        unknown = await client.post(f"{url}/{batch}_missing/regenerate")
        regenerated = await client.post(f"{url}/{workflow}/regenerate")
        assert regenerated.status_code == 200, regenerated.text
        replacement = regenerated.json()["item"]
        again = await client.post(f"{url}/{workflow}/regenerate")
        pending = await _pending_until(
            client, tenant, {routing, replacement["item_id"]}
        )
        history = await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: (
                body["rejected"][0]["replacement_id"] == replacement["item_id"]
            ),
        )

    assert (pending_item.status_code, pending_item.json()) == (
        409,
        {"detail": f"Item {routing} is pending_review, not rejected."},
    )
    assert (unknown.status_code, unknown.json()) == (
        404,
        {"detail": f"Item {batch}_missing of batch {batch} was not found."},
    )
    assert (
        regenerated.json()["status"],
        replacement["status"],
        replacement["data"],
        replacement["metadata"]["original_item_id"],
        replacement["metadata"]["regeneration_feedback"],
    ) == (
        "regenerated",
        "regenerated",
        dict(WORKFLOW, agent_sequence=CORRECTED),
        workflow,
        "a report needs the report agent",
    )
    assert (again.status_code, again.json()) == (
        409,
        {
            "detail": f"Item {workflow} was already regenerated as "
            f"{replacement['item_id']}."
        },
    )
    assert pending[replacement["item_id"]]["data"] == dict(
        WORKFLOW, agent_sequence=CORRECTED
    )
    assert (
        history["rejected"][0]["replacement_status"],
        history["rejected"][0]["reviewer"],
    ) == ("regenerated", "reviewer@example.com")


async def test_a_rejection_that_recorded_no_corrections_is_not_regenerated(review):
    batch, tenant, storage = review["batch"], review["tenant"], review["storage"]
    workflow = f"{batch}_workflow"
    await reject_without_regenerating(
        storage, batch, workflow, feedback="wrong", reviewer="r"
    )
    async with await _client(review["app"]) as client:
        await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: _ids(body["rejected"]) == [workflow],
        )
        response = await client.post(
            f"/admin/tenant/{tenant}/approvals/{batch}/{workflow}/regenerate"
        )
        batch_after = await storage.get_batch(batch)
    assert (response.status_code, response.json()) == (
        400,
        {
            "detail": f"Item {workflow} was rejected without corrections, which a "
            "WorkflowExecutionSchema needs to be regenerated."
        },
    )
    assert [
        item.item_id
        for item in batch_after.items
        if item.metadata.get("original_item_id") == workflow
    ] == []


async def test_two_regenerations_of_one_item_at_once_elect_one_replacement(review):
    batch, tenant, storage = review["batch"], review["tenant"], review["storage"]
    workflow = f"{batch}_workflow"
    await reject_without_regenerating(
        storage,
        batch,
        workflow,
        feedback="wrong agents",
        corrections={"agent_sequence": CORRECTED},
        reviewer="r",
    )
    url = f"/admin/tenant/{tenant}/approvals/{batch}/{workflow}/regenerate"
    async with await _client(review["app"]) as client:
        await _get_until(
            client,
            f"/admin/tenant/{tenant}/approvals/history",
            lambda body: _ids(body["rejected"]) == [workflow],
        )
        responses = await asyncio.gather(client.post(url), client.post(url))
    batch_after = await storage.get_batch(batch)
    (replacement,) = [
        item.item_id
        for item in batch_after.items
        if item.metadata.get("original_item_id") == workflow
    ]
    # Both requests answer the one elected replacement, or the later one
    # finds the item already regenerated.
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {
                "detail": f"Item {workflow} was already regenerated as {replacement}."
            }
        else:
            assert (response.status_code, response.json()["item"]["item_id"]) == (
                200,
                replacement,
            )
    assert sorted(response.status_code for response in responses)[0] == 200


async def test_an_unreachable_store_fails_history_and_stats_not_empties_them(
    phoenix_container, telemetry_manager_with_phoenix, workflow_state_redis_url
):
    approvals.set_config_manager(
        review_config_manager(
            phoenix_container,
            workflow_state_redis_url,
            telemetry_url="http://127.0.0.1:9",
        )
    )
    app = FastAPI()
    app.include_router(approvals.router, prefix="/admin/tenant")
    tenant = f"apprv{uuid4().hex[:8]}:main"
    try:
        async with await _client(app) as client:
            responses = [
                await client.get(f"/admin/tenant/{tenant}/approvals/{path}")
                for path in ("history", "stats")
            ]
    finally:
        approvals.set_config_manager(None)
    for response in responses:
        detail = response.json()["detail"]
        assert (
            response.status_code,
            {key: detail[key] for key in ("error", "message", "tenant_id")},
        ) == (
            502,
            {
                "error": "approval_store_unavailable",
                "message": f"Could not read the review history of tenant {tenant}.",
                "tenant_id": tenant,
            },
        )
