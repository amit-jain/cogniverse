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
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from cogniverse_runtime.routers import approvals
from cogniverse_synthetic.approval.corrections import SCHEMA_CORRECTION_FIELDS
from cogniverse_synthetic.schemas import (
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

ROUTING = {
    "query": "find the lecture on gradient descent",
    "entities": [{"text": "gradient descent", "type": "TOPIC"}],
    "relationships": [],
    "enhanced_query": "find the lecture video on gradient descent",
    "chosen_agent": "video_search_agent",
    "routing_confidence": 0.4,
    "search_quality": 0.0,
    "agent_success": False,
    "processing_time": 0.0,
    "metadata": {},
}
WORKFLOW = {
    "workflow_id": "wf-review-1",
    "query": "summarize the lecture and write a report",
    "query_type": "video",
    "execution_time": 12.5,
    "success": True,
    "agent_sequence": ["video_search_agent", "summarizer_agent"],
    "task_count": 2,
    "parallel_efficiency": 0.5,
    "confidence_score": 0.3,
    "metadata": {},
}


def _config_manager(phoenix, redis_url, *, telemetry_url=None) -> ConfigManager:
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        SystemConfig(
            telemetry_url=telemetry_url or phoenix["http_endpoint"],
            telemetry_collector_endpoint=phoenix["grpc_endpoint"],
            redis_url=redis_url,
        )
    )
    return manager


@pytest.fixture
def review(phoenix_container, telemetry_manager_with_phoenix, workflow_state_redis_url):
    """A tenant with one saved batch: a routing and a workflow item awaiting
    review and one auto-approved item."""
    config_manager = _config_manager(phoenix_container, workflow_state_redis_url)
    approvals.set_config_manager(config_manager)
    tenant_id = f"apprv{uuid4().hex[:8]}:main"
    batch_id = f"batch_{uuid4().hex[:8]}"
    storage = ApprovalStorageImpl.from_system_config(
        config_manager, telemetry_manager_with_phoenix, tenant_id
    )
    items = [
        ReviewItem(
            item_id=f"{batch_id}_routing",
            data=dict(ROUTING),
            metadata={"agent_type": "routing"},
            confidence=0.4,
            status=ApprovalStatus.PENDING_REVIEW,
        ),
        ReviewItem(
            item_id=f"{batch_id}_workflow",
            data=dict(WORKFLOW),
            metadata={"agent_type": "workflow"},
            confidence=0.3,
            status=ApprovalStatus.PENDING_REVIEW,
        ),
        ReviewItem(
            item_id=f"{batch_id}_confident",
            data=dict(ROUTING, query="play the intro clip"),
            metadata={"agent_type": "routing"},
            confidence=0.95,
            status=ApprovalStatus.AUTO_APPROVED,
        ),
    ]
    asyncio.run(
        storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=items,
                context={"tenant_id": tenant_id, "optimizer": "routing"},
            )
        )
    )
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


def _approved_rows(storage, item_id) -> int:
    try:
        frame = asyncio.run(
            storage.provider.datasets.get_dataset(
                approved_synthetic_dataset_name(storage.tenant_id)
            )
        )
    except DatasetNotFoundError:
        return 0
    return sum(
        1
        for _, row in frame.iterrows()
        if any(
            isinstance(row[column], dict) and row[column].get("item_id") == item_id
            for column in ("input", "output", "metadata")
            if column in frame.columns
        )
    )


def _approved_rows_until(storage, item_id, want, timeout=30.0) -> int:
    deadline = time.monotonic() + timeout
    while (count := _approved_rows(storage, item_id)) != want:
        if time.monotonic() > deadline:
            return count
        time.sleep(2)
    return count


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
        await asyncio.to_thread(_approved_rows_until, review["storage"], item_id, 1)
        == 1
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
            f"{url}/{workflow}",
            json={"approved": False, "reviewer": "r", "corrections": {"task_count": 3}},
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
        unknown = await client.post(
            f"{url}/{batch}_missing", json={"approved": True, "reviewer": "r"}
        )
        auto = await client.post(
            f"{url}/{batch}_confident", json={"approved": True, "reviewer": "r"}
        )
        after = await _pending_until(client, tenant, {workflow, f"{batch}_routing"})

    assert (no_feedback.status_code, no_feedback.json()) == (
        400,
        {"detail": "A rejection needs feedback."},
    )
    assert (bad_field.status_code, bad_field.json()) == (
        400,
        {"detail": "WorkflowExecutionSchema unsupported correction fields: bogus"},
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
        await asyncio.to_thread(_approved_rows_until, review["storage"], item_id, 1)
        == 1
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
    assert await asyncio.to_thread(_approved_rows, review["storage"], item_id) == 0


async def test_an_unreachable_store_reads_as_an_outage_not_an_empty_queue(
    phoenix_container, telemetry_manager_with_phoenix, workflow_state_redis_url
):
    approvals.set_config_manager(
        _config_manager(
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
