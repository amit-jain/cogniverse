"""DecisionOrchestrator approval flow against real Phoenix and Redis."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.orchestrator import DecisionOrchestrator
from cogniverse_agents.workflow.state_machine import WorkflowState
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ConfidenceExtractor,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.no_shared_memory_vespa,
]


class _ConfidenceFieldExtractor(ConfidenceExtractor):
    def extract(self, data: dict) -> float:
        return float(data["confidence"])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def approval_redis_url():
    port = _free_port()
    container_name = f"cogniverse-orchestrator-redis-{os.getpid()}-{port}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
            timeout=10,
        )
        pytest.fail("Redis did not become ready within 30 seconds")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
            timeout=10,
        )


async def test_tenant_survives_async_generation_human_approval_and_reload(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
):
    tenant_alias = f"orchestrator{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    workflow_id = f"wf-{uuid4().hex}"
    batch_id = f"{workflow_id}_step_0"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    agent = HumanApprovalAgent(
        confidence_extractor=_ConfidenceFieldExtractor(),
        confidence_threshold=0.85,
        storage=storage,
    )
    orchestrator = DecisionOrchestrator(
        approval_agent=agent,
        workflow_id=workflow_id,
        initial_context={
            "tenant_id": tenant_alias,
            "agent_type": "routing",
            "request_id": "request-42",
        },
    )

    async def generate(context):
        return [
            {
                "query": "find the recorded product launch",
                "chosen_agent": "video_search_agent",
                "confidence": 0.25,
                "generation_tenant": context["tenant_id"],
            }
        ]

    orchestrator.register_step("generate", generate, requires_approval=True)

    first_context = await orchestrator.execute()

    expected_batch_context = {
        "agent_type": "routing",
        "tenant_id": tenant_id,
        "workflow_id": workflow_id,
        "step_name": "generate",
        "step_index": 0,
    }
    assert orchestrator.state_machine.current_state is WorkflowState.AWAITING_APPROVAL
    assert first_context["tenant_id"] == tenant_id
    assert first_context["current_batch"].context == expected_batch_context

    persisted_pending = await storage.get_batch(batch_id)
    assert persisted_pending is not None
    assert persisted_pending.context == expected_batch_context
    assert [
        (
            item.item_id,
            item.data,
            item.confidence,
            item.status,
            item.metadata["agent_type"],
            item.metadata["approval_batch_id"],
        )
        for item in persisted_pending.items
    ] == [
        (
            f"{batch_id}_0",
            {
                "query": "find the recorded product launch",
                "chosen_agent": "video_search_agent",
                "confidence": 0.25,
                "generation_tenant": tenant_id,
            },
            0.25,
            ApprovalStatus.PENDING_REVIEW,
            "routing",
            batch_id,
        )
    ]

    reviewed_at = datetime(2026, 8, 5, 4, 5, 6, tzinfo=timezone.utc)
    await orchestrator.apply_approvals(
        [
            ReviewDecision(
                item_id=f"{batch_id}_0",
                approved=True,
                feedback="The route targets the exact video-search capability.",
                reviewer="reviewer@example.com",
                timestamp=reviewed_at,
            )
        ]
    )
    final_context = await orchestrator.execute()

    assert orchestrator.state_machine.current_state is WorkflowState.COMPLETED
    assert final_context["tenant_id"] == tenant_id
    assert final_context["current_batch"].context == expected_batch_context
    assert final_context["current_batch"].items[0].status is ApprovalStatus.APPROVED
    assert final_context["current_batch"].items[0].reviewed_at == reviewed_at

    dataset_name = approved_synthetic_dataset_name(tenant_id)
    dataset = await storage.provider.datasets.get_dataset(name=dataset_name)
    assert len(dataset) == 1
    record = dataset.iloc[0]["input"]
    assert record["item_id"] == f"{batch_id}_0"
    assert record["query"] == "find the recorded product launch"
    assert record["chosen_agent"] == "video_search_agent"
    assert record["status"] == "approved"
    assert record["reviewed_at"] == reviewed_at.isoformat()
    assert record["metadata.agent_type"] == "routing"
    assert record["metadata.approval_batch_id"] == batch_id
    assert record["context.tenant_id"] == tenant_id
    assert record["context.workflow_id"] == workflow_id
    assert record["context.step_name"] == "generate"
    assert record["context.step_index"] == 0


@pytest.mark.parametrize(
    "confidence",
    [float("nan"), float("inf"), -0.01, 1.01],
    ids=["nan", "infinity", "below-zero", "above-one"],
)
async def test_orchestrator_rejects_invalid_extracted_confidence_before_persistence(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
    confidence,
):
    tenant_alias = f"invalidconfidence{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    workflow_id = f"wf-{uuid4().hex}"
    batch_id = f"{workflow_id}_step_0"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    orchestrator = DecisionOrchestrator(
        approval_agent=HumanApprovalAgent(
            confidence_extractor=_ConfidenceFieldExtractor(),
            confidence_threshold=0.85,
            storage=storage,
        ),
        workflow_id=workflow_id,
        initial_context={"tenant_id": tenant_alias, "agent_type": "routing"},
    )
    orchestrator.register_step(
        "generate",
        lambda _context: [
            {
                "query": "find the recorded product launch",
                "chosen_agent": "video_search_agent",
                "confidence": confidence,
            }
        ],
        requires_approval=True,
    )

    context = await orchestrator.execute()

    assert orchestrator.state_machine.current_state is WorkflowState.FAILED
    assert "current_batch" not in context
    assert await storage.get_batch(batch_id) is None
    with pytest.raises(DatasetNotFoundError):
        await storage.provider.datasets.get_dataset(
            name=approved_synthetic_dataset_name(tenant_id)
        )
    assert orchestrator.state_machine.history[-1].transition_reason == (
        "forced: "
        f"Review item '{batch_id}_0' confidence must be a finite number in [0, 1]"
    )


async def test_regenerated_item_is_approved_without_rerunning_generation(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
):
    tenant_alias = f"regeneratedresume{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    workflow_id = f"wf-{uuid4().hex}"
    batch_id = f"{workflow_id}_step_0"
    calls = []

    class FeedbackHandler:
        async def process_rejection(self, item, decision):
            assert item.item_id == f"{batch_id}_0"
            assert item.data == {
                "query": "find launch footage",
                "chosen_agent": "video_search_agent",
                "confidence": 0.2,
            }
            assert decision.feedback == "Use the exact product-launch wording."
            return ReviewItem(
                item_id=f"{item.item_id}_regenerated",
                data={
                    "query": "find the exact product launch recording",
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.65,
                },
                confidence=0.65,
            )

    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    orchestrator = DecisionOrchestrator(
        approval_agent=HumanApprovalAgent(
            confidence_extractor=_ConfidenceFieldExtractor(),
            feedback_handler=FeedbackHandler(),
            confidence_threshold=0.85,
            storage=storage,
        ),
        workflow_id=workflow_id,
        initial_context={"tenant_id": tenant_alias, "agent_type": "routing"},
    )

    def generate(_context):
        calls.append("generate")
        return [
            {
                "query": "find launch footage",
                "chosen_agent": "video_search_agent",
                "confidence": 0.2,
            }
        ]

    def finalize(context):
        calls.append("finalize")
        return {"approved_query": context["current_batch"].approved[0].data["query"]}

    orchestrator.register_step("generate", generate, requires_approval=True)
    orchestrator.register_step("finalize", finalize)

    await orchestrator.execute()
    original_id = f"{batch_id}_0"
    replacement_id = f"{original_id}_regenerated"
    rejected_at = datetime(2026, 8, 5, 10, 11, 12, tzinfo=timezone.utc)
    await orchestrator.apply_approvals(
        [
            ReviewDecision(
                item_id=original_id,
                approved=False,
                feedback="Use the exact product-launch wording.",
                reviewer="reviewer@example.com",
                timestamp=rejected_at,
            )
        ]
    )

    awaiting_replacement = orchestrator.state_machine.context["current_batch"]
    assert orchestrator.state_machine.current_state is WorkflowState.AWAITING_APPROVAL
    assert orchestrator.current_step_index == 0
    assert calls == ["generate"]
    assert orchestrator.state_machine.context["pending_review_count"] == 1
    assert orchestrator.state_machine.context["rejection_count"] == 0
    assert [
        (
            item.item_id,
            item.status,
            item.data["query"],
            item.metadata.get("original_item_id"),
        )
        for item in awaiting_replacement.items
    ] == [
        (
            original_id,
            ApprovalStatus.REJECTED,
            "find launch footage",
            None,
        ),
        (
            replacement_id,
            ApprovalStatus.REGENERATED,
            "find the exact product launch recording",
            original_id,
        ),
    ]

    await orchestrator.execute()
    assert calls == ["generate"]

    approved_at = datetime(2026, 8, 5, 10, 12, 13, tzinfo=timezone.utc)
    await orchestrator.apply_approvals(
        [
            ReviewDecision(
                item_id=replacement_id,
                approved=True,
                feedback="The corrected query names the exact recording.",
                reviewer="reviewer@example.com",
                timestamp=approved_at,
            )
        ]
    )
    final_context = await orchestrator.execute()

    assert calls == ["generate", "finalize"]
    assert orchestrator.current_step_index == 2
    assert orchestrator.state_machine.current_state is WorkflowState.COMPLETED
    assert final_context["step_finalize_result"] == {
        "approved_query": "find the exact product launch recording"
    }
    assert [
        (item.item_id, item.status, item.reviewed_at)
        for item in final_context["current_batch"].items
    ] == [
        (original_id, ApprovalStatus.REJECTED, None),
        (replacement_id, ApprovalStatus.APPROVED, approved_at),
    ]

    dataset = await storage.provider.datasets.get_dataset(
        name=approved_synthetic_dataset_name(tenant_id)
    )
    assert [
        (row["item_id"], row["query"], row["status"], row["reviewed_at"])
        for row in dataset["input"]
    ] == [
        (
            replacement_id,
            "find the exact product launch recording",
            "approved",
            approved_at.isoformat(),
        )
    ]
    assert await storage.get_pending_batches() == []


async def test_approve_and_regenerate_race_selects_one_consistent_decision(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
):
    tenant_alias = f"decisionrace{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    batch_id = f"batch-{uuid4().hex}"
    item = ReviewItem(
        item_id=f"{batch_id}-item",
        data={
            "query": "find the recorded product launch",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.25,
        metadata={"agent_type": "routing"},
    )
    batch = ApprovalBatch(
        batch_id=batch_id,
        items=[item],
        context={"tenant_id": tenant_id, "agent_type": "routing"},
    )
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    await storage.save_batch(batch)

    approved_at = datetime(2026, 8, 5, 4, 5, 6, tzinfo=timezone.utc)
    rejected_at = datetime(2026, 8, 5, 4, 5, 7, tzinfo=timezone.utc)
    approval = ReviewDecision(
        item_id=item.item_id,
        approved=True,
        feedback="The route is exact.",
        reviewer="approver@example.com",
        timestamp=approved_at,
    )
    replacement = ReviewItem(
        item_id=f"{item.item_id}-regenerated",
        data={
            "query": "find the exact launch recording",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.4,
        status=ApprovalStatus.REGENERATED,
        metadata={
            "agent_type": "routing",
            "original_item_id": item.item_id,
            "decision": {
                "reviewer": "rejector@example.com",
                "feedback": "Use the exact launch wording.",
                "corrections": {"query": "find the exact launch recording"},
                "timestamp": rejected_at.isoformat(),
            },
        },
    )
    start = asyncio.Barrier(2)

    async def approve():
        await start.wait()
        return await storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=approved_synthetic_dataset_name(tenant_id),
            item=item,
            decision=approval,
            project_context=batch.context,
        )

    async def regenerate():
        await start.wait()
        return await storage.replace_item(batch_id, item, replacement)

    approval_result, replacement_result = await asyncio.gather(
        approve(),
        regenerate(),
        return_exceptions=True,
    )
    successes = [
        result
        for result in (approval_result, replacement_result)
        if not isinstance(result, BaseException)
    ]
    failures = [
        result
        for result in (approval_result, replacement_result)
        if isinstance(result, BaseException)
    ]
    assert len(successes) == 1, (
        f"approval={approval_result!r} replacement={replacement_result!r}"
    )
    assert len(failures) == 1, (
        f"approval={approval_result!r} replacement={replacement_result!r}"
    )
    failure_messages = []
    failure = failures[0]
    while failure is not None:
        failure_messages.append(str(failure))
        failure = failure.__cause__
    assert any(
        "conflicts with canonical review decision" in message
        for message in failure_messages
    )

    persisted = await storage.get_batch(batch_id)
    assert persisted is not None
    dataset_name = approved_synthetic_dataset_name(tenant_id)
    if not isinstance(approval_result, BaseException):
        assert [
            (review_item.item_id, review_item.status) for review_item in persisted.items
        ] == [(item.item_id, ApprovalStatus.APPROVED)]
        dataset = await storage.provider.datasets.get_dataset(name=dataset_name)
        assert [row["item_id"] for row in dataset["input"]] == [item.item_id]
    else:
        assert [
            (
                review_item.item_id,
                review_item.status,
                review_item.metadata.get("original_item_id"),
            )
            for review_item in persisted.items
        ] == [
            (item.item_id, ApprovalStatus.REJECTED, None),
            (replacement.item_id, ApprovalStatus.REGENERATED, item.item_id),
        ]
        with pytest.raises(DatasetNotFoundError):
            await storage.provider.datasets.get_dataset(name=dataset_name)


async def test_dataset_failure_retry_persists_redis_selected_decision_timestamp(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
    monkeypatch,
):
    tenant_alias = f"decisionretry{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    batch_id = f"batch-{uuid4().hex}"
    dataset_name = approved_synthetic_dataset_name(tenant_id)
    item = ReviewItem(
        item_id=f"{batch_id}-item",
        data={
            "query": "find the recorded product launch",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.25,
        metadata={"agent_type": "routing"},
    )
    batch = ApprovalBatch(
        batch_id=batch_id,
        items=[item],
        context={"tenant_id": tenant_id, "agent_type": "routing"},
    )
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    await storage.save_batch(batch)

    selected_at = datetime(2026, 8, 5, 6, 1, 2, tzinfo=timezone.utc)
    retry_at = datetime(2026, 8, 5, 6, 9, 10, tzinfo=timezone.utc)
    selected_decision = ReviewDecision(
        item_id=item.item_id,
        approved=True,
        feedback="The route targets the exact video-search capability.",
        corrections={"chosen_agent": "video_search_agent"},
        reviewer="reviewer@example.com",
        timestamp=selected_at,
    )
    original_get_dataset = storage.provider.datasets.get_dataset
    dataset_read_attempts = 0

    async def fail_first_dataset_read(name):
        nonlocal dataset_read_attempts
        dataset_read_attempts += 1
        if dataset_read_attempts == 1:
            raise ConnectionError("Phoenix dataset read interrupted")
        return await original_get_dataset(name=name)

    monkeypatch.setattr(
        storage.provider.datasets,
        "get_dataset",
        fail_first_dataset_read,
    )

    with pytest.raises(RuntimeError) as failure:
        await storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=dataset_name,
            item=item,
            decision=selected_decision,
            project_context=batch.context,
        )

    assert str(failure.value) == (
        "Failed to persist approved item: "
        f"tenant={tenant_id} dataset={dataset_name} "
        f"batch={batch_id} item={item.item_id}"
    )
    assert isinstance(failure.value.__cause__, ConnectionError)
    assert str(failure.value.__cause__) == "Phoenix dataset read interrupted"
    with pytest.raises(DatasetNotFoundError):
        await original_get_dataset(name=dataset_name)

    retry_decision = ReviewDecision(
        item_id=item.item_id,
        approved=True,
        feedback="The route targets the exact video-search capability.",
        corrections={"chosen_agent": "video_search_agent"},
        reviewer="reviewer@example.com",
        timestamp=retry_at,
    )
    approved = await storage.persist_approved_item(
        batch_id=batch_id,
        dataset_name=dataset_name,
        item=item,
        decision=retry_decision,
        project_context=batch.context,
    )

    expected_decision = {
        "reviewer": "reviewer@example.com",
        "feedback": "The route targets the exact video-search capability.",
        "corrections": {"chosen_agent": "video_search_agent"},
        "timestamp": selected_at.isoformat(),
    }
    assert retry_decision.timestamp == selected_at
    assert approved.reviewed_at == selected_at
    assert approved.metadata["decision"] == expected_decision

    dataset = await original_get_dataset(name=dataset_name)
    assert len(dataset) == 1
    record = dataset.iloc[0]["input"]
    assert record["reviewed_at"] == selected_at.isoformat()
    assert record["metadata.approval_decision_timestamp"] == selected_at.isoformat()
    assert record["metadata.decision"] == expected_decision

    annotations = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        spans = await storage.provider.traces.get_spans(
            project=storage.full_project_name,
            filters={"name": "approval_item"},
        )
        annotations = await storage.provider.annotations.get_annotations(
            spans,
            project=storage.full_project_name,
            annotation_names=["human_approval"],
        )
        if len(annotations) == 1:
            break
        await asyncio.sleep(0.5)
    assert annotations is not None
    assert len(annotations) == 1
    annotation = annotations.iloc[0]
    assert annotation["result.label"] == "approved"
    assert annotation["result.score"] == 1.0
    assert annotation["metadata"] == {
        "item_id": item.item_id,
        "timestamp": selected_at.isoformat(),
        "reviewed_at": selected_at.isoformat(),
        "reviewer": "reviewer@example.com",
        "feedback": "The route targets the exact video-search capability.",
    }


async def test_concurrent_same_intent_persists_redis_winner_timestamp(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
    monkeypatch,
):
    tenant_alias = f"decisionconcurrent{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    batch_id = f"batch-{uuid4().hex}"
    dataset_name = approved_synthetic_dataset_name(tenant_id)
    item = ReviewItem(
        item_id=f"{batch_id}-item",
        data={
            "query": "find the recorded product launch",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.25,
        metadata={"agent_type": "routing"},
    )
    batch = ApprovalBatch(
        batch_id=batch_id,
        items=[item],
        context={"tenant_id": tenant_id, "agent_type": "routing"},
    )
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    await storage.save_batch(batch)

    redis_winner_at = datetime(2026, 8, 5, 7, 1, 2, tzinfo=timezone.utc)
    dataset_contender_at = datetime(2026, 8, 5, 7, 9, 10, tzinfo=timezone.utc)

    def decision(timestamp):
        return ReviewDecision(
            item_id=item.item_id,
            approved=True,
            feedback="The route targets the exact video-search capability.",
            corrections={"chosen_agent": "video_search_agent"},
            reviewer="reviewer@example.com",
            timestamp=timestamp,
        )

    redis_winner = decision(redis_winner_at)
    dataset_contender = decision(dataset_contender_at)
    first_selected = asyncio.Event()
    release_first = asyncio.Event()
    select_review_decision = storage.select_review_decision

    async def order_selection(*, batch_id, original_item_id, decision):
        selected = await select_review_decision(
            batch_id=batch_id,
            original_item_id=original_item_id,
            decision=decision,
        )
        if decision is redis_winner:
            first_selected.set()
            await release_first.wait()
        return selected

    monkeypatch.setattr(storage, "select_review_decision", order_selection)

    first_task = asyncio.create_task(
        storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=dataset_name,
            item=item,
            decision=redis_winner,
            project_context=batch.context,
        )
    )
    await asyncio.wait_for(first_selected.wait(), timeout=10)
    second_task = asyncio.create_task(
        storage.persist_approved_item(
            batch_id=batch_id,
            dataset_name=dataset_name,
            item=item,
            decision=dataset_contender,
            project_context=batch.context,
        )
    )
    try:
        second_result = await asyncio.wait_for(second_task, timeout=60)
    finally:
        release_first.set()
    first_result = await asyncio.wait_for(first_task, timeout=60)

    selected = await storage._replacement_records.select_review_decision(
        tenant_id=tenant_id,
        batch_id=batch_id,
        original_item_id=item.item_id,
        candidate={
            "item_id": item.item_id,
            "approved": True,
            "reviewer": "reviewer@example.com",
            "feedback": "The route targets the exact video-search capability.",
            "corrections": {"chosen_agent": "video_search_agent"},
            "timestamp": redis_winner_at.isoformat(),
        },
    )
    assert selected.payload["timestamp"] == redis_winner_at.isoformat()
    assert redis_winner.timestamp == redis_winner_at
    assert dataset_contender.timestamp == redis_winner_at
    assert first_result.reviewed_at == redis_winner_at
    assert second_result.reviewed_at == redis_winner_at

    dataset = await storage.provider.datasets.get_dataset(name=dataset_name)
    assert len(dataset) == 1
    record = dataset.iloc[0]["input"]
    assert record["reviewed_at"] == redis_winner_at.isoformat()
    assert record["metadata.approval_decision_timestamp"] == (
        redis_winner_at.isoformat()
    )
    assert record["metadata.decision"] == {
        "reviewer": "reviewer@example.com",
        "feedback": "The route targets the exact video-search capability.",
        "corrections": {"chosen_agent": "video_search_agent"},
        "timestamp": redis_winner_at.isoformat(),
    }


async def test_rejection_and_regeneration_use_redis_selected_timestamps(
    phoenix_container,
    telemetry_manager_with_phoenix,
    approval_redis_url,
):
    tenant_alias = f"decisionreject{uuid4().hex[:8]}"
    tenant_id = f"{tenant_alias}:{tenant_alias}"
    batch_id = f"batch-{uuid4().hex}"
    rejected_item = ReviewItem(
        item_id=f"{batch_id}-rejected",
        data={
            "query": "find launch footage",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.2,
        metadata={"agent_type": "routing"},
    )
    regenerated_item = ReviewItem(
        item_id=f"{batch_id}-regenerated",
        data={
            "query": "find product footage",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.3,
        metadata={"agent_type": "routing"},
    )
    batch = ApprovalBatch(
        batch_id=batch_id,
        items=[rejected_item, regenerated_item],
        context={"tenant_id": tenant_id, "agent_type": "routing"},
    )
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_alias,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=approval_redis_url,
    )
    await storage.save_batch(batch)

    rejected_at = datetime(2026, 8, 5, 8, 1, 2, tzinfo=timezone.utc)
    regenerated_at = datetime(2026, 8, 5, 8, 2, 3, tzinfo=timezone.utc)
    retry_at = datetime(2026, 8, 5, 8, 9, 10, tzinfo=timezone.utc)

    def rejection(item, timestamp):
        return ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback="Use the exact product-launch wording.",
            corrections={"query": "find the exact product launch recording"},
            reviewer="reviewer@example.com",
            timestamp=timestamp,
        )

    await storage.select_review_decision(
        batch_id=batch_id,
        original_item_id=rejected_item.item_id,
        decision=rejection(rejected_item, rejected_at),
    )
    await storage.select_review_decision(
        batch_id=batch_id,
        original_item_id=regenerated_item.item_id,
        decision=rejection(regenerated_item, regenerated_at),
    )

    class FeedbackHandler:
        def __init__(self):
            self.seen = []

        async def process_rejection(self, item, decision):
            self.seen.append((item.item_id, decision.timestamp))
            if item.item_id == rejected_item.item_id:
                return None
            return ReviewItem(
                item_id=f"{item.item_id}-replacement",
                data={
                    "query": "find the exact product launch recording",
                    "chosen_agent": "video_search_agent",
                },
                confidence=0.9,
            )

    feedback_handler = FeedbackHandler()
    agent = HumanApprovalAgent(
        confidence_extractor=_ConfidenceFieldExtractor(),
        feedback_handler=feedback_handler,
        storage=storage,
    )
    rejected_retry = rejection(rejected_item, retry_at)
    regenerated_retry = rejection(regenerated_item, retry_at)

    rejected = await agent.apply_decision(batch_id, rejected_retry)
    regenerated = await agent.apply_decision(batch_id, regenerated_retry)

    rejected_decision = {
        "reviewer": "reviewer@example.com",
        "feedback": "Use the exact product-launch wording.",
        "corrections": {"query": "find the exact product launch recording"},
        "timestamp": rejected_at.isoformat(),
    }
    regenerated_decision = {
        **rejected_decision,
        "timestamp": regenerated_at.isoformat(),
    }
    assert rejected_retry.timestamp == rejected_at
    assert regenerated_retry.timestamp == regenerated_at
    assert rejected.status is ApprovalStatus.REJECTED
    assert rejected.reviewed_at == rejected_at
    assert rejected.metadata["decision"] == rejected_decision
    assert regenerated.status is ApprovalStatus.REGENERATED
    assert regenerated.metadata["decision"] == regenerated_decision
    assert feedback_handler.seen == [
        (rejected_item.item_id, rejected_at),
        (regenerated_item.item_id, regenerated_at),
    ]

    persisted = await storage.get_batch(batch_id)
    assert persisted is not None
    assert [
        (
            candidate.item_id,
            candidate.status,
            candidate.reviewed_at,
            candidate.metadata.get("decision"),
        )
        for candidate in persisted.items
    ] == [
        (
            rejected_item.item_id,
            ApprovalStatus.REJECTED,
            rejected_at,
            rejected_decision,
        ),
        (
            regenerated_item.item_id,
            ApprovalStatus.REJECTED,
            None,
            None,
        ),
        (
            f"{regenerated_item.item_id}-replacement",
            ApprovalStatus.REGENERATED,
            None,
            regenerated_decision,
        ),
    ]

    annotations = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        spans = await storage.provider.traces.get_spans(
            project=storage.full_project_name,
            filters={"name": "approval_item"},
        )
        annotations = await storage.provider.annotations.get_annotations(
            spans,
            project=storage.full_project_name,
            annotation_names=["human_approval"],
        )
        if len(annotations) == 1:
            break
        await asyncio.sleep(0.5)
    assert annotations is not None
    assert len(annotations) == 1
    annotation = annotations.iloc[0]
    assert annotation["result.label"] == "rejected"
    assert annotation["result.score"] == 0.0
    assert annotation["metadata"] == {
        "item_id": rejected_item.item_id,
        "timestamp": rejected_at.isoformat(),
        "reviewed_at": rejected_at.isoformat(),
        "reviewer": "reviewer@example.com",
        "feedback": "Use the exact product-launch wording.",
    }
