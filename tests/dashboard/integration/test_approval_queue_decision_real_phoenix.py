"""Real-Phoenix round-trip for the Approval Queue tab's decision persistence.

The dashboard's ``_persist_decision`` used to call only ``record_decision``
(a diagnostic ``approval_decision`` span) plus the training append. The
pending queue is reconstructed from each item's ``item_status_update`` /
``human_approval`` annotations, which that span is not -- so an approved item
reappeared as pending on the next refresh and a re-approval duplicated the
training append. These tests drive the real dashboard function against a real
Phoenix container and assert the item durably leaves the pending queue.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cogniverse_agents.approval import HumanApprovalAgent
from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_core.approval.training_schema import (
    APPROVED_SYNTHETIC_OUTPUT_FIELDS,
)
from cogniverse_dashboard.tabs import approval_queue
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from cogniverse_synthetic.approval import SyntheticDataConfidenceExtractor

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

# Approving an item appends it to the training dataset, which validates the
# supervision shape. The output field comes from the production mapping so a
# rename here fails rather than being absorbed.
_AGENT_TYPE = "routing"
_OUTPUT_FIELD = APPROVED_SYNTHETIC_OUTPUT_FIELDS[_AGENT_TYPE]
_TRAINING_METADATA = {"agent_type": _AGENT_TYPE}


def _run(coro):
    return asyncio.run(coro)


def _storage(phoenix_container, telemetry_manager_with_phoenix, tenant_id, redis_url):
    # Redis is where the canonical review decision is elected; without it the
    # storage refuses to apply a decision at all.
    return ApprovalStorageImpl(
        redis_url=redis_url,
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager_with_phoenix,
    )


def _agent(storage):
    """The agent the dashboard builds, wrapping the real storage.

    ``_persist_decision`` reads ``approval_agent`` and delegates to
    ``apply_decision``; seeding only ``approval_storage`` left it unset and
    the persist path raised before touching Phoenix.
    """
    return HumanApprovalAgent.from_approval_config(
        ApprovalConfig(),
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        storage=storage,
    )


def _stored_item(storage, item_id):
    """The item as the queue serves it, carrying its ``approval_batch_id``.

    The persist path requires that metadata, which is attached on save --
    the locally-constructed item does not have it.
    """
    for batch in _run(storage.get_pending_batches(None)):
        for item in batch.pending_review:
            if item.item_id == item_id:
                return item
    raise AssertionError(f"item {item_id} is not in the pending queue")


def _pending_ids(storage) -> set[str]:
    # Exactly what HumanApprovalAgent.get_pending_items does: flatten the
    # PENDING_REVIEW items across the tenant's pending batches.
    batches = _run(storage.get_pending_batches(None))
    return {i.item_id for b in batches for i in b.pending_review}


def _poll_pending(storage, item_id, *, want_present, timeout=60.0) -> bool:
    """Poll the pending queue until item_id's presence matches want_present."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (item_id in _pending_ids(storage)) == want_present:
            return True
        time.sleep(2.0)
    return (item_id in _pending_ids(storage)) == want_present


def _training_rows(storage, item_id) -> int:
    # Phoenix's to_dataframe() nests the record fields inside the input/
    # output/metadata dict columns, so search those rather than expecting a
    # flat item_id column.
    try:
        # The dataset is per tenant; the shared literal named a dataset no
        # tenant writes, so every lookup answered not-found and counted zero.
        df = _run(
            storage.provider.datasets.get_dataset(
                approved_synthetic_dataset_name(storage.tenant_id)
            )
        )
    except DatasetNotFoundError:
        return 0
    count = 0
    for _, row in df.iterrows():
        cells = [row[c] for c in ("input", "output", "metadata") if c in df.columns]
        if any(isinstance(c, dict) and c.get("item_id") == item_id for c in cells):
            count += 1
    return count


def _wait_training_rows(storage, item_id, want: int, timeout=30.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = _training_rows(storage, item_id)
        if n == want:
            return n
        time.sleep(2.0)
    return _training_rows(storage, item_id)


def test_dashboard_approval_removes_item_from_pending_queue(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
    monkeypatch,
):
    tenant_id = f"apprvq{uuid4().hex[:8]}"
    item_id = f"pending_item_{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )

    item = ReviewItem(
        item_id=item_id,
        data={"query": "is this relevant?", _OUTPUT_FIELD: "video_search_agent"},
        metadata=dict(_TRAINING_METADATA),
        confidence=0.4,
        status=ApprovalStatus.PENDING_REVIEW,
    )
    batch = ApprovalBatch(
        batch_id=f"batch_{uuid4().hex[:8]}",
        items=[item],
        context={"purpose": "dashboard_approval_roundtrip", "tenant_id": tenant_id},
    )
    _run(storage.save_batch(batch))

    # The item is pending until a decision lands.
    assert _poll_pending(storage, item_id, want_present=True), (
        f"item {item_id} never became visible as pending"
    )

    # The persist path takes the item as the queue serves it.
    item = _stored_item(storage, item_id)

    # Drive the real dashboard persist path with the real storage.
    monkeypatch.setattr(
        approval_queue,
        "st",
        SimpleNamespace(
            session_state={
                "approval_storage": storage,
                "approval_agent": _agent(storage),
            }
        ),
    )
    approval_queue._persist_decision(
        ReviewDecision(item_id=item_id, approved=True, reviewer="reviewer@example.com"),
        item,
    )

    # After the decision the item must NOT reappear as pending. On the old
    # record_decision-only code this stays True forever and the assertion fails.
    assert _poll_pending(storage, item_id, want_present=False), (
        f"item {item_id} still pending after dashboard approval -- the "
        "decision was not persisted to the annotation store"
    )

    # The approved item landed in the training dataset exactly once.
    assert _wait_training_rows(storage, item_id, 1) == 1, (
        f"expected exactly one training row for {item_id}"
    )


def test_dashboard_rejection_removes_item_from_pending_queue(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
    monkeypatch,
):
    tenant_id = f"apprvq{uuid4().hex[:8]}"
    item_id = f"reject_item_{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )

    item = ReviewItem(
        item_id=item_id,
        data={"query": "off topic", _OUTPUT_FIELD: "text_search_agent"},
        metadata=dict(_TRAINING_METADATA),
        confidence=0.3,
        status=ApprovalStatus.PENDING_REVIEW,
    )
    batch = ApprovalBatch(
        batch_id=f"batch_{uuid4().hex[:8]}",
        items=[item],
        context={"purpose": "dashboard_rejection_roundtrip", "tenant_id": tenant_id},
    )
    _run(storage.save_batch(batch))
    assert _poll_pending(storage, item_id, want_present=True)

    # The persist path takes the item as the queue serves it.
    item = _stored_item(storage, item_id)

    monkeypatch.setattr(
        approval_queue,
        "st",
        SimpleNamespace(
            session_state={
                "approval_storage": storage,
                "approval_agent": _agent(storage),
            }
        ),
    )
    approval_queue._persist_decision(
        ReviewDecision(
            item_id=item_id,
            approved=False,
            feedback="not relevant",
            reviewer="reviewer@example.com",
        ),
        item,
    )

    assert _poll_pending(storage, item_id, want_present=False), (
        f"item {item_id} still pending after dashboard rejection"
    )


def test_status_write_failure_leaves_item_pending_and_retryable(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
    monkeypatch,
):
    """A failed status write surfaces the error and keeps the item pending.

    On approval the training append runs before the status flip, so when the
    annotation write fails the exception propagates, the item stays in the
    pending queue, and a retry completes the decision. The retry re-appends,
    so the dataset ends up with two rows for the item -- the cost of keeping
    a half-persisted decision retryable instead of dropping it from the
    queue with its training row missing.
    """
    tenant_id = f"apprvq{uuid4().hex[:8]}"
    item_id = f"fault_item_{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )

    item = ReviewItem(
        item_id=item_id,
        data={"query": "does this persist?", _OUTPUT_FIELD: "video_search_agent"},
        metadata=dict(_TRAINING_METADATA),
        confidence=0.6,
        status=ApprovalStatus.PENDING_REVIEW,
    )
    batch = ApprovalBatch(
        batch_id=f"batch_{uuid4().hex[:8]}",
        items=[item],
        context={"purpose": "dashboard_fault_roundtrip", "tenant_id": tenant_id},
    )
    _run(storage.save_batch(batch))
    assert _poll_pending(storage, item_id, want_present=True)

    # The persist path takes the item as the queue serves it.
    item = _stored_item(storage, item_id)

    monkeypatch.setattr(
        approval_queue,
        "st",
        SimpleNamespace(
            session_state={
                "approval_storage": storage,
                "approval_agent": _agent(storage),
            }
        ),
    )

    async def _fail_update(item, batch_id=None):
        raise RuntimeError("annotation store write failed")

    decision = ReviewDecision(
        item_id=item_id, approved=True, reviewer="reviewer@example.com"
    )
    storage.update_item = _fail_update
    try:
        with pytest.raises(RuntimeError) as failed:
            approval_queue._persist_decision(decision, item)
    finally:
        del storage.update_item

    # The failure names what a reader needs to locate it, and the injected
    # cause survives instead of being flattened into the message text.
    canonical = canonical_tenant_id(tenant_id)
    assert str(failed.value) == (
        "Failed to persist approved item: "
        f"tenant={canonical} "
        f"dataset={approved_synthetic_dataset_name(tenant_id)} "
        f"batch={batch.batch_id} item={item_id}"
    )
    assert str(failed.value.__cause__) == "annotation store write failed"

    # No status annotation was written: the item is still pending, and the
    # append (which runs first) landed exactly once.
    assert item_id in _pending_ids(storage), (
        f"item {item_id} left the pending queue despite the failed status write"
    )
    assert _wait_training_rows(storage, item_id, 1) == 1

    # Retry with the annotation store healthy: the decision completes.
    approval_queue._persist_decision(decision, item)
    assert _poll_pending(storage, item_id, want_present=False), (
        f"item {item_id} still pending after the retried approval"
    )
    # The retry does not re-append: the first attempt's row is the only one,
    # so a status-write failure costs a retry rather than a duplicate
    # training example. Expecting a second row pinned the duplication this
    # module's docstring describes as the defect.
    assert _wait_training_rows(storage, item_id, 1) == 1


def test_concurrent_decisions_persist_without_cross_talk(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
    monkeypatch,
):
    """Two reviewer threads deciding different items through one shared
    storage both persist durably: no exceptions, both items leave the
    pending queue, and only the approved item lands in the training dataset.
    """
    tenant_id = f"apprvq{uuid4().hex[:8]}"
    approve_id = f"conc_approve_{uuid4().hex[:8]}"
    reject_id = f"conc_reject_{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )

    items = {
        approve_id: ReviewItem(
            item_id=approve_id,
            data={"query": "keep me", _OUTPUT_FIELD: "video_search_agent"},
            metadata=dict(_TRAINING_METADATA),
            confidence=0.5,
            status=ApprovalStatus.PENDING_REVIEW,
        ),
        reject_id: ReviewItem(
            item_id=reject_id,
            data={"query": "drop me", _OUTPUT_FIELD: "text_search_agent"},
            metadata=dict(_TRAINING_METADATA),
            confidence=0.2,
            status=ApprovalStatus.PENDING_REVIEW,
        ),
    }
    batch = ApprovalBatch(
        batch_id=f"batch_{uuid4().hex[:8]}",
        items=list(items.values()),
        context={"purpose": "dashboard_concurrent_decisions", "tenant_id": tenant_id},
    )
    _run(storage.save_batch(batch))
    assert _poll_pending(storage, approve_id, want_present=True)
    assert _poll_pending(storage, reject_id, want_present=True)
    items = {i: _stored_item(storage, i) for i in (approve_id, reject_id)}

    monkeypatch.setattr(
        approval_queue,
        "st",
        SimpleNamespace(
            session_state={
                "approval_storage": storage,
                "approval_agent": _agent(storage),
            }
        ),
    )

    decisions = {
        approve_id: ReviewDecision(
            item_id=approve_id, approved=True, reviewer="reviewer@example.com"
        ),
        reject_id: ReviewDecision(
            item_id=reject_id,
            approved=False,
            feedback="off topic",
            reviewer="reviewer@example.com",
        ),
    }
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _decide(item_id: str) -> None:
        try:
            barrier.wait(timeout=30)
            approval_queue._persist_decision(decisions[item_id], items[item_id])
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [
        threading.Thread(target=_decide, args=(approve_id,)),
        threading.Thread(target=_decide, args=(reject_id,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "decision thread hung"
    assert errors == []

    assert _poll_pending(storage, approve_id, want_present=False), (
        f"approved item {approve_id} still pending after concurrent decisions"
    )
    assert _poll_pending(storage, reject_id, want_present=False), (
        f"rejected item {reject_id} still pending after concurrent decisions"
    )
    assert _wait_training_rows(storage, approve_id, 1) == 1
    assert _training_rows(storage, reject_id) == 0
