"""Approval storage reads/writes must raise on a telemetry outage, not flatten
it to "not found" / False.

get_batch returned None on outage, so the orchestrator kept the stale
pre-decision batch and the workflow sat awaiting_approval forever;
get_item_span_id returned None, so apply_decision silently skipped the approval
annotation; append_to_training_dataset returned False, so an approved item
never reached the dataset while apply_decision reported success. get_pending_batches
(the sibling) already raises — these pin the same contract on the rest.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_storage(get_spans_side_effect=None):
    storage = object.__new__(ApprovalStorageImpl)
    storage.full_project_name = "acme__approvals"
    provider = MagicMock()
    provider.traces.get_spans = AsyncMock(side_effect=get_spans_side_effect)
    provider.datasets = MagicMock()
    provider.datasets.get_dataset = AsyncMock(
        side_effect=RuntimeError("Phoenix write failed")
    )
    storage.provider = provider
    return storage


@pytest.mark.asyncio
async def test_get_batch_raises_on_outage():
    storage = _bare_storage(RuntimeError("Phoenix unreachable"))
    with pytest.raises(Exception, match="Phoenix unreachable"):
        await storage.get_batch("batch-1")


@pytest.mark.asyncio
async def test_get_item_span_id_raises_on_outage():
    storage = _bare_storage(RuntimeError("Phoenix unreachable"))
    with pytest.raises(Exception, match="Phoenix unreachable"):
        await storage.get_item_span_id("item-1", batch_id="batch-1")


@pytest.mark.asyncio
async def test_log_approval_decision_raises_on_annotation_outage():
    """The reviewer identity and feedback live only in the human_approval
    annotation — a swallowed write failure drops the who/why audit trail
    while apply_decision reports the approval applied."""
    storage = _bare_storage()
    storage.provider.annotations = MagicMock()
    storage.provider.annotations.add_annotation = AsyncMock(
        side_effect=RuntimeError("Phoenix annotation write failed")
    )
    with pytest.raises(Exception, match="annotation write failed"):
        await storage.log_approval_decision(
            span_id="span-1",
            item_id="item-1",
            approved=True,
            feedback="looks right",
            reviewer="ops@acme",
        )


def _approved_item():
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem

    return ReviewItem(
        item_id="item-1",
        data={"q": "x"},
        confidence=0.9,
        status=ApprovalStatus.APPROVED,
    )


@pytest.mark.asyncio
async def test_append_to_training_dataset_raises_on_outage():
    """A get_dataset OUTAGE must propagate — not masquerade as first-run and
    recreate a live dataset. The store signals genuine absence with
    KeyError/ValueError; anything else is a failure."""
    storage = _bare_storage()
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    with pytest.raises(RuntimeError, match="Phoenix write failed"):
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data", items=[_approved_item()]
        )
    assert storage.provider.datasets.create_dataset.await_count == 0


@pytest.mark.asyncio
async def test_append_creates_dataset_on_genuine_absence():
    storage = _bare_storage()
    storage.provider.datasets.get_dataset = AsyncMock(side_effect=KeyError("absent"))
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    ok = await storage.append_to_training_dataset(
        dataset_name="approved_synthetic_data", items=[_approved_item()]
    )
    assert ok is True
    assert storage.provider.datasets.create_dataset.await_count == 1


@pytest.mark.asyncio
async def test_append_outage_mid_append_does_not_recreate():
    """append_to_dataset failing after the dataset was confirmed to exist is
    an outage — it must raise, not fall into the create-new branch."""
    storage = _bare_storage()
    storage.provider.datasets.get_dataset = AsyncMock(return_value=MagicMock())
    storage.provider.datasets.append_to_dataset = AsyncMock(
        side_effect=ConnectionError("reset mid-append")
    )
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    with pytest.raises(ConnectionError, match="reset mid-append"):
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data", items=[_approved_item()]
        )
    assert storage.provider.datasets.create_dataset.await_count == 0


def _item_row(span_id, item_id, *, data, confidence, status="pending_review"):
    return {
        "name": "approval_item",
        "parent_id": "s0",
        "context.span_id": span_id,
        "attributes.batch_id": "b1",
        "attributes.item_id": item_id,
        "attributes.status": status,
        "attributes.created_at": "2026-07-01T00:00:00+00:00",
        "attributes.reviewed_at": None,
        "attributes.data": data,
        "attributes.metadata": "{}",
        "attributes.confidence": confidence,
        "attributes.pending_review": None,
        "attributes.context": None,
    }


def _batch_frame():
    import pandas as pd

    rows = [
        {
            "name": "approval_batch",
            "parent_id": None,
            "context.span_id": "s0",
            "attributes.batch_id": "b1",
            "attributes.item_id": None,
            "attributes.status": None,
            "attributes.created_at": None,
            "attributes.reviewed_at": None,
            "attributes.data": None,
            "attributes.metadata": None,
            "attributes.confidence": None,
            "attributes.pending_review": 2,
            "attributes.context": "{}",
        },
        _item_row("s1", "i1", data='{"q": "good"}', confidence=0.9),
        _item_row("s2", "i2", data='{"q": ', confidence=0.8),
        _item_row("s3", "i3", data='{"q": "x"}', confidence="high"),
        _item_row("s4", "i4", data='{"q": "also good"}', confidence=0.7),
    ]
    return pd.DataFrame(rows)


@pytest.mark.asyncio
async def test_one_malformed_item_does_not_poison_the_batch():
    """One span with a truncated data blob or junk confidence costs that
    ITEM, not the tenant's whole pending-approvals view — the outage
    contract (raise) is for backend failures, not for one bad row."""
    import pandas as pd

    storage = _bare_storage()
    storage.provider.annotations.get_annotations = AsyncMock(
        return_value=pd.DataFrame()
    )

    batch = await storage.get_batch("b1", spans_df=_batch_frame())

    assert batch is not None
    assert [i.item_id for i in batch.items] == ["i1", "i4"]
    assert batch.items[0].confidence == 0.9
    assert batch.items[1].confidence == 0.7


@pytest.mark.asyncio
async def test_get_batch_raises_on_annotation_outage():
    """Item approve/reject status lives ONLY in annotations. An annotation-
    store outage must propagate (raise), not get swallowed — swallowing left
    the frame empty and rebuilt every item at its span-time pending_review,
    silently reverting all decisions so the workflow re-prompted resolved
    items and sat in awaiting_approval."""
    storage = _bare_storage()
    storage.provider.annotations.get_annotations = AsyncMock(
        side_effect=ConnectionError("annotations backend unreachable")
    )
    with pytest.raises(ConnectionError, match="annotations backend unreachable"):
        await storage.get_batch("b1", spans_df=_batch_frame())


def test_ctor_canonicalizes_tenant_for_project_and_provider():
    """Runtime writers register approval spans under the canonical tenant;
    a storage built with a raw id must register, name, and query the SAME
    scope or the approval queue reads an empty project."""
    mgr = MagicMock()
    mgr.config.provider_config = {}

    storage = ApprovalStorageImpl(
        grpc_endpoint="http://localhost:4317",
        http_endpoint="http://localhost:6006",
        tenant_id="acme",
        telemetry_manager=mgr,
    )

    assert storage.tenant_id == "acme:acme"
    assert storage.full_project_name == "cogniverse-acme:acme-synthetic_data"
    assert mgr.register_project.call_args.kwargs["tenant_id"] == "acme:acme"
    assert mgr.get_provider.call_args.kwargs["tenant_id"] == "acme:acme"


class _FlushSpanCtx:
    def set_status(self, *a, **k):
        return None

    def set_attribute(self, *a, **k):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FlushTelemetry:
    """Minimal telemetry manager: span() works, force_flush is configurable."""

    def __init__(self, flush_ok):
        self._flush_ok = flush_ok
        self._tenant_providers = {}

    def span(self, **kwargs):
        return _FlushSpanCtx()


def _flush_storage(flush_ok=None, flush_exc=None):
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = "acme:acme"
    storage.project_name = "proj"
    storage.full_project_name = "proj"
    tm = _FlushTelemetry(flush_ok)

    class _TracerProvider:
        def force_flush(self, timeout_millis=None):
            if flush_exc is not None:
                raise flush_exc
            return flush_ok

    tm._tenant_providers = {"acme:acme:proj": _TracerProvider()}
    storage.telemetry_manager = tm
    return storage


@pytest.mark.asyncio
async def test_save_batch_raises_when_force_flush_fails():
    """force_flush returning False means the batch's spans were NOT exported;
    returning batch_id anyway reports the batch persisted when nothing reached
    the backend."""
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_ok=False)
    batch = ApprovalBatch(batch_id="b1", items=[], context={})
    with pytest.raises(RuntimeError, match="failed to export"):
        await storage.save_batch(batch)


@pytest.mark.asyncio
async def test_save_batch_raises_when_force_flush_itself_raises():
    """A force_flush that RAISES (collector unreachable) must surface the same
    RuntimeError as a False return — swallowing the exception left flush_ok=None,
    skipping the guard and reporting the batch persisted when nothing exported."""
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_exc=ConnectionError("collector down"))
    batch = ApprovalBatch(batch_id="b3", items=[], context={})
    with pytest.raises(RuntimeError, match="failed to export"):
        await storage.save_batch(batch)


@pytest.mark.asyncio
async def test_save_batch_returns_id_when_flush_succeeds():
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_ok=True)
    batch = ApprovalBatch(batch_id="b2", items=[], context={})
    assert await storage.save_batch(batch) == "b2"


@pytest.mark.asyncio
async def test_annotation_outage_leaves_item_pending_not_approved():
    """The audit annotation is written before the status commit: when the
    annotation backend is down, apply_decision must raise with the item
    still pending and nothing appended to the training dataset. The old
    order committed status=APPROVED first, so an outage produced an
    approved-but-unaudited item that never reached the dataset."""
    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_core.approval.interfaces import (
        ApprovalBatch,
        ApprovalStatus,
        ReviewDecision,
        ReviewItem,
    )

    storage = _bare_storage()
    item = ReviewItem(item_id="item-1", data={"q": "x"}, confidence=0.4)
    batch = ApprovalBatch(batch_id="b1", items=[item], context={})
    storage.get_batch = AsyncMock(return_value=batch)
    storage.get_item_span_id = AsyncMock(return_value="span-1")
    storage.log_approval_decision = AsyncMock(
        side_effect=RuntimeError("annotation backend down")
    )
    storage.update_item = AsyncMock()
    storage.append_to_training_dataset = AsyncMock()

    agent = HumanApprovalAgent(confidence_extractor=MagicMock(), storage=storage)

    with pytest.raises(RuntimeError, match="annotation backend down"):
        await agent.apply_decision(
            "b1", ReviewDecision(item_id="item-1", approved=True, reviewer="r")
        )

    assert item.status == ApprovalStatus.PENDING_REVIEW
    storage.update_item.assert_not_awaited()
    storage.append_to_training_dataset.assert_not_awaited()
