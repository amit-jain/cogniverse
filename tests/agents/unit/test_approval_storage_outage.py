"""Approval storage reads/writes must raise on a telemetry outage, not flatten
it to "not found" / False.

get_batch returned None on outage, so the orchestrator kept the stale
pre-decision batch and the workflow sat awaiting_approval forever;
get_item_span_id returned None, so apply_decision silently skipped the approval
annotation; append_to_training_dataset returned False, so an approved item
never reached the dataset while apply_decision reported success. get_pending_batches
(the sibling) already raises — these pin the same contract on the rest.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_storage(get_spans_side_effect=None):
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = "acme:acme"
    storage.full_project_name = "acme__approvals"
    storage._replacement_records = None
    provider = MagicMock()
    provider.traces.get_all_spans = AsyncMock(side_effect=get_spans_side_effect)
    provider.datasets = MagicMock()
    provider.datasets.get_dataset = AsyncMock(
        side_effect=RuntimeError("Phoenix write failed")
    )
    storage.provider = provider

    @asynccontextmanager
    async def unlocked_dataset(_dataset_name):
        yield

    storage._approval_dataset_lock = unlocked_dataset
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
    annotation — a swallowed write failure drops the reviewer and explanation
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
        data={"query": "find the incident recording", "chosen_agent": "search"},
        confidence=0.9,
        metadata={"agent_type": "routing"},
        status=ApprovalStatus.APPROVED,
        reviewed_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_append_to_training_dataset_raises_on_outage():
    """A get_dataset OUTAGE must propagate — not masquerade as first-run and
    recreate a live dataset. The store signals genuine absence with
    DatasetNotFoundError; anything else is a failure."""
    storage = _bare_storage()
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to append items to training dataset: "
            "tenant=acme:acme dataset=approved_synthetic_data-acme:acme"
        ),
    ) as exc_info:
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:acme",
            items=[_approved_item()],
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Phoenix write failed"
    assert storage.provider.datasets.create_dataset.await_count == 0


@pytest.mark.asyncio
async def test_append_creates_dataset_on_genuine_absence():
    from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

    storage = _bare_storage()
    storage.provider.datasets.get_dataset = AsyncMock(
        side_effect=DatasetNotFoundError("absent")
    )
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    ok = await storage.append_to_training_dataset(
        dataset_name="approved_synthetic_data-acme:acme",
        items=[_approved_item()],
    )
    assert ok is True
    assert storage.provider.datasets.create_dataset.await_count == 1


@pytest.mark.asyncio
async def test_append_rejects_unqualified_dataset_name_before_backend_access():
    storage = _bare_storage()

    with pytest.raises(
        ValueError,
        match=("Approval dataset name must be 'approved_synthetic_data-acme:acme'"),
    ):
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data",
            items=[_approved_item()],
        )

    storage.provider.datasets.get_dataset.assert_not_awaited()


@pytest.mark.asyncio
async def test_append_outage_mid_append_does_not_recreate():
    """append_to_dataset failing after the dataset was confirmed to exist is
    an outage — it must raise, not fall into the create-new branch."""
    storage = _bare_storage()
    import pandas as pd

    storage.provider.datasets.get_dataset = AsyncMock(
        return_value=pd.DataFrame({"input": []})
    )
    storage.provider.datasets.append_to_dataset = AsyncMock(
        side_effect=ConnectionError("reset mid-append")
    )
    storage.provider.datasets.create_dataset = AsyncMock(return_value="id")

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to append items to training dataset: "
            "tenant=acme:acme dataset=approved_synthetic_data-acme:acme"
        ),
    ) as exc_info:
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:acme",
            items=[_approved_item()],
        )
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert str(exc_info.value.__cause__) == "reset mid-append"
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
async def test_one_malformed_item_invalidates_the_entire_batch():
    """A truncated item cannot be omitted from a batch that claims a fixed
    pending count; the whole reconstruction fails with item context."""
    import pandas as pd

    storage = _bare_storage()
    storage.provider.annotations.get_annotations = AsyncMock(
        return_value=pd.DataFrame()
    )

    with pytest.raises(
        RuntimeError,
        match="Approval batch 'b1' contains malformed item 'i2'",
    ) as exc_info:
        await storage.get_batch("b1", spans_df=_batch_frame())

    assert exc_info.value.__cause__.__class__.__name__ == "JSONDecodeError"


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


class _CheckedExportTelemetry:
    def __init__(self, export_ok, export_exc):
        self._export_ok = export_ok
        self._export_exc = export_exc

    def span(self, **kwargs):
        assert kwargs["require_export"] is True
        telemetry = self

        class _CheckedSpan(_FlushSpanCtx):
            def __exit__(self, *args):
                if telemetry._export_exc is not None:
                    raise telemetry._export_exc
                if telemetry._export_ok is False:
                    raise RuntimeError("telemetry exporter rejected span")
                return False

        return _CheckedSpan()


def _flush_storage(flush_ok=None, flush_exc=None):
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = "acme:acme"
    storage.project_name = "proj"
    storage.full_project_name = "proj"
    storage.telemetry_manager = _CheckedExportTelemetry(flush_ok, flush_exc)
    storage._replacement_records = None
    return storage


@pytest.mark.asyncio
async def test_save_batch_raises_when_checked_export_is_rejected():
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_ok=False)
    batch = ApprovalBatch(batch_id="b1", items=[], context={"tenant_id": "acme:acme"})
    with pytest.raises(RuntimeError, match="failed to export"):
        await storage.save_batch(batch)


@pytest.mark.asyncio
async def test_save_batch_raises_when_checked_export_raises():
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_exc=ConnectionError("collector down"))
    batch = ApprovalBatch(batch_id="b3", items=[], context={"tenant_id": "acme:acme"})
    with pytest.raises(RuntimeError, match="failed to export"):
        await storage.save_batch(batch)


@pytest.mark.asyncio
async def test_save_batch_returns_id_when_flush_succeeds():
    from cogniverse_core.approval.interfaces import ApprovalBatch

    storage = _flush_storage(flush_ok=True)
    batch = ApprovalBatch(batch_id="b2", items=[], context={"tenant_id": "acme:acme"})
    assert await storage.save_batch(batch) == "b2"


@pytest.mark.asyncio
async def test_save_batch_checked_export_does_not_block_event_loop():
    from cogniverse_core.approval.interfaces import ApprovalBatch

    export_entered = threading.Event()
    loop_advanced = threading.Event()

    class _BlockingSpan(_FlushSpanCtx):
        def __exit__(self, *args):
            export_entered.set()
            assert loop_advanced.wait(timeout=2), "checked export blocked event loop"
            return False

    class _BlockingTelemetry:
        def __init__(self):
            self.export_thread = None
            self.span_kwargs = None

        def span(self, **kwargs):
            self.export_thread = threading.get_ident()
            self.span_kwargs = kwargs
            return _BlockingSpan()

    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = "acme:acme"
    storage.project_name = "synthetic_data"
    storage.full_project_name = "cogniverse-acme:acme-synthetic_data"
    storage.telemetry_manager = _BlockingTelemetry()
    storage._replacement_records = None
    batch = ApprovalBatch(
        batch_id="nonblocking",
        items=[],
        context={"tenant_id": "acme:acme"},
    )

    async def advance_loop():
        while not export_entered.is_set():
            await asyncio.sleep(0)
        loop_advanced.set()

    saved_batch_id, _ = await asyncio.gather(
        storage.save_batch(batch),
        advance_loop(),
    )

    assert saved_batch_id == "nonblocking"
    assert storage.telemetry_manager.export_thread != threading.get_ident()
    assert storage.telemetry_manager.span_kwargs["require_export"] is True


@pytest.mark.asyncio
async def test_save_batch_retry_after_mid_export_is_exactly_readable():
    import itertools

    import pandas as pd

    from cogniverse_core.approval.interfaces import ApprovalBatch, ReviewItem

    class _RecordingTelemetry:
        def __init__(self):
            self.rows = []
            self.ids = itertools.count(1)
            self.stack = []
            self.item_exports = 0
            self.failed_once = False

        def span(self, **kwargs):
            assert kwargs["require_export"] is True
            telemetry = self
            span_id = f"span-{next(self.ids)}"
            parent_id = self.stack[-1] if self.stack else None

            class _RecordingSpan(_FlushSpanCtx):
                def __enter__(self):
                    telemetry.stack.append(span_id)
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    assert telemetry.stack.pop() == span_id
                    if kwargs["name"] == "approval_item":
                        telemetry.item_exports += 1
                        if telemetry.item_exports == 2 and not telemetry.failed_once:
                            telemetry.failed_once = True
                            raise TimeoutError("collector failed after first item")
                    telemetry.rows.append(
                        {
                            "name": kwargs["name"],
                            "context.span_id": span_id,
                            "parent_id": parent_id,
                            **{
                                f"attributes.{key}": value
                                for key, value in kwargs["attributes"].items()
                            },
                        }
                    )
                    return False

            return _RecordingSpan()

    telemetry = _RecordingTelemetry()
    storage = object.__new__(ApprovalStorageImpl)
    storage.tenant_id = "acme:acme"
    storage.project_name = "synthetic_data"
    storage.full_project_name = "cogniverse-acme:acme-synthetic_data"
    storage.telemetry_manager = telemetry
    storage._replacement_records = None
    storage.provider = MagicMock()
    storage.provider.annotations.get_annotations = AsyncMock(
        return_value=pd.DataFrame()
    )
    batch = ApprovalBatch(
        batch_id="retry-batch",
        items=[
            ReviewItem(
                item_id="item-one",
                data={"query": "find the first incident"},
                confidence=0.25,
            ),
            ReviewItem(
                item_id="item-two",
                data={"query": "find the second incident"},
                confidence=0.45,
            ),
        ],
        context={"tenant_id": "acme:acme", "agent_type": "routing"},
    )

    with pytest.raises(RuntimeError, match="spans failed to export"):
        await storage.save_batch(batch)
    assert await storage.save_batch(batch) == "retry-batch"

    restored = await storage.get_batch(
        "retry-batch",
        spans_df=pd.DataFrame(telemetry.rows),
    )
    assert restored.context == batch.context
    assert [
        (item.item_id, item.data, item.confidence, item.status)
        for item in restored.items
    ] == [
        (item.item_id, item.data, item.confidence, item.status) for item in batch.items
    ]


@pytest.mark.asyncio
async def test_replacement_visibility_wait_uses_every_backoff(monkeypatch):
    storage = object.__new__(ApprovalStorageImpl)
    storage._replacement_event_exists = AsyncMock(
        side_effect=[False, False, False, False, False, True]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    visible = await storage._wait_for_replacement_event(
        batch_id="batch-1",
        original_item_id="original-1",
        selected_json='{"item_id":"replacement-1"}',
        selected_sha256="a" * 64,
    )

    assert visible is True
    assert [call.args for call in sleep.await_args_list] == [
        (0.25,),
        (0.5,),
        (1,),
        (2,),
        (4,),
    ]


@pytest.mark.asyncio
async def test_persistence_outage_leaves_item_pending_not_approved():
    """A persistence failure must raise while the caller's item remains
    pending and unmodified."""
    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_core.approval.interfaces import (
        ApprovalBatch,
        ApprovalStatus,
        ReviewDecision,
        ReviewItem,
    )

    storage = _bare_storage()
    item = ReviewItem(item_id="item-1", data={"q": "x"}, confidence=0.4)
    batch = ApprovalBatch(
        batch_id="b1",
        items=[item],
        context={"tenant_id": "acme:acme", "optimizer": "routing"},
    )
    storage.get_batch = AsyncMock(return_value=batch)
    storage.persist_approved_item = AsyncMock(
        side_effect=RuntimeError("dataset backend down")
    )
    storage.select_review_decision = AsyncMock(
        side_effect=lambda **kwargs: kwargs["decision"]
    )

    agent = HumanApprovalAgent(confidence_extractor=MagicMock(), storage=storage)

    with pytest.raises(RuntimeError, match="dataset backend down"):
        await agent.apply_decision(
            "b1", ReviewDecision(item_id="item-1", approved=True, reviewer="r")
        )

    assert item.status == ApprovalStatus.PENDING_REVIEW
    assert item.reviewed_at is None
    storage.persist_approved_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_approved_submit_requires_canonical_dataset_persistence():
    from types import SimpleNamespace

    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_core.approval.interfaces import ApprovalBatch, ReviewItem

    save_batch = AsyncMock(return_value="batch-1")
    storage = SimpleNamespace(
        tenant_id="acme:acme",
        save_batch=save_batch,
    )
    batch = ApprovalBatch(
        batch_id="batch-1",
        items=[
            ReviewItem(
                item_id="routing-1",
                data={
                    "query": "find the exact launch recording",
                    "chosen_agent": "video_search_agent",
                },
                confidence=0.93,
                metadata={"agent_type": "routing"},
            )
        ],
        context={"tenant_id": "acme:acme", "agent_type": "routing"},
    )
    agent = HumanApprovalAgent(
        confidence_extractor=MagicMock(),
        confidence_threshold=0.85,
        storage=storage,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Approval storage must implement persist_approved_item before "
            "auto-approved items can be submitted"
        ),
    ):
        await agent.submit_for_review(batch)

    save_batch.assert_not_awaited()


def test_phoenix_string_coercions_preserve_exact_boolean_and_null_values():
    from cogniverse_agents.approval.approval_storage import (
        _provider_value_matches_canonical,
    )

    assert _provider_value_matches_canonical("True", True) is True
    assert _provider_value_matches_canonical("False", False) is True
    assert _provider_value_matches_canonical("", None) is True
    assert _provider_value_matches_canonical(float("nan"), None) is True
    assert _provider_value_matches_canonical("true", True) is False
    assert _provider_value_matches_canonical("false", False) is False
    assert _provider_value_matches_canonical("null", None) is False
    assert _provider_value_matches_canonical("None", None) is False
    assert _provider_value_matches_canonical("nan", None) is False
    assert _provider_value_matches_canonical("NaN", None) is False
