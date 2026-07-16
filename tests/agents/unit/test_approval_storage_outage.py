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
async def test_append_to_training_dataset_raises_on_outage():
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem

    storage = _bare_storage()
    item = ReviewItem(
        item_id="item-1",
        data={"q": "x"},
        confidence=0.9,
        status=ApprovalStatus.APPROVED,
    )
    with pytest.raises(Exception):
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data", items=[item]
        )
