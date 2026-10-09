"""Real-Phoenix round trip for the entity self-consistency review queue.

``_queue_entity_self_consistency_review`` files the rows the teacher did not
agree on as a pending approval batch. The reviewer's queue is reconstructed
from that storage, so what the optimizer queued must be exactly what the
pending queue serves back -- and rows the teacher agreed on must not be
queued at all.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_core.approval.interfaces import ApprovalStatus
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


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


def test_self_consistency_row_round_trips_through_the_pending_queue(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
):
    """What the optimizer queued is what the reviewer's queue serves back."""
    from cogniverse_agents.optimizer.entity_self_consistency import (
        NO_UNANIMOUS_KEY,
        SELF_CONSISTENCY_METADATA_KEY,
        review_row,
        row_confidence,
    )
    from cogniverse_runtime.optimization_cli import (
        _queue_entity_self_consistency_review,
    )

    tenant_id = f"selfcons{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )
    row = review_row(
        "a man riding a dirt bike",
        [
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [{"text": "man", "type": "PERSON"}],
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
        ],
    )
    queued = _run(
        _queue_entity_self_consistency_review(
            [row],
            storage_factory=lambda: storage,
            tenant_id=canonical_tenant_id(tenant_id),
        )
    )

    assert queued[NO_UNANIMOUS_KEY] == []
    assert queued["rows_queued"] == 1
    batch_id = queued["batch_id"]

    batches = _run(storage.get_pending_batches(None))
    items = [item for batch in batches for item in batch.pending_review]
    assert [item.item_id for item in items] == [f"{batch_id}_0"]
    served = items[0]

    assert served.data == {
        "query": "a man riding a dirt bike",
        "entities": [{"text": "man", "type": "PERSON"}],
        "relationships": [],
    }
    assert served.metadata[SELF_CONSISTENCY_METADATA_KEY] == row["metadata"]
    assert served.metadata["agent_type"] == "entity_extraction"
    assert served.metadata["optimizer_type"] == "entity_extraction"
    assert served.confidence == pytest.approx(row_confidence(row))
    assert served.status is ApprovalStatus.PENDING_REVIEW


def test_unanimous_rows_are_not_queued_for_review(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
):
    """A row the teacher agreed on holds no question, so nothing is queued."""
    from cogniverse_agents.optimizer.entity_self_consistency import (
        NO_UNANIMOUS_KEY,
        review_row,
    )
    from cogniverse_runtime.optimization_cli import (
        _queue_entity_self_consistency_review,
    )

    tenant_id = f"selfcons{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )
    row = review_row(
        "interns working at Nokia",
        [[{"text": "interns", "type": "PERSON"}]] * 3,
    )

    queued = _run(
        _queue_entity_self_consistency_review(
            [row],
            storage_factory=lambda: storage,
            tenant_id=canonical_tenant_id(tenant_id),
        )
    )

    assert queued == {"batch_id": None, "rows_queued": 0, NO_UNANIMOUS_KEY: []}
    assert _run(storage.get_pending_batches(None)) == []


def test_a_row_with_no_unanimous_mention_is_queued_for_review(
    phoenix_container,
    telemetry_manager_with_phoenix,
    workflow_state_redis_url,
):
    """No agreed mention carries no training example, so a human is asked."""
    from cogniverse_agents.optimizer.entity_self_consistency import (
        NO_UNANIMOUS_KEY,
        SELF_CONSISTENCY_METADATA_KEY,
        SELF_CONSISTENCY_SAMPLES,
        review_row,
    )
    from cogniverse_runtime.optimization_cli import (
        _queue_entity_self_consistency_review,
    )

    tenant_id = f"selfcons{uuid4().hex[:8]}"
    storage = _storage(
        phoenix_container,
        telemetry_manager_with_phoenix,
        tenant_id,
        workflow_state_redis_url,
    )
    draws = [
        [{"text": "man", "type": "PERSON"}],
        [{"text": "dirt bike", "type": "CONCEPT"}],
        [{"text": "bike", "type": "CONCEPT"}],
    ]
    row = review_row("a man riding a dirt bike", draws)

    queued = _run(
        _queue_entity_self_consistency_review(
            [row],
            storage_factory=lambda: storage,
            tenant_id=canonical_tenant_id(tenant_id),
        )
    )

    assert queued[NO_UNANIMOUS_KEY] == ["a man riding a dirt bike"]
    assert queued["rows_queued"] == 1
    batch_id = queued["batch_id"]

    batches = _run(storage.get_pending_batches(None))
    items = [item for batch in batches for item in batch.pending_review]
    assert [item.item_id for item in items] == [f"{batch_id}_0"]
    served = items[0]

    # No mention survives into the training example; every one of them is
    # served to the reviewer with the fraction of draws that produced it.
    assert served.data == {
        "query": "a man riding a dirt bike",
        "entities": [],
        "relationships": [],
    }
    assert served.metadata[SELF_CONSISTENCY_METADATA_KEY] == {
        "samples": len(draws),
        "entities": [
            {
                "text": "man",
                "type": "PERSON",
                "agreement": 1 / len(draws),
                "needs_review": True,
            },
            {
                "text": "dirt bike",
                "type": "CONCEPT",
                "agreement": 1 / len(draws),
                "needs_review": True,
            },
            {
                "text": "bike",
                "type": "CONCEPT",
                "agreement": 1 / len(draws),
                "needs_review": True,
            },
        ],
    }
    assert len(draws) == SELF_CONSISTENCY_SAMPLES
    assert served.confidence == 1 / len(draws)
    assert served.status is ApprovalStatus.PENDING_REVIEW
