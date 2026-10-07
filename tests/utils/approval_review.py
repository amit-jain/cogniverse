"""A tenant's synthetic review queue in the production approval store, for
the approval route and web client tests."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from tests.utils.memory_store import InMemoryConfigStore

ROUTING = {
    "query": "find the lecture on gradient descent",
    "entities": [{"text": "gradient descent", "type": "CONCEPT"}],
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


def run_in_own_loop(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coroutine`` to completion on a thread with its own event loop,
    so callers inside a running loop (sync Playwright holds one) can use it."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def review_config_manager(phoenix, redis_url, *, telemetry_url=None) -> ConfigManager:
    """An in-memory config store whose system config points the approval
    store at ``phoenix`` and ``redis_url``."""
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        SystemConfig(
            telemetry_url=telemetry_url or phoenix["http_endpoint"],
            telemetry_collector_endpoint=phoenix["grpc_endpoint"],
            redis_url=redis_url,
        )
    )
    return manager


def save_review_batch(storage: ApprovalStorageImpl, batch_id: str) -> None:
    """Save a batch with a routing and a workflow item awaiting review and one
    auto-approved item, ``{batch_id}_routing``, ``_workflow`` and
    ``_confident``."""
    tenant_id = storage.tenant_id
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
    run_in_own_loop(
        storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=items,
                context={"tenant_id": tenant_id, "optimizer": "routing"},
            )
        )
    )


def approved_rows(storage, item_id) -> int:
    """How many rows of the tenant's approved training dataset hold ``item_id``."""
    try:
        frame = run_in_own_loop(
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


def approved_rows_until(storage, item_id, want, timeout=30.0) -> int:
    """``approved_rows`` once it equals ``want``, or its value at ``timeout``
    (Phoenix serves dataset rows after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while (count := approved_rows(storage, item_id)) != want:
        if time.monotonic() > deadline:
            return count
        time.sleep(2)
    return count
