"""A.9 integration test — LifecycleScheduler against real Mem0 + real Vespa.

Spins up the shared memory Vespa + denseon fixtures, registers two tenants
with the warm Mem0 cache, seeds memories with manipulated ``created_at``
timestamps so the scheduler's tick produces a deterministic delete count,
and asserts the surviving memory set against a fresh Vespa search.

Skips on environments where the shared fixtures cannot start (e.g. no Docker
or no denseon service available).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT_RECENT = "lifecycle_recent"
TENANT_OLD = "lifecycle_old"


def _build_manager(
    tenant_id: str,
    vespa_info: dict,
    denseon_url: str,
    config_manager: ConfigManager,
) -> Mem0MemoryManager:
    """Build and initialize a tenant-scoped Mem0 manager against shared Vespa."""
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=vespa_info["http_port"],
        backend_config_port=vespa_info["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://localhost:11434",
        embedder_base_url=denseon_url,
        auto_create_schema=False,
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    return mm


@pytest.fixture(scope="module")
def lifecycle_managers(shared_memory_vespa, shared_denseon):
    """Warm two tenants in the Mem0 LRU cache with seeded memories."""
    Mem0MemoryManager._instances.clear()

    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )

    recent = _build_manager(TENANT_RECENT, shared_memory_vespa, shared_denseon, cm)
    old = _build_manager(TENANT_OLD, shared_memory_vespa, shared_denseon, cm)

    # Both tenants get one fresh memory and one old memory each. The
    # `time.time` patch makes Mem0 stamp the old memories with an epoch
    # 60 days in the past so the scheduler's 30-day max_age cleans them up.
    sixty_days_ago = int(time.time() - 60 * 24 * 3600)

    real_time = time.time
    try:
        # Seed old entries first (under patched clock).
        time.time = lambda: float(sixty_days_ago)  # type: ignore[assignment]
        recent.add_memory(
            content="recent-tenant old fact about rabbits",
            tenant_id=TENANT_RECENT,
            agent_name="lifecycle_test",
            metadata={"era": "old"},
            infer=False,
        )
        old.add_memory(
            content="old-tenant ancient fact about pyramids",
            tenant_id=TENANT_OLD,
            agent_name="lifecycle_test",
            metadata={"era": "old"},
            infer=False,
        )
    finally:
        time.time = real_time  # type: ignore[assignment]

    # Now seed fresh entries with the real clock.
    recent.add_memory(
        content="recent-tenant fresh fact about lasers",
        tenant_id=TENANT_RECENT,
        agent_name="lifecycle_test",
        metadata={"era": "fresh"},
        infer=False,
    )
    old.add_memory(
        content="old-tenant fresh fact about volcanoes",
        tenant_id=TENANT_OLD,
        agent_name="lifecycle_test",
        metadata={"era": "fresh"},
        infer=False,
    )

    yield recent, old

    # Best-effort teardown.
    for mm, t in [(recent, TENANT_RECENT), (old, TENANT_OLD)]:
        try:
            mm.clear_agent_memory(t, "lifecycle_test")
        except Exception:
            pass
    Mem0MemoryManager._instances.clear()


@pytest.mark.asyncio
async def test_scheduler_tick_deletes_old_memories_across_warm_tenants(
    lifecycle_managers,
):
    """Real Mem0 + Vespa: scheduler tick must delete only the >30d entries."""
    recent, old = lifecycle_managers

    # Sanity: each tenant currently has 2 memories.
    pre_recent = recent.get_all_memories(TENANT_RECENT, "lifecycle_test")
    pre_old = old.get_all_memories(TENANT_OLD, "lifecycle_test")
    assert len(pre_recent) == 2, f"expected 2 seeded memories; got {pre_recent}"
    assert len(pre_old) == 2, f"expected 2 seeded memories; got {pre_old}"

    scheduler = LifecycleScheduler(
        get_warm_managers=Mem0MemoryManager._instances.values,
        interval_seconds=3600.0,
        max_age_seconds=30 * 24 * 3600,
    )
    summary = await scheduler.tick_once()

    # Each warm tenant deleted exactly one expired memory (the 60-day-old one).
    assert summary["tenants"][TENANT_RECENT] == 1, summary
    assert summary["tenants"][TENANT_OLD] == 1, summary
    assert summary["total_deleted"] == 2

    # Survivor checks: only the fresh memory remains in each tenant.
    post_recent = recent.get_all_memories(TENANT_RECENT, "lifecycle_test")
    post_old = old.get_all_memories(TENANT_OLD, "lifecycle_test")
    assert len(post_recent) == 1
    assert len(post_old) == 1
    assert "fresh fact" in post_recent[0]["memory"]
    assert "fresh fact" in post_old[0]["memory"]
