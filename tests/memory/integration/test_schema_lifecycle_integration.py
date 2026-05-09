"""A.7 integration — schema-driven lifecycle against real Mem0 + Vespa.

Seeds memories of three different kinds (one permanent, one ephemeral_days,
one schema_driven) with manipulated ``created_at`` so the scheduler's tick
produces deterministic per-kind deletion counts. Verifies pinned memories
are never deleted regardless of policy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.pinning import PinService
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    Retention,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "a7_lifecycle_tenant"
AGENT = "a7_lifecycle_agent"


@pytest.fixture(scope="module")
def lifecycle_env(shared_memory_vespa, shared_denseon):
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
    mm = Mem0MemoryManager(tenant_id=TENANT)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://localhost:11434",
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )

    yield mm, cm

    try:
        mm.clear_agent_memory(TENANT, AGENT)
        mm.clear_agent_memory(TENANT, "_pinning")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _registry_for_test() -> KnowledgeRegistry:
    """Registry with deterministic per-kind retention for the test."""
    reg = build_default_registry()

    # Deletes anything older than 7 days
    reg.register(
        KnowledgeSchema(
            kind="ephemeral_test_kind",
            retention=Retention.EPHEMERAL_DAYS,
            retention_days=7,
            provenance_required=False,
            pinnable_by=Pinnable.USER,
        ),
        replace=True,
    )

    # Deletes when a hook returns True. Hook here: deletes anything whose
    # content contains the marker "DELETE_ME".
    def _hook(memory, schema):
        return "DELETE_ME" in (memory.get("memory") or memory.get("content") or "")

    reg.register(
        KnowledgeSchema(
            kind="schema_driven_test_kind",
            retention=Retention.SCHEMA_DRIVEN,
            provenance_required=False,
            pinnable_by=Pinnable.USER,
            cleanup_hook=_hook,
        ),
        replace=True,
    )

    # Permanent: never deletes (default schema). Use a dedicated kind so the
    # test isolates from the seed registry's defaults.
    reg.register(
        KnowledgeSchema(
            kind="permanent_test_kind",
            retention=Retention.PERMANENT,
            provenance_required=False,
            pinnable_by=Pinnable.USER,
        ),
        replace=True,
    )
    return reg


def _seed(mm, kind: str, content: str, age_days: int = 0) -> str:
    """Seed a memory under TENANT/AGENT with optional back-dated created_at.

    mem0 stamps ``created_at`` on add via ``datetime.now(timezone.utc)`` —
    a C-level clock that doesn't respect a Python-level ``time.time``
    monkeypatch. The only reliable way to back-date a memory is to inject
    ``created_at`` into the metadata so mem0 honours it directly
    (mem0/memory/main.py respects a pre-set ``created_at`` in metadata).
    """
    from datetime import datetime, timedelta, timezone

    metadata = {"kind": kind}
    if age_days > 0:
        backdated = datetime.now(timezone.utc) - timedelta(days=age_days)
        metadata["created_at"] = backdated.isoformat()
    return mm.add_memory(
        content=content,
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=metadata,
        infer=False,
    )


@pytest.mark.asyncio
async def test_schema_driven_tick_per_kind_against_real_vespa(lifecycle_env):
    mm, _cm = lifecycle_env
    registry = _registry_for_test()

    # Seed: 1 permanent, 1 ephemeral old (>7d), 1 ephemeral fresh (<7d),
    # 1 schema-driven hook=True, 1 schema-driven hook=False.
    permanent_id = _seed(mm, "permanent_test_kind", "permanent content")
    ephemeral_old_id = _seed(mm, "ephemeral_test_kind", "ephemeral old", age_days=10)
    ephemeral_fresh_id = _seed(mm, "ephemeral_test_kind", "ephemeral fresh", age_days=0)
    hook_delete_id = _seed(mm, "schema_driven_test_kind", "this is DELETE_ME content")
    hook_keep_id = _seed(mm, "schema_driven_test_kind", "this is keepable")

    scheduler = LifecycleScheduler(
        get_warm_managers=Mem0MemoryManager._instances.values,
        registry=registry,
    )
    summary = await scheduler.tick_once()

    per_kind = summary["tenants"][TENANT]
    # Soft-delete: age 10d is in the (cutoff, 2×cutoff] window → archived.
    # schema_driven hook returns True for DELETE_ME → hard-deleted.
    assert per_kind.get("ephemeral_test_kind:archived", 0) == 1, summary
    assert per_kind.get("schema_driven_test_kind", 0) == 1, summary
    assert "permanent_test_kind" not in per_kind, summary

    # Default get_all_memories filters archived; ephemeral_old_id must not appear.
    surviving = mm.get_all_memories(TENANT, AGENT)
    surviving_ids = {m["id"] for m in surviving}
    assert permanent_id in surviving_ids
    assert ephemeral_fresh_id in surviving_ids
    assert hook_keep_id in surviving_ids
    assert ephemeral_old_id not in surviving_ids
    assert hook_delete_id not in surviving_ids

    # With include_archived=True, the soft-deleted memory is still present.
    with_archived = mm.get_all_memories(TENANT, AGENT, include_archived=True)
    with_archived_ids = {m["id"] for m in with_archived}
    assert ephemeral_old_id in with_archived_ids, (
        "soft-deleted memory must be retrievable via include_archived=True"
    )


@pytest.mark.asyncio
async def test_pinned_memory_survives_schema_driven_tick(lifecycle_env):
    mm, _cm = lifecycle_env
    registry = _registry_for_test()

    # Seed two old ephemeral memories; pin one of them.
    old_pinned = _seed(
        mm, "ephemeral_test_kind", "would be cleaned but pinned", age_days=20
    )
    old_unpinned = _seed(mm, "ephemeral_test_kind", "doomed by lifecycle", age_days=20)

    pin_svc = PinService(mm, registry)
    pin_svc.pin(
        target_memory_id=old_pinned,
        target_kind="ephemeral_test_kind",
        pinned_by=Pinnable.USER,
        actor_id="user_alpha",
        tenant_id=TENANT,
    )

    def _pin_lookup(manager) -> set:
        return {
            rec.target_memory_id
            for rec in PinService(manager, registry).list_pins(TENANT)
        }

    scheduler = LifecycleScheduler(
        get_warm_managers=Mem0MemoryManager._instances.values,
        registry=registry,
        pin_lookup=_pin_lookup,
    )
    await scheduler.tick_once()

    surviving = {m["id"] for m in mm.get_all_memories(TENANT, AGENT)}
    assert old_pinned in surviving, "pinned memory must not be deleted"
    assert old_unpinned not in surviving, "unpinned old memory must be deleted"
