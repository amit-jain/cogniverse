"""soft-delete lifecycle: archive at TTL, restore, hard-delete at 2x TTL.

this was missing soft-delete entirely: the lifecycle
scheduler hard-deleted at TTL with no restore window. The plan
required ``archived: bool`` with a 90-day undelete window before hard
delete; default reads filter archived; admin restore endpoint surfaces
the soft-deleted record.

This test verifies, against real Vespa:

  * a memory past its kind's TTL but before 2x TTL has its
    ``metadata.archived=true`` flipped (NOT hard-deleted);
  * default reads (search + get_all) hide archived memories;
  * ``include_archived=True`` surfaces them for admin tooling;
  * ``restore_archived_memory`` clears the flag and the memory
    reappears in default reads;
  * a memory past 2x TTL is hard-deleted by the next tick.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    Pinnable,
    Retention,
    Sensitivity,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

pytestmark = pytest.mark.integration

TENANT = "test_tenant"
AGENT = "h8_lifecycle"
KIND = "h8_ephemeral"


def _registry_with_short_ttl() -> "object":
    """Registry where ``h8_ephemeral`` has a 1-day TTL — easy synthetic ageing."""
    reg = build_default_registry()
    reg.register(
        KnowledgeSchema(
            kind=KIND,
            retention=Retention.EPHEMERAL_DAYS,
            retention_days=1,  # TTL=1 day, hard-delete at 2 days
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=False,
            default_trust=0.5,
        ),
        replace=True,
    )
    return reg


@pytest.fixture(scope="module")
def lifecycle_mm(shared_memory_vespa, shared_denseon):
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
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=_registry_with_short_ttl(),
    )
    yield mm
    try:
        mm.clear_agent_memory(TENANT, AGENT)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _seed_aged(mm, content: str, age_days: float) -> str:
    """Seed a memory whose created_at is back-dated by age_days."""
    backdated = datetime.now(timezone.utc) - timedelta(days=age_days)
    return mm.add_memory(
        content=content,
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata={
            "kind": KIND,
            "created_at": backdated.isoformat(),
        },
        infer=False,
    )


class TestSoftDeleteCycle:
    def test_past_ttl_under_2x_archives_not_deletes(self, lifecycle_mm):
        mm = lifecycle_mm
        # 1.5 days old: past TTL=1 but under 2*TTL=2 → archive.
        mid = _seed_aged(mm, "h8 archive me", age_days=1.5)
        assert mid

        deleted_by_kind = mm.cleanup_with_schema(_registry_with_short_ttl())
        # The archive bucket is named "{kind}:archived" by the lifecycle hook.
        assert deleted_by_kind.get(f"{KIND}:archived", 0) >= 1, (
            f"memory at age 1.5*TTL should be archived (not hard-deleted); "
            f"got buckets={deleted_by_kind!r}"
        )

        # Default read MUST hide it.
        default = mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)
        assert not any(str(r.get("id")) == mid for r in default), (
            "archived memory must not appear in default get_all_memories"
        )

        # Admin read with include_archived=True surfaces it.
        with_archived = mm.get_all_memories(
            tenant_id=TENANT, agent_name=AGENT, include_archived=True
        )
        archived_row = next((r for r in with_archived if str(r.get("id")) == mid), None)
        assert archived_row is not None, (
            "include_archived=True must surface the soft-deleted memory; "
            f"got rows={[r.get('id') for r in with_archived]!r}"
        )
        meta = archived_row.get("metadata") or {}
        assert meta.get("archived") is True
        assert meta.get("archived_at"), (
            "archived_at timestamp missing on the soft-deleted record"
        )

    def test_restore_clears_archived_flag(self, lifecycle_mm):
        mm = lifecycle_mm
        # Reuse archived state from previous test if present, else create.
        with_archived = mm.get_all_memories(
            tenant_id=TENANT, agent_name=AGENT, include_archived=True
        )
        archived_rows = [
            r
            for r in with_archived
            if (r.get("metadata") or {}).get("archived") is True
        ]
        if not archived_rows:
            mid = _seed_aged(mm, "h8 restore-me-too", age_days=1.5)
            mm.cleanup_with_schema(_registry_with_short_ttl())
            with_archived = mm.get_all_memories(
                tenant_id=TENANT, agent_name=AGENT, include_archived=True
            )
            archived_rows = [
                r
                for r in with_archived
                if (r.get("metadata") or {}).get("archived") is True
                and str(r.get("id")) == mid
            ]
        assert archived_rows, "test setup failed to produce an archived memory"
        mid = str(archived_rows[0].get("id"))

        ok = mm.restore_archived_memory(mid)
        assert ok is True, "restore must report success on an archived memory"

        # After restore, it shows up in default reads again.
        default = mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)
        restored = next((r for r in default if str(r.get("id")) == mid), None)
        assert restored is not None, (
            f"restored memory {mid} must reappear in default reads"
        )
        meta = restored.get("metadata") or {}
        assert "archived" not in meta or meta.get("archived") is not True
        assert "archived_at" not in meta

    def test_restore_unknown_memory_returns_false(self, lifecycle_mm):
        assert lifecycle_mm.restore_archived_memory("no-such-id") is False

    def test_past_2x_ttl_hard_deletes(self, lifecycle_mm):
        mm = lifecycle_mm
        # 3 days old: past 2*TTL=2 → hard-delete.
        mid = _seed_aged(mm, "h8 hard delete me", age_days=3.0)
        assert mid

        deleted_by_kind = mm.cleanup_with_schema(_registry_with_short_ttl())
        # Hard-delete bucket uses the bare kind name.
        assert deleted_by_kind.get(KIND, 0) >= 1, (
            f"memory at age 3*TTL should be hard-deleted; got buckets={deleted_by_kind!r}"
        )

        # Even include_archived doesn't surface it — it's gone.
        full = mm.get_all_memories(
            tenant_id=TENANT, agent_name=AGENT, include_archived=True
        )
        assert not any(str(r.get("id")) == mid for r in full), (
            "hard-deleted memory must not appear under include_archived=True either"
        )


class _ArchiveSpyVectorStore:
    def __init__(self):
        self.update_calls: list[dict] = []

    def update(self, vector_id, vector=None, payload=None):
        self.update_calls.append(
            {"vector_id": vector_id, "vector": vector, "payload": payload}
        )


class _ArchiveSpyMemory:
    """Mem0 stand-in exposing the partial-update store and recording any
    (forbidden) ``Memory.update`` call — that path re-embeds the text."""

    def __init__(self, rows):
        self._rows = rows
        self.vector_store = _ArchiveSpyVectorStore()
        self.reembedding_update_calls: list[str] = []

    def get_all(self, user_id=None):
        return {"results": list(self._rows)}

    def update(self, memory_id=None, data=None, metadata=None):
        self.reembedding_update_calls.append(memory_id)


def _spy_mm() -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
    mm = Mem0MemoryManager(tenant_id="archive_spy")
    mm._initialized = True
    mm.tenant_id = "archive_spy"
    mm.config = None
    mm._knowledge_registry = None
    return mm


@pytest.mark.unit
@pytest.mark.ci_fast
class TestArchiveRestoreDoNotReembed:
    """Archive and restore only toggle ``metadata.archived`` — the text is
    unchanged, so they must route through the vector store's partial update
    (``vector=None`` keeps the stored embedding) instead of Mem0's
    ``Memory.update``, which re-embeds the text over HTTP on every call."""

    def test_archive_flips_flag_without_reembedding(self):
        mm = _spy_mm()
        spy = _ArchiveSpyMemory([])
        mm.memory = spy

        mm._archive_memory("m1", {"kind": "external_doc"}, existing_data="the fact")

        assert spy.reembedding_update_calls == [], (
            "archive must not call Memory.update — that re-embeds the text"
        )
        assert len(spy.vector_store.update_calls) == 1
        call = spy.vector_store.update_calls[0]
        assert call["vector_id"] == "m1"
        assert call["vector"] is None, "vector=None keeps the stored embedding"
        assert call["payload"]["data"] == "the fact", "text is preserved unchanged"
        md = call["payload"]["metadata"]
        assert md["archived"] is True
        assert md["archived_at"]
        assert md["kind"] == "external_doc", "existing metadata keys survive"

    def test_restore_clears_flag_without_reembedding(self):
        mm = _spy_mm()
        archived_row = {
            "id": "m1",
            "memory": "the fact",
            "metadata": {
                "kind": "external_doc",
                "archived": True,
                "archived_at": "2026-01-01T00:00:00+00:00",
            },
        }
        spy = _ArchiveSpyMemory([archived_row])
        mm.memory = spy

        assert mm.restore_archived_memory("m1") is True

        assert spy.reembedding_update_calls == [], (
            "restore must not call Memory.update — that re-embeds the text"
        )
        assert len(spy.vector_store.update_calls) == 1
        call = spy.vector_store.update_calls[0]
        assert call["vector"] is None
        assert call["payload"]["data"] == "the fact"
        md = call["payload"]["metadata"]
        assert "archived" not in md, "restore clears the archived flag"
        assert "archived_at" not in md
        assert md["kind"] == "external_doc"
