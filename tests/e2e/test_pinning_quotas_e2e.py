"""Phase 4b — Pin quotas + soft-delete restore end-to-end.

Pins the shipped PinService quota path + the admin restore route:

  * per-role quotas configured via ``PUT /admin/tenants/{t}/pin_quotas``
    are enforced — third pin beyond a quota of 2 returns 429;
  * org_admin pins succeed past the user quota (override authority);
  * schema pinnable_by floor rejects role≠floor with HTTP 403;
  * ``GET /admin/tenants/{t}/pins`` returns the canonical PinListResponse;
  * ``POST /admin/tenants/{t}/memories/{m}/restore`` clears the
    ``archived`` flag on a soft-deleted memory and persists the change.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    ContradictionPolicy,
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
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
    )
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=VESPA_HTTP_PORT,
            inference_service_urls={"denseon": DENSEON_URL},
        )
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        backend_config_port=VESPA_CONFIG_PORT,
        base_schema_name="agent_memories",
        llm_model="google/gemma-4-e4b-it",
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://cogniverse-vllm-llm-student.cogniverse:8000/v1",
        embedder_base_url=DENSEON_URL,
        auto_create_schema=True,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    return mm


def _write_entity_fact(
    mm: Mem0MemoryManager,
    *,
    subject_key: str,
    content: str,
    agent_name: str = "phase4_agent",
) -> str:
    """Write an entity_fact (provenance_required, pinnable_by=tenant_admin)."""
    prov = make_provenance(
        written_by="agent:phase4",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("phase4://src")],
    )
    metadata = attach_to_metadata(
        {"kind": "entity_fact", "subject_key": subject_key}, prov
    )
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


def _set_pin_quota(tenant_id: str, **quotas: int) -> dict:
    with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
        resp = client.put(
            f"/admin/tenants/{tenant_id}/pin_quotas",
            json=quotas,
        )
    assert resp.status_code == 200, resp.text[:300]
    return resp.json()


def _pin(
    tenant_id: str, memory_id: str, *, target_kind: str, pinned_by: str, actor_id: str
) -> httpx.Response:
    with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
        return client.post(
            f"/admin/tenants/{tenant_id}/memories/{memory_id}/pin",
            json={
                "target_kind": target_kind,
                "pinned_by": pinned_by,
                "actor_id": actor_id,
            },
        )


# ---------------------------------------------------------------------------
# 1. user quota = 2 → third pin rejected
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestUserQuotaEnforced:
    """Setting user pin quota to 2 caps the user role's pins at 2."""

    def test_third_user_pin_rejected_with_429(self) -> None:
        tenant_id = unique_id("know_quota") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            # entity_fact's pinnable_by is tenant_admin (floor), so a user
            # cannot pin it even without quota. Use conversation_turn —
            # pinnable_by=USER — so the role floor allows the user role
            # and quota becomes the only gate.
            cnvs = []
            for i in range(3):
                mid = mm.add_memory(
                    content=f"turn {i}",
                    tenant_id=tenant_id,
                    agent_name="phase4_agent",
                    metadata={"kind": "conversation_turn", "subject_key": f"c{i}"},
                    infer=False,
                )
                assert mid is not None
                cnvs.append(mid)

            _set_pin_quota(tenant_id, user=2)

            r1 = _pin(
                tenant_id,
                cnvs[0],
                target_kind="conversation_turn",
                pinned_by="user",
                actor_id="alice",
            )
            assert r1.status_code == 200, r1.text[:300]
            r2 = _pin(
                tenant_id,
                cnvs[1],
                target_kind="conversation_turn",
                pinned_by="user",
                actor_id="alice",
            )
            assert r2.status_code == 200, r2.text[:300]
            r3 = _pin(
                tenant_id,
                cnvs[2],
                target_kind="conversation_turn",
                pinned_by="user",
                actor_id="alice",
            )
            assert r3.status_code == 429, r3.text[:300]
            detail = r3.json().get("detail", "")
            assert "quota" in detail.lower(), detail
            assert "user" in detail.lower(), detail
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase4_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. org_admin role overrides user quota
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestOrgAdminOverridesQuota:
    """user quota=2 capped users; org_admin pins past it succeed."""

    def test_org_admin_pin_past_user_quota_succeeds(self) -> None:
        tenant_id = unique_id("know_qua") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            cnvs = []
            for i in range(3):
                mid = mm.add_memory(
                    content=f"turn {i}",
                    tenant_id=tenant_id,
                    agent_name="phase4_agent",
                    metadata={"kind": "conversation_turn", "subject_key": f"c{i}"},
                    infer=False,
                )
                assert mid is not None
                cnvs.append(mid)

            _set_pin_quota(tenant_id, user=2)

            for mid in cnvs[:2]:
                r = _pin(
                    tenant_id,
                    mid,
                    target_kind="conversation_turn",
                    pinned_by="user",
                    actor_id="alice",
                )
                assert r.status_code == 200, r.text[:300]

            r3 = _pin(
                tenant_id,
                cnvs[2],
                target_kind="conversation_turn",
                pinned_by="org_admin",
                actor_id="boss",
            )
            assert r3.status_code == 200, r3.text[:300]
            body3 = r3.json()
            assert body3["target_memory_id"] == cnvs[2]
            assert body3["pinned_by"] == "org_admin"
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase4_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. schema's pinnable_by floor is enforced (HTTP 403)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSchemaPinnableFloorRejected:
    """kg_node has pinnable_by=tenant_admin; user role cannot pin."""

    def test_user_pin_of_kg_node_returns_403(self) -> None:
        tenant_id = unique_id("know_floor") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            prov = make_provenance(
                written_by="agent:phase4",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                confidence=0.9,
                derived_from=[CitationRef.external("phase4://src")],
            )
            metadata = attach_to_metadata(
                {"kind": "kg_node", "subject_key": "node_x"}, prov
            )
            mid = mm.add_memory(
                content="kg node x",
                tenant_id=tenant_id,
                agent_name="phase4_agent",
                metadata=metadata,
                infer=False,
            )
            assert mid is not None

            r = _pin(
                tenant_id,
                mid,
                target_kind="kg_node",
                pinned_by="user",
                actor_id="alice",
            )
            assert r.status_code == 403, r.text[:300]
            detail = r.json().get("detail", "")
            assert "forbids pin from role=user" in detail, detail
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase4_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 4. GET /pins returns the exact list shape
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestListPinsRoundTrip:
    """After pinning 2 memories, GET /pins returns exactly those two ids."""

    def test_two_pins_listed_with_correct_target_ids(self) -> None:
        tenant_id = unique_id("know_list") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            mids = []
            for i in range(2):
                mid = mm.add_memory(
                    content=f"turn {i}",
                    tenant_id=tenant_id,
                    agent_name="phase4_agent",
                    metadata={"kind": "conversation_turn", "subject_key": f"c{i}"},
                    infer=False,
                )
                assert mid is not None
                mids.append(mid)

            for mid in mids:
                r = _pin(
                    tenant_id,
                    mid,
                    target_kind="conversation_turn",
                    pinned_by="user",
                    actor_id="alice",
                )
                assert r.status_code == 200, r.text[:300]

            with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
                resp = client.get(f"/admin/tenants/{tenant_id}/pins")
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["tenant_id"] == tenant_id
            assert len(body["pins"]) == 2
            assert sorted(p["target_memory_id"] for p in body["pins"]) == sorted(mids)
            for p in body["pins"]:
                assert p["target_kind"] == "conversation_turn"
                assert p["pinned_by"] == "user"
                assert p["pinned_by_actor"] == "alice"
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase4_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 5. Soft-deleted memory restored via admin route
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRestoreSoftDeletedMemory:
    """Aged-past-TTL memory → cleanup_with_schema soft-deletes → POST /restore clears archived."""

    def test_restore_clears_archived_flag(self) -> None:
        tenant_id = unique_id("know_rest") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            # Register a 1-day ephemeral kind on mm's write-time registry
            # AND a local copy used at cleanup time. Phase 1 caught the
            # gotcha that these must be the same instance or kind.
            ephemeral = KnowledgeSchema(
                kind="know_rest_ephemeral",
                retention=Retention.EPHEMERAL_DAYS,
                retention_days=1,
                sensitivity=Sensitivity.TENANT_PRIVATE,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
                contradiction_policy=ContradictionPolicy.LATEST_WINS,
                default_trust=0.4,
            )
            registry = build_default_registry()
            registry.register(ephemeral, replace=True)
            mm._knowledge_registry.register(ephemeral, replace=True)

            # Aged 2 days → past 1-day TTL, but under 2× TTL (= 2 days).
            # So cleanup soft-deletes (archives) but does not hard-delete.
            backdated = (
                datetime.now(timezone.utc) - timedelta(days=1, hours=12)
            ).isoformat()
            mid = mm.add_memory(
                content="aged memory awaiting restore",
                tenant_id=tenant_id,
                agent_name="phase4_agent",
                metadata={
                    "kind": "know_rest_ephemeral",
                    "subject_key": "restore_subj",
                    "created_at": backdated,
                },
                infer=False,
            )
            assert mid is not None

            # Soft-delete via cleanup tick.
            result = mm.cleanup_with_schema(registry, set())
            assert result.get("know_rest_ephemeral:archived") == 1, result
            archived_mem = mm.memory.get(mid)
            assert archived_mem is not None
            archived_meta = archived_mem.get("metadata") or {}
            if isinstance(archived_meta, str):
                archived_meta = json.loads(archived_meta)
            assert archived_meta.get("archived") is True

            # Restore via the admin route.
            with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/restore",
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["tenant_id"] == tenant_id
            assert body["memory_id"] == mid
            assert body["restored"] is True

            # Re-fetch; archived flag is gone (either absent or False).
            restored_mem = mm.memory.get(mid)
            assert restored_mem is not None
            restored_meta = restored_mem.get("metadata") or {}
            if isinstance(restored_meta, str):
                restored_meta = json.loads(restored_meta)
            assert not restored_meta.get("archived"), restored_meta
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase4_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
