"""Phase 1 — KnowledgeRegistry + schema-driven retention end-to-end.

Exercises the shipped knowledge subsystem against the deployed e2e stack:
  * registry defaults (conservative permanent-by-default, exact seed set);
  * permanent-by-default kinds survive a cleanup tick;
  * EPHEMERAL_DAYS soft-delete then hard-delete window via real Vespa;
  * SCHEMA_DRIVEN cleanup hook (learned_strategy retirement);
  * EPHEMERAL_SESSION drop_session round-trip with negative-case enforcement;
  * pin survives a lifecycle scheduler tick (admin pin route + scheduler).

Mem0MemoryManager is built in-process and pointed at the deployed cluster's
Vespa NodePort + denseon NodePort — the same pattern integration tests use,
but here run against the actual cogniverse up cluster (not a per-test Vespa
container). Every test mints a fresh ``unique_id("know_")`` tenant so the
session-end sweep at ``_cleanup_test_tenants`` cleans up.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    ContradictionPolicy,
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    Retention,
    SchemaViolationError,
    Sensitivity,
    _retire_unconfirmed_strategy,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, run_async, skip_if_no_runtime, unique_id

# k3d-cogniverse-serverlb forwards these to the in-cluster cogniverse-vespa
# and cogniverse-denseon services. Verified via:
#   docker ps --filter name=k3d-cogniverse-serverlb --format '{{.Ports}}'
# which shows 8080->8080 (Vespa /query/), 19071->19071 (Vespa /config),
# and the 29004-29006 NodePort range (denseon at 29006).
VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    """Build a Mem0MemoryManager pointing at the deployed cluster.

    Uses ``auto_create_schema=True`` so the per-test tenant gets its
    ``agent_memories_<tenant>`` schema deployed on first use. Mem0's
    embedder hits the in-cluster denseon sidecar via the host NodePort.
    The LLM is unused here (every write uses ``infer=False``).
    """
    Mem0MemoryManager._instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=VESPA_HTTP_PORT,
    )
    cm = ConfigManager(store=config_store)
    # In-memory only: cm.set_system_config would persist a denseon-only
    # localhost URL map into config_metadata and starve the in-cluster
    # ingestor (which reads inference_service_urls from the same store).
    cm._system_config_cache = SystemConfig(  # noqa: SLF001
        backend_url="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        inference_service_urls={"denseon": DENSEON_URL},
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        backend_config_port=VESPA_CONFIG_PORT,
        base_schema_name="agent_memories",
        # LLM never invoked (we use infer=False everywhere); the
        # embedder is the only inference-side dependency.
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


def _provenance_metadata(kind: str, subject_key: str, source: str) -> dict:
    """Compose a minimal valid metadata dict for a provenance_required kind."""
    prov = make_provenance(
        written_by=f"agent:phase1_{kind}",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external(source, label=f"src_{kind}")],
    )
    base: dict = {"kind": kind, "subject_key": subject_key}
    return attach_to_metadata(base, prov)


def _seed(
    mm: Mem0MemoryManager,
    *,
    kind: str,
    content: str,
    metadata_extra: dict | None = None,
    age_days: float = 0.0,
    agent_name: str = "phase1_agent",
) -> str:
    """Add a memory with optional pre-set ``created_at`` for age tests.

    Mem0 stamps ``created_at`` from a C-level clock that ignores Python
    monkeypatching; the only deterministic way to age a memory is to
    pass ``created_at`` in metadata at write time (Mem0 honours it).
    """
    if kind in (
        "external_doc",
        "entity_fact",
        "kg_node",
        "kg_edge",
        "learned_strategy",
    ):
        meta = _provenance_metadata(
            kind=kind, subject_key=f"{kind}_subj", source="phase1://seed"
        )
    else:
        meta = {"kind": kind}
    if metadata_extra:
        meta.update(metadata_extra)
    if age_days > 0:
        backdated = datetime.now(timezone.utc) - timedelta(days=age_days)
        meta["created_at"] = backdated.isoformat()
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=meta,
        infer=False,
    )
    assert mid is not None, (
        f"Mem0.add returned None for kind={kind!r} content={content[:60]!r}; "
        f"metadata kept: {sorted(meta.keys())}"
    )
    return mid


def _memory_by_id(mm: Mem0MemoryManager, memory_id: str) -> dict | None:
    """Pull a memory by id from the per-tenant Mem0 store."""
    raw = mm.memory.get(memory_id)
    if raw is None:
        return None
    return raw if isinstance(raw, dict) else dict(raw)


# ---------------------------------------------------------------------------
# 1. Registry defaults
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRegistryDefaultsAreConservative:
    """KnowledgeRegistry's safe defaults + the seed set must be exact."""

    def test_unregistered_kind_returns_safe_default(self) -> None:
        registry = KnowledgeRegistry()
        schema = registry.get("unseen_kind_xyz")
        # Each field pinned to its exact default — drift here would silently
        # change the policy applied to every unregistered write.
        assert schema.kind == "unseen_kind_xyz"
        assert schema.retention is Retention.PERMANENT
        assert schema.sensitivity is Sensitivity.TENANT_PRIVATE
        assert schema.pinnable_by is Pinnable.TENANT_ADMIN
        assert schema.provenance_required is True
        assert schema.contradiction_policy is ContradictionPolicy.LATEST_WINS
        assert schema.default_trust == 0.5
        assert schema.retention_days is None
        assert schema.cleanup_hook is None

    def test_default_registry_seed_set(self) -> None:
        registry = build_default_registry()
        kinds = registry.all_kinds()
        # Exact set + sorted order, no drift permitted.
        assert kinds == sorted(
            [
                "conflict_set",
                "conversation_turn",
                "entity_fact",
                "external_doc",
                "kg_edge",
                "kg_node",
                "learned_strategy",
                "pin_record",
                "session_scratch",
                "tenant_instruction",
            ]
        )


# ---------------------------------------------------------------------------
# 2. Permanent-by-default kinds survive cleanup
# ---------------------------------------------------------------------------


_PERMANENT_KINDS = [
    "external_doc",
    "entity_fact",
    "kg_node",
    "kg_edge",
    "tenant_instruction",
]


@pytest.mark.e2e
@skip_if_no_runtime
class TestPermanentByDefault:
    """Every permanent-by-default kind must survive an unrestricted cleanup."""

    @pytest.mark.parametrize("kind", _PERMANENT_KINDS)
    def test_kind_survives_cleanup(self, kind: str) -> None:
        tenant_id = unique_id("know_perm")
        mm = _build_manager(tenant_id)
        try:
            mid = _seed(mm, kind=kind, content=f"phase1 permanent {kind}")
            registry = build_default_registry()
            result = mm.cleanup_with_schema(registry, set())
            # Permanent kinds must NOT contribute either a hard-delete
            # (kind key) or soft-delete (kind:archived key) entry.
            assert kind not in result
            assert f"{kind}:archived" not in result
            survived = _memory_by_id(mm, mid)
            assert survived is not None, (
                f"permanent kind {kind!r} memory {mid} disappeared after cleanup"
            )
        finally:
            mm.clear_agent_memory(tenant_id, "phase1_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. EPHEMERAL_DAYS soft-delete then hard-delete window
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRetentionEphemeralDays:
    """3-day TTL: 4d-old gets soft-deleted, 7d-old gets hard-deleted."""

    def test_soft_then_hard_delete_window(self) -> None:
        tenant_id = unique_id("know_eph")
        mm = _build_manager(tenant_id)
        try:
            # The write path uses mm._knowledge_registry (set in
            # _build_manager); the cleanup path takes its registry by
            # argument. Register the custom kind on BOTH so the write
            # passes schema validation AND cleanup uses the same policy.
            ephemeral_3d = KnowledgeSchema(
                kind="know_ephemeral_3d",
                retention=Retention.EPHEMERAL_DAYS,
                retention_days=3,
                sensitivity=Sensitivity.TENANT_PRIVATE,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
                contradiction_policy=ContradictionPolicy.LATEST_WINS,
                default_trust=0.4,
            )
            registry = build_default_registry()
            registry.register(ephemeral_3d, replace=True)
            mm._knowledge_registry.register(ephemeral_3d, replace=True)

            old_id = _seed(mm, kind="know_ephemeral_3d", content="aged 4d", age_days=4)
            fresh_id = _seed(
                mm, kind="know_ephemeral_3d", content="aged 1d", age_days=1
            )

            result1 = mm.cleanup_with_schema(registry, set())
            # 4d-old crosses the 3d soft-delete cutoff but not the 6d
            # hard-delete cutoff: archived bit set, exactly one
            # ":archived" entry. 1d-old is untouched.
            assert result1.get("know_ephemeral_3d:archived") == 1
            assert "know_ephemeral_3d" not in result1, result1
            old_mem = _memory_by_id(mm, old_id)
            assert old_mem is not None
            assert mm._read_metadata(old_mem).get("archived") is True
            fresh_mem = _memory_by_id(mm, fresh_id)
            assert fresh_mem is not None
            assert "archived" not in mm._read_metadata(fresh_mem)

            # Re-seed the SAME content with age=7d (past 2× TTL = 6d) to
            # force the hard-delete branch. Mem0 dedupes on content +
            # tenant + agent unless content differs, so use a distinct
            # content string for clarity.
            very_old_id = _seed(
                mm,
                kind="know_ephemeral_3d",
                content="aged 7d (past 2x TTL)",
                age_days=7,
            )
            result2 = mm.cleanup_with_schema(registry, set())
            assert result2.get("know_ephemeral_3d") == 1, result2
            assert _memory_by_id(mm, very_old_id) is None
        finally:
            mm.clear_agent_memory(tenant_id, "phase1_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 4. SCHEMA_DRIVEN cleanup hook — learned_strategy retirement
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRetentionSchemaDriven:
    """``_retire_unconfirmed_strategy`` retires only old + low-confirmation rows."""

    def test_only_unconfirmed_old_strategy_retires(self) -> None:
        tenant_id = unique_id("know_sd")
        mm = _build_manager(tenant_id)
        try:
            registry = build_default_registry()

            a_id = _seed(
                mm,
                kind="learned_strategy",
                content="A: low-confirm + old",
                metadata_extra={"confirmation_count": 1},
                age_days=40,
            )
            b_id = _seed(
                mm,
                kind="learned_strategy",
                content="B: low-confirm + young",
                metadata_extra={"confirmation_count": 1},
                age_days=10,
            )
            c_id = _seed(
                mm,
                kind="learned_strategy",
                content="C: high-confirm + old",
                metadata_extra={"confirmation_count": 4},
                age_days=40,
            )

            result = mm.cleanup_with_schema(registry, set())
            assert result.get("learned_strategy") == 1, result
            survivors_present = {b_id, c_id}
            for mid in survivors_present:
                assert _memory_by_id(mm, mid) is not None, (
                    f"survivor {mid} unexpectedly removed; result={result}"
                )
            assert _memory_by_id(mm, a_id) is None, (
                "A (low-confirm + 40d) should have been retired but survives"
            )
        finally:
            mm.clear_agent_memory(tenant_id, "phase1_agent")
            Mem0MemoryManager._instances.clear()

    def test_retire_hook_unit_contract(self) -> None:
        """Hook returns True only when confirmation_count<3 AND age>30d."""
        # A pure unit-shape assertion on the hook bound to the registry —
        # ensures the lifecycle behaviour above is what the hook returns.
        old_unconfirmed = {
            "id": "x",
            "metadata": {
                "kind": "learned_strategy",
                "confirmation_count": 1,
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=40)
                ).isoformat(),
            },
        }
        old_confirmed = {
            **old_unconfirmed,
            "metadata": {**old_unconfirmed["metadata"], "confirmation_count": 4},
        }
        young_unconfirmed = {
            **old_unconfirmed,
            "metadata": {
                **old_unconfirmed["metadata"],
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=10)
                ).isoformat(),
            },
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(old_unconfirmed, schema) is True
        assert _retire_unconfirmed_strategy(old_confirmed, schema) is False
        assert _retire_unconfirmed_strategy(young_unconfirmed, schema) is False


# ---------------------------------------------------------------------------
# 5. EPHEMERAL_SESSION drop_session + write enforcement
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRetentionEphemeralSession:
    """session_scratch survives lifecycle, drops on session-end; write must carry session_id."""

    def test_session_scratch_survives_lifecycle_dropped_on_session_end(self) -> None:
        tenant_id = unique_id("know_sess")
        mm = _build_manager(tenant_id)
        try:
            registry = build_default_registry()
            mid = _seed(
                mm,
                kind="session_scratch",
                content="phase1 session scratch",
                metadata_extra={"session_id": "sess_abc"},
            )

            # Lifecycle scheduler does NOT touch EPHEMERAL_SESSION memories.
            cleanup_result = mm.cleanup_with_schema(registry, set())
            assert "session_scratch" not in cleanup_result, cleanup_result
            assert "session_scratch:archived" not in cleanup_result, cleanup_result
            assert _memory_by_id(mm, mid) is not None

            # drop_session DOES — exact dict equality on the returned summary.
            drop_result = mm.drop_session("sess_abc", registry)
            assert drop_result == {"session_scratch": 1}
            assert _memory_by_id(mm, mid) is None
        finally:
            mm.clear_agent_memory(tenant_id, "phase1_agent")
            Mem0MemoryManager._instances.clear()

    def test_session_scratch_write_without_session_id_rejected(self) -> None:
        tenant_id = unique_id("know_sess_neg")
        mm = _build_manager(tenant_id)
        try:
            with pytest.raises(SchemaViolationError) as exc:
                mm.add_memory(
                    content="should be rejected",
                    tenant_id=tenant_id,
                    agent_name="phase1_agent",
                    metadata={"kind": "session_scratch"},  # no session_id
                    infer=False,
                )
            # Pin on the exact substring the schema validator uses so a
            # message rewrite breaks this test loudly.
            assert "requires metadata.session_id" in str(exc.value), exc.value
        finally:
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 6. Pinning survives a lifecycle scheduler tick
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPinningSurvivesLifecycle:
    """Pin via admin route + schedule tick → pinned memory survives past TTL."""

    def test_pin_survives_tick(self) -> None:
        tenant_id = unique_id("know_pin")
        mm = _build_manager(tenant_id)
        try:
            # See TestRetentionEphemeralDays for why the kind must be
            # registered on BOTH the local cleanup-time registry and
            # mm._knowledge_registry (the write-time validator).
            pin_ephemeral = KnowledgeSchema(
                kind="know_pin_ephemeral",
                retention=Retention.EPHEMERAL_DAYS,
                retention_days=1,
                sensitivity=Sensitivity.TENANT_PRIVATE,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
                contradiction_policy=ContradictionPolicy.LATEST_WINS,
                default_trust=0.4,
            )
            registry = build_default_registry()
            registry.register(pin_ephemeral, replace=True)
            mm._knowledge_registry.register(pin_ephemeral, replace=True)

            # Aged past 2× TTL so without pin protection it would
            # hard-delete on the next tick.
            mid = _seed(
                mm,
                kind="know_pin_ephemeral",
                content="phase1 pin candidate",
                age_days=5,
            )

            # Pin via the admin HTTP route. Per-tenant admin pin must
            # succeed (Pinnable.TENANT_ADMIN is the schema floor).
            with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/pin",
                    json={
                        "target_kind": "know_pin_ephemeral",
                        "pinned_by": "tenant_admin",
                        "actor_id": "phase1_admin",
                    },
                )
            assert resp.status_code == 200, (
                f"pin failed rc={resp.status_code}: {resp.text[:300]}"
            )
            body = resp.json()
            assert body["target_memory_id"] == mid
            assert body["pinned_by"] == "tenant_admin"

            # Run the scheduler against this single warm manager. The
            # pin_lookup returns {mid} so the cleanup pathway treats it
            # as pinned and skips it.
            scheduler = LifecycleScheduler(
                get_warm_managers=lambda: [mm],
                registry=registry,
                interval_seconds=3600.0,
                pin_lookup=lambda _mgr: {mid},
            )
            summary = run_async(scheduler.tick_once())

            per_tenant = summary.get("tenants", {}).get(tenant_id, {})
            # Pinned memory was the only candidate that would have been
            # touched; the kind key must report 0 deletions for this tenant.
            assert per_tenant.get("know_pin_ephemeral", 0) == 0, per_tenant
            assert per_tenant.get("know_pin_ephemeral:archived", 0) == 0, per_tenant
            assert _memory_by_id(mm, mid) is not None, (
                "pinned memory was deleted despite pin_lookup returning its id"
            )
        finally:
            try:
                with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
                    client.request(
                        "DELETE",
                        f"/admin/tenants/{tenant_id}/memories/{mid}/pin",
                        json={
                            "requester_role": "tenant_admin",
                            "actor_id": "phase1_admin",
                        },
                    )
            except Exception:
                pass
            mm.clear_agent_memory(tenant_id, "phase1_agent")
            Mem0MemoryManager._instances.clear()
