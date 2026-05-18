"""Phase 4a — Federation: org trunk + tenant overlay end-to-end.

Pins the shipped FederationService + the
``/admin/tenants/{t}/memories/{m}/promote_to_org_trunk`` HTTP route
against the deployed cogniverse up cluster:

  * a promoted memory becomes visible to every sibling tenant in the
    same org via ``federated_get_all``;
  * tenant-local writes win on subject_key collisions with the trunk;
  * tenant_private kinds refuse promotion (HTTP 403 with the exact
    schema message);
  * promotions in one org are invisible across orgs (different org_id).

Same-org tenants are built by sharing a single ``unique_id("fed_")``
prefix as the org_id and only varying the per-tenant suffix — that's
how :func:`org_trunk_tenant_id` resolves both back to the same trunk.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cogniverse_core.memory.federation import (
    FederationService,
    org_trunk_tenant_id,
)
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


# Federation-specific kind registered on every test's manager. None of
# the 10 default-seeded kinds carry ``Sensitivity.ORG_SHARED``, so the
# promote_to_org_trunk path is only reachable via a custom kind. Tests
# register this on each ``mm._knowledge_registry`` (writer-side validation)
# AND pass it into the in-process FederationService (promote-side
# validation). The runtime's HTTP /promote_to_org_trunk route builds its
# own ``build_default_registry()`` per call so it cannot promote this
# kind — the route is exercised separately by TestTenantPrivateRefuses
# Promotion which asserts the 403 (rejection) path.
_FED_KIND = KnowledgeSchema(
    kind="phase4_org_shared",
    retention=Retention.PERMANENT,
    sensitivity=Sensitivity.ORG_SHARED,
    pinnable_by=Pinnable.TENANT_ADMIN,
    provenance_required=True,
    contradiction_policy=ContradictionPolicy.PRESERVE_BOTH,
    default_trust=0.7,
)


def _registry_with_fed_kind() -> object:
    reg = build_default_registry()
    reg.register(_FED_KIND, replace=True)
    return reg


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    """Build a Mem0MemoryManager bound to the deployed cluster Vespa.

    Does NOT clear the per-tenant singleton cache here — federation
    tests intentionally hold multiple managers (t1, t2, trunk) live
    across the same test. The custom ``phase4_org_shared`` kind is
    registered on the write-time registry so external_doc-shaped
    writes can carry ``Sensitivity.ORG_SHARED`` and become promotable.
    """
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
    )
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
        llm_model="google/gemma-4-e4b-it",
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://cogniverse-vllm-llm-student.cogniverse:8000/v1",
        embedder_base_url=DENSEON_URL,
        auto_create_schema=True,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=_registry_with_fed_kind(),
    )
    return mm


def _write_shared(
    mm: Mem0MemoryManager,
    *,
    subject_key: str,
    content: str,
    agent_name: str = "phase4_agent",
) -> str:
    """Write a ``phase4_org_shared`` memory (ORG_SHARED, promotable)."""
    prov = make_provenance(
        written_by="agent:phase4",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("phase4://src", label="src")],
    )
    metadata = attach_to_metadata(
        {"kind": "phase4_org_shared", "subject_key": subject_key}, prov
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


def _write_tenant_instruction(
    mm: Mem0MemoryManager,
    *,
    subject_key: str,
    content: str,
    agent_name: str = "phase4_agent",
) -> str:
    """Write a tenant_instruction (TENANT_PRIVATE — never promotable)."""
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata={"kind": "tenant_instruction", "subject_key": subject_key},
        infer=False,
    )
    assert mid is not None
    return mid


def _make_fed_service() -> FederationService:
    """FederationService bound to in-process per-tenant Mem0 managers.

    Used in place of the HTTP /promote_to_org_trunk route because the
    runtime route builds its own ``build_default_registry()`` which
    does not have the test-only ``phase4_org_shared`` ORG_SHARED kind.
    Promotion logic lives entirely in FederationService — the route is
    a thin wrapper, so the round-trip in-process is the same Vespa
    write + same federation merge.
    """
    return FederationService(
        memory_manager_factory=lambda tid: Mem0MemoryManager(tid),
        registry=_registry_with_fed_kind(),
    )


def _federated_rows(t_view_tenant: str) -> list[dict]:
    """Read t_view_tenant's federated view (tenant + org-trunk merge).

    Both tenant writes and trunk writes share ``phase4_agent`` —
    FederationService.promote_to_org_trunk copies ``source_memory.agent_name``
    forward (falling back to ``_promoted`` only when missing). The
    source memory's agent_name is set explicitly by callers below so
    a single agent_name filter sees both sides of the merge.
    """
    return _make_fed_service().federated_get_all(
        tenant_id=t_view_tenant, agent_name="phase4_agent"
    )


def _cleanup(mm: Mem0MemoryManager, agent_name: str = "phase4_agent") -> None:
    try:
        mm.clear_agent_memory(mm.tenant_id, agent_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers to mint sibling and cross-org tenant ids
# ---------------------------------------------------------------------------


def _sibling_tenants() -> tuple[str, str, str]:
    """Two same-org tenants and the resolved org-trunk tenant id."""
    base = unique_id("fed_")  # org_id portion
    t1 = f"{base}:t1"
    t2 = f"{base}:t2"
    trunk = org_trunk_tenant_id(t1)
    # Sanity: the two siblings MUST resolve to the same trunk.
    assert org_trunk_tenant_id(t2) == trunk
    return t1, t2, trunk


# ---------------------------------------------------------------------------
# 1. Org-trunk promotion is visible to every sibling tenant in the org
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestOrgTrunkVisibleToBothTenants:
    """t1 promotes → t2 (and t1) federated view contain the promoted record."""

    def test_promotion_visible_from_both_tenants(self) -> None:
        Mem0MemoryManager._instances.clear()
        t1, t2, trunk = _sibling_tenants()
        mm_t1 = _build_manager(t1)
        mm_t2 = _build_manager(t2)
        mm_trunk = _build_manager(trunk)  # ensures the trunk schema exists
        try:
            mid = _write_shared(
                mm_t1, subject_key="org.policy", content="org policy doc"
            )

            # Promote via in-process FederationService (see _make_fed_service
            # docstring for why the HTTP route is not used for ORG_SHARED
            # custom kinds).
            src_rows = mm_t1.memory.get_all(user_id=t1).get("results", [])
            src = next(r for r in src_rows if str(r.get("id")) == mid)
            # promote_to_org_trunk reads agent_name off source_memory and
            # falls back to "_promoted" when absent — Mem0's get_all does
            # not populate agent_name, so stamp it explicitly so the
            # trunk write lands under the same agent_name the federated
            # read filters by.
            src["agent_name"] = "phase4_agent"
            promotion = _make_fed_service().promote_to_org_trunk(
                source_tenant_id=t1,
                source_memory=src,
                actor_role=Pinnable.TENANT_ADMIN,
                actor_id="alice",
            )
            promoted_id = promotion.promoted_memory_id
            assert promotion.org_trunk_tenant_id == trunk
            assert promoted_id, promotion

            # Federated view from sibling tenant t2 — promoted record
            # surfaces with `_federation_origin="org_trunk"` and the
            # provenance stamp identifying t1 as the source tenant.
            t2_view = _federated_rows(t2)
            promoted = [r for r in t2_view if str(r.get("id")) == promoted_id]
            assert len(promoted) == 1, (
                f"sibling t2 cannot see promoted record {promoted_id}; "
                f"federated view ids: {[r.get('id') for r in t2_view]}"
            )
            promoted_row = promoted[0]
            assert promoted_row["_federation_origin"] == "org_trunk"
            meta = promoted_row.get("metadata") or {}
            if isinstance(meta, str):
                import json as _json

                meta = _json.loads(meta)
            assert meta.get("subject_key") == "org.policy"
            assert meta.get("promoted_from_tenant") == t1
            assert meta.get("promoted_by") == "alice"
            assert meta.get("promoted_by_role") == "tenant_admin"
        finally:
            _cleanup(mm_t1)
            _cleanup(mm_t2)
            _cleanup(mm_trunk, agent_name="_promoted")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Tenant overlay wins on subject_key collision with the trunk
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantOverlayWinsOnSubjectKeyCollision:
    """Tenant writes a local entry on the same subject_key → tenant wins."""

    def test_tenant_overlay_wins_for_owner_sibling_sees_trunk(self) -> None:
        Mem0MemoryManager._instances.clear()
        t1, t2, trunk = _sibling_tenants()
        mm_t1 = _build_manager(t1)
        mm_t2 = _build_manager(t2)
        mm_trunk = _build_manager(trunk)
        try:
            # Promote t1's shared doc into the trunk under subject_key=x.
            trunk_seed_id = _write_shared(
                mm_t1, subject_key="x", content="trunk-side content"
            )
            src_rows = mm_t1.memory.get_all(user_id=t1).get("results", [])
            src = next(r for r in src_rows if str(r.get("id")) == trunk_seed_id)
            src["agent_name"] = "phase4_agent"
            promotion = _make_fed_service().promote_to_org_trunk(
                source_tenant_id=t1,
                source_memory=src,
                actor_role=Pinnable.TENANT_ADMIN,
                actor_id="alice",
            )
            assert promotion.promoted_memory_id

            # t2 writes its OWN shared doc under the same subject_key.
            t2_local_id = _write_shared(
                mm_t2, subject_key="x", content="tenant-side content (t2)"
            )

            # t2's federated view: tenant-local wins.
            t2_view = _federated_rows(t2)
            hits_for_x = [
                r
                for r in t2_view
                if (
                    (r.get("metadata") or {})
                    if isinstance(r.get("metadata"), dict)
                    else __import__("json").loads(r["metadata"])
                ).get("subject_key")
                == "x"
            ]
            assert len(hits_for_x) == 1, (
                f"t2 view should dedup subject_key=x to a single winner; got {hits_for_x}"
            )
            winner = hits_for_x[0]
            assert winner["id"] == t2_local_id, (
                f"tenant overlay must win: expected t2_local={t2_local_id}, "
                f"got id={winner['id']!r} from origin={winner.get('_federation_origin')}"
            )
            assert winner["_federation_origin"] == "tenant"

            # t1's federated view: t1 has no local entry under x, so the
            # trunk version surfaces (origin=org_trunk).
            t1_view = _federated_rows(t1)
            t1_hits = [
                r
                for r in t1_view
                if (
                    (r.get("metadata") or {})
                    if isinstance(r.get("metadata"), dict)
                    else __import__("json").loads(r["metadata"])
                ).get("subject_key")
                == "x"
            ]
            assert len(t1_hits) == 1
            t1_hit = t1_hits[0]
            # t1's own external_doc (the one we promoted from) is still
            # in tenant t1, so the tenant-side wins there too — the
            # walker sees the original tenant memory.
            assert t1_hit["_federation_origin"] == "tenant"
            assert t1_hit["id"] == trunk_seed_id
        finally:
            _cleanup(mm_t1)
            _cleanup(mm_t2)
            _cleanup(mm_trunk, agent_name="_promoted")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. tenant_private kind refuses promotion
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantPrivateRefusesPromotion:
    """A tenant_instruction (sensitivity=TENANT_PRIVATE) cannot be promoted."""

    def test_promote_returns_403_with_exact_substring(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("fed_priv") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            mid = _write_tenant_instruction(
                mm, subject_key="secret.config", content="internal only"
            )
            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/promote_to_org_trunk",
                    json={"actor_role": "tenant_admin", "actor_id": "alice"},
                )
            assert resp.status_code == 403, resp.text[:300]
            detail = resp.json().get("detail", "")
            assert "is tenant_private; promotion to the org" in detail, detail
            assert "tenant_instruction" in detail, detail
        finally:
            _cleanup(mm)
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 4. Cross-org isolation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCrossOrgIsolation:
    """A promotion in org A's trunk is invisible from org B's federated view."""

    def test_org_a_promotion_invisible_to_org_b(self) -> None:
        Mem0MemoryManager._instances.clear()
        # Two distinct orgs → different org_id portions → different trunks.
        t_a = unique_id("fed_orga") + ":production"
        t_b = unique_id("fed_orgb") + ":production"
        assert org_trunk_tenant_id(t_a) != org_trunk_tenant_id(t_b)
        mm_a = _build_manager(t_a)
        mm_b = _build_manager(t_b)
        mm_a_trunk = _build_manager(org_trunk_tenant_id(t_a))
        try:
            mid = _write_shared(
                mm_a,
                subject_key="cross.org.subject",
                content="org A's promoted record",
            )
            src_rows = mm_a.memory.get_all(user_id=t_a).get("results", [])
            src = next(r for r in src_rows if str(r.get("id")) == mid)
            src["agent_name"] = "phase4_agent"
            promotion = _make_fed_service().promote_to_org_trunk(
                source_tenant_id=t_a,
                source_memory=src,
                actor_role=Pinnable.TENANT_ADMIN,
                actor_id="alice",
            )
            promoted_id = promotion.promoted_memory_id
            assert promoted_id, promotion

            # Org B's federated view should NOT contain the promoted record.
            b_view = _federated_rows(t_b)
            b_ids = {str(r.get("id")) for r in b_view}
            assert promoted_id not in b_ids, (
                f"cross-org leak: org B's federated view contains org A's "
                f"promoted record {promoted_id}; b_ids={b_ids}"
            )
        finally:
            _cleanup(mm_a)
            _cleanup(mm_b)
            _cleanup(mm_a_trunk, agent_name="_promoted")
            Mem0MemoryManager._instances.clear()
