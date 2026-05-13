"""Phase 2 — Provenance round-trip + per-tenant store end-to-end.

Pins the contracts the citation/audit/synthesis agents rely on:
  * a provenance write lands in the per-tenant Vespa schema and is
    walkable in BFS order via ProvenanceWalker;
  * the HTTP route (POST /admin/tenants/{t}/knowledge/citations/trace)
    returns the same shape with exact node ordering, derivation kinds,
    and primary-source list;
  * provenance_required schemas reject writes whose metadata lacks the
    provenance block;
  * external-only citations (URLs / docs) survive a walk as terminal
    primary sources;
  * per-tenant Vespa schema isolation: a memory written in tenant A is
    invisible to a Mem0MemoryManager bound to tenant B.

All assertions pin exact orderings, exact ref tuples, and exact dict
shapes — no ``is not None`` / ``len(x) >= 1`` weak shapes.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    ProvenanceWalker,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    SchemaViolationError,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

# k3d-cogniverse-serverlb forwards these. Same constants as Phase 1; if
# those tests pass, the host port mapping is correct.
VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    """In-process Mem0MemoryManager bound to the deployed cluster's Vespa.

    auto_create_schema=True deploys the per-tenant agent_memories AND
    provenance schemas the very first time a manager for this tenant
    init's. Tests downstream therefore can write and walk without any
    additional admin call.
    """
    Mem0MemoryManager._instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=VESPA_HTTP_PORT,
    )
    cm = ConfigManager(store=config_store)
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
    _warmup_provenance_schema(mm)
    return mm


def _warmup_provenance_schema(mm: Mem0MemoryManager, timeout_s: float = 30.0) -> None:
    """Block until the per-tenant provenance Vespa schema accepts writes.

    ``mm.initialize(auto_create_schema=True)`` triggers the deploy_schema
    HTTP call but Vespa's app-package re-deploy is asynchronous: the
    deploy returns immediately, then content nodes pick up the new
    schema definition asynchronously. The first attach() to a brand-new
    tenant's provenance schema can hit "No handler for document type"
    if the content nodes haven't picked up the deploy yet.

    Probing with a real attach + read cycle is the only deterministic
    signal the schema is live for both the write and query paths.
    """
    import time
    import uuid

    from cogniverse_core.memory.provenance import (
        CitationRef as _CitationRef,
    )
    from cogniverse_core.memory.provenance import (
        DerivationKind as _DerivationKind,
    )
    from cogniverse_core.memory.provenance import (
        make_provenance as _make_provenance,
    )

    probe_id = f"warmup-{uuid.uuid4().hex[:8]}"
    probe_prov = _make_provenance(
        written_by="warmup",
        derivation_kind=_DerivationKind.DIRECT_INGEST,
        confidence=0.5,
        derived_from=[_CitationRef.external("warmup://probe", label="probe")],
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            mm.provenance_store.attach(probe_id, probe_prov)
            # Read-back: schema is live for queries too.
            time.sleep(1.0)
            rec = mm.provenance_store.get(probe_id)
            if rec is not None:
                return
        except Exception:
            # Schema not yet picked up by content nodes — retry.
            pass
        time.sleep(1.0)
    pytest.fail(
        f"provenance schema for tenant {mm.tenant_id!r} never came online "
        f"within {timeout_s}s of Mem0MemoryManager.initialize"
    )


def _write_with_provenance(
    mm: Mem0MemoryManager,
    *,
    kind: str,
    content: str,
    derivation_kind: DerivationKind,
    derived_from: list[CitationRef],
    written_by: str = "agent:phase2",
    confidence: float = 0.85,
    agent_name: str = "phase2_agent",
) -> str:
    """Write a memory whose metadata carries a Provenance record.

    The Mem0MemoryManager auto-attaches the provenance to the per-tenant
    provenance Vespa schema in addition to the agent_memories schema.
    """
    prov = make_provenance(
        written_by=written_by,
        derivation_kind=derivation_kind,
        confidence=confidence,
        derived_from=derived_from,
    )
    base: dict = {"kind": kind, "subject_key": f"{kind}_subj"}
    metadata = attach_to_metadata(base, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None, (
        f"add_memory returned None for kind={kind!r}; provenance_required "
        f"schemas should not get None on a valid write"
    )
    return mid


def _wait_for_provenance(
    mm: Mem0MemoryManager, memory_id: str, timeout_s: float = 30.0
):
    """Poll the per-tenant provenance store until the indexed record lands.

    The provenance store writes via the same Vespa cluster but indexing
    is asynchronous. The k3d single-node cluster shows multi-second
    indexing latency (vs. sub-second on a warm prod cluster), so the
    poll deadline is generous to give it room without making the test
    skip on a real index outage. Returns the ProvenanceRecord or fails.
    """
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rec = mm.provenance_store.get(memory_id)
        if rec is not None:
            return rec
        time.sleep(1.0)
    pytest.fail(
        f"provenance for memory {memory_id!r} never indexed within {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# 1. ProvenanceWalker round-trip via real Vespa
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProvenanceRoundTripThroughVespa:
    """Write ROOT then CHILD-with-derived_from-ROOT; walk must rebuild the chain."""

    def test_walk_returns_root_then_child_in_bfs_order(self) -> None:
        tenant_id = unique_id("prov_rt")
        mm = _build_manager(tenant_id)
        try:
            root_id = _write_with_provenance(
                mm,
                kind="external_doc",
                content="root document content",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[CitationRef.external("phase2://root", label="root_src")],
            )
            _wait_for_provenance(mm, root_id)

            child_id = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="derived fact based on root",
                derivation_kind=DerivationKind.SYNTHESIS,
                derived_from=[CitationRef.memory(root_id, label="parent_mem")],
            )
            _wait_for_provenance(mm, child_id)

            # Direct store fetch — rebuild the Provenance and assert
            # the parent ref lines up exactly.
            child_rec = mm.provenance_store.get(child_id)
            assert child_rec is not None, "child provenance not indexed"
            child_prov = child_rec.to_provenance()
            assert child_prov.derivation_kind is DerivationKind.SYNTHESIS
            assert len(child_prov.derived_from) == 1
            assert child_prov.derived_from[0].ref_kind == "memory"
            assert child_prov.derived_from[0].ref_id == root_id

            # Walker traversal — BFS order, depths exact, primary source pinned.
            walker = ProvenanceWalker(mm, max_depth=5, max_nodes=10)
            graph = walker.walk(child_id, tenant_id=tenant_id)
            assert graph.root_memory_id == child_id
            assert [n.memory_id for n in graph.nodes] == [child_id, root_id]
            assert [n.depth for n in graph.nodes] == [0, 1]
            assert graph.truncated_at_max_depth is False
            # primary_sources is deduped leaf-refs. Walker contract
            # (provenance_store.walk:283-289): for ROOT (which has only
            # external derived_from), it surfaces the external URL AND
            # the ROOT memory ref itself (no memory children → leaf).
            # CHILD is NOT added to primary_sources (it has memory
            # children — ROOT — so it descends rather than terminating).
            primary = [(r.ref_kind, r.ref_id) for r in graph.primary_sources]
            assert primary == [
                ("url", "phase2://root"),
                ("memory", root_id),
            ], primary
        finally:
            mm.clear_agent_memory(tenant_id, "phase2_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. HTTP citation-trace route returns the same shape
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCitationTraceHTTPRoute:
    """The /admin/tenants/{t}/knowledge/citations/trace JSON shape is pinned."""

    def test_trace_returns_node_list_in_order(self) -> None:
        tenant_id = unique_id("prov_http")
        mm = _build_manager(tenant_id)
        try:
            root_id = _write_with_provenance(
                mm,
                kind="external_doc",
                content="root for HTTP trace",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[CitationRef.external("phase2://http_root")],
            )
            _wait_for_provenance(mm, root_id)
            child_id = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="child for HTTP trace",
                derivation_kind=DerivationKind.SYNTHESIS,
                derived_from=[CitationRef.memory(root_id)],
            )
            _wait_for_provenance(mm, child_id)

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/citations/trace",
                    json={"memory_id": child_id, "max_depth": 5, "max_nodes": 10},
                )
            assert resp.status_code == 200, (
                f"trace failed rc={resp.status_code}: {resp.text[:300]}"
            )
            body = resp.json()

            assert body["root_memory_id"] == child_id
            assert [n["memory_id"] for n in body["nodes"]] == [child_id, root_id]
            assert [n["depth"] for n in body["nodes"]] == [0, 1]
            assert [n["derivation_kind"] for n in body["nodes"]] == [
                "synthesis",
                "direct_ingest",
            ]
            assert body["truncated"] is False
            # Telemetry block is filled in by the agent. primary_source_count
            # is 2: external URL + ROOT memory leaf-self-ref (see Phase 2's
            # TestExternalCitationLeafSurvivesWalk for the walker contract).
            assert body["metadata"]["nodes_visited"] == 2, body["metadata"]
            assert body["metadata"]["primary_source_count"] == 2, body["metadata"]
            primary = [(r["ref_kind"], r["ref_id"]) for r in body["primary_sources"]]
            assert primary == [
                ("url", "phase2://http_root"),
                ("memory", root_id),
            ], primary
        finally:
            mm.clear_agent_memory(tenant_id, "phase2_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. provenance_required enforcement on write
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProvenanceRequiredEnforcement:
    """external_doc requires provenance — write without it must raise."""

    def test_write_without_provenance_rejected(self) -> None:
        tenant_id = unique_id("prov_enf")
        mm = _build_manager(tenant_id)
        try:
            with pytest.raises(SchemaViolationError) as exc:
                mm.add_memory(
                    content="should be rejected",
                    tenant_id=tenant_id,
                    agent_name="phase2_agent",
                    metadata={"kind": "external_doc"},  # no provenance block
                    infer=False,
                )
            # The schema's exact substring; a future rewrite of the
            # message will trip this loudly.
            assert "requires provenance" in str(exc.value)
            assert "external_doc" in str(exc.value)
        finally:
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 4. External citation leaf survives a walk as a terminal primary source
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestExternalCitationLeafSurvivesWalk:
    """A memory whose only derived_from is an external URL surfaces as primary."""

    def test_external_url_leaf_in_primary_sources(self) -> None:
        tenant_id = unique_id("prov_ext")
        mm = _build_manager(tenant_id)
        try:
            mid = _write_with_provenance(
                mm,
                kind="external_doc",
                content="external-only paper",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[
                    CitationRef.external("https://example.com/paper.pdf", label="paper")
                ],
            )
            _wait_for_provenance(mm, mid)

            walker = ProvenanceWalker(mm, max_depth=5, max_nodes=10)
            graph = walker.walk(mid, tenant_id=tenant_id)

            assert [n.memory_id for n in graph.nodes] == [mid]
            assert [n.depth for n in graph.nodes] == [0]
            assert graph.truncated_at_max_depth is False
            # Walker contract (provenance_store.walk lines 287-289):
            # a memory whose ``derived_from_memory_ids`` is empty is
            # itself a primary source — so this leaf yields BOTH its
            # external URL ref AND a self-ref (memory ref to itself).
            # Order is "external first, then self" because the walker
            # surfaces non-memory refs at line 285 before the leaf
            # check at line 289.
            assert graph.primary_sources == [
                CitationRef(
                    ref_kind="url",
                    ref_id="https://example.com/paper.pdf",
                    label="paper",
                ),
                CitationRef(ref_kind="memory", ref_id=mid, label=None),
            ]
        finally:
            mm.clear_agent_memory(tenant_id, "phase2_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 5. Per-tenant provenance Vespa schema isolation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPerTenantProvenanceSchemaIsolation:
    """Provenance written in tenant A must not be readable from tenant B's manager."""

    def test_tenant_a_provenance_not_visible_to_tenant_b(self) -> None:
        tid_a = unique_id("prov_isoa")
        mm_a = _build_manager(tid_a)
        try:
            mid_a = _write_with_provenance(
                mm_a,
                kind="entity_fact",
                content="tenant A only",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[CitationRef.external("phase2://iso_a")],
            )
            _wait_for_provenance(mm_a, mid_a)

            # Now construct a separate tenant B manager and try the
            # same lookup. Tenant B's provenance store hits a DIFFERENT
            # Vespa schema (provenance_<tid_b>), which has no record
            # for mid_a. Cross-tenant leak proof.
            tid_b = unique_id("prov_isob")
            Mem0MemoryManager._instances.clear()
            mm_b = _build_manager(tid_b)
            try:
                rec = mm_b.provenance_store.get(mid_a)
                assert rec is None, (
                    f"tenant B leaked tenant A's provenance: {rec}; "
                    f"per-tenant provenance schema isolation is broken"
                )
            finally:
                mm_b.clear_agent_memory(tid_b, "phase2_agent")
        finally:
            try:
                mm_a.clear_agent_memory(tid_a, "phase2_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
