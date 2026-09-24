"""Provenance round-trips through real Mem0 + Vespa.

Verifies:
  * a Provenance attached via ``attach_to_metadata`` survives a Mem0 add /
    search round-trip with all fields intact;
  * the citation chain walker recovers the source graph from the live
    backend (not a mock store);
  * a missing ``derived_from`` is rejected by the schema validator before
    any backend write attempt happens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.multi_document_synthesis_agent import (
    MultiDocSynthesisDeps,
    MultiDocumentSynthesisAgent,
)
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    ProvenanceConsistencyError,
    ProvenanceRepairConflictError,
    ProvenanceWalker,
    attach_to_metadata,
    extract_from_memory,
    make_provenance,
)
from cogniverse_core.memory.provenance_store import ProvenanceWriteError
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    SchemaViolationError,
)
from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.routers import knowledge as knowledge_router
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "a2_provenance_tenant"
AGENT = "a2_provenance_agent"


class _StaticEmbedder:
    def embed(self, _text, _memory_action):
        return [0.01] * 768


class _UpdateDecisionLLM:
    def __init__(self, old_content: str, new_content: str):
        self._old_content = old_content
        self._new_content = new_content
        self._calls = 0

    def generate_response(self, **_kwargs):
        self._calls += 1
        if self._calls == 1:
            return json.dumps({"facts": [self._new_content]})
        if self._calls == 2:
            return json.dumps(
                {
                    "memory": [
                        {
                            "id": "0",
                            "text": self._new_content,
                            "event": "UPDATE",
                            "old_memory": self._old_content,
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected Mem0 LLM call {self._calls}")


@pytest.fixture(scope="module")
def memory_env(shared_memory_vespa, shared_denseon):
    Mem0MemoryManager._instances.clear()

    def upstream(path):
        port = (
            shared_memory_vespa["config_port"]
            if path.startswith(("/application/", "/config/"))
            else shared_memory_vespa["http_port"]
        )
        return f"http://127.0.0.1:{port}"

    with HTTPFaultProxy(upstream) as proxy:
        config_store = VespaConfigStore(
            backend_url="http://127.0.0.1",
            backend_port=proxy.port,
        )
        cm = ConfigManager(store=config_store)
        cm.set_system_config(
            SystemConfig(
                backend_url="http://127.0.0.1",
                backend_port=proxy.port,
                inference_service_urls={"denseon": shared_denseon},
            )
        )
        mm = Mem0MemoryManager(tenant_id=TENANT)
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        mm.initialize(
            backend_host="http://127.0.0.1",
            backend_port=proxy.port,
            backend_config_port=proxy.port,
            base_schema_name="agent_memories",
            llm_model=get_llm_model(),
            embedding_model="lightonai/DenseOn",
            llm_base_url=get_llm_base_url(),
            embedder_base_url=shared_denseon,
            auto_create_schema=True,
            config_manager=cm,
            schema_loader=schema_loader,
        )

        yield SimpleNamespace(
            manager=mm,
            proxy=proxy,
            config_manager=cm,
            schema_loader=schema_loader,
            denseon=shared_denseon,
        )

        try:
            mm.clear_agent_memory(TENANT, AGENT)
            mm.clear_agent_memory(TENANT, "multi_document_synthesis_agent")
        except Exception:
            pass
    Mem0MemoryManager._instances.clear()


def _add_with_provenance(mm, content: str, prov, kind: str = "entity_fact") -> str:
    """Helper that mirrors the canonical write path: validate then persist."""
    schema = KnowledgeSchema(kind=kind, provenance_required=True)
    schema.validate_provenance(prov)
    return mm.add_memory(
        content=content,
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": kind}, prov),
        infer=False,
    )


def test_provenance_round_trips_through_real_vespa(memory_env):
    mm = memory_env.manager
    prov = make_provenance(
        written_by="agent:integration",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.82,
        derived_from=[
            CitationRef.external("https://wiki/integration-source"),
            CitationRef.memory("m_seed_42", label="prior_synthesis"),
        ],
        trace_id="trace-integration-001",
    )

    memory_id = _add_with_provenance(
        mm, "Integration test: round-trippable provenance content.", prov
    )
    assert memory_id, "Mem0 must return an id when infer=False"

    found = mm.search_memory(
        query="round-trippable provenance content",
        tenant_id=TENANT,
        agent_name=AGENT,
        top_k=5,
    )
    matched = [m for m in found if "round-trippable provenance" in m.get("memory", "")]
    assert matched, f"seeded memory not retrievable; got {found}"

    rebuilt = extract_from_memory(matched[0])
    assert rebuilt is not None, "provenance must round-trip through Vespa"
    assert rebuilt.derivation_kind is DerivationKind.SYNTHESIS
    assert rebuilt.confidence == pytest.approx(0.82)
    assert rebuilt.trace_id == "trace-integration-001"
    refs = {(r.ref_kind, r.ref_id) for r in rebuilt.derived_from}
    assert ("url", "https://wiki/integration-source") in refs
    assert ("memory", "m_seed_42") in refs


def test_missing_provenance_rejected_before_vespa_write(memory_env):
    """A required-provenance kind must reject empty derived_from BEFORE write."""
    mm = memory_env.manager
    bad_prov = make_provenance(
        written_by="agent:bad",
        derivation_kind=DerivationKind.AGENT_INFERENCE,
        confidence=0.5,
        derived_from=[],  # empty -> schema rejects
    )
    schema = KnowledgeSchema(kind="entity_fact", provenance_required=True)
    with pytest.raises(SchemaViolationError):
        schema.validate_provenance(bad_prov)

    # Confirm the rejected write did not leak into Vespa.
    found = mm.search_memory(
        query="bad provenance",
        tenant_id=TENANT,
        agent_name=AGENT,
        top_k=10,
    )
    assert all("bad provenance" not in m.get("memory", "") for m in found), (
        "rejected write must not have persisted"
    )


def test_walker_recovers_chain_from_real_vespa(memory_env):
    """Build a 3-node chain in real Vespa, walk it back to primary sources."""
    mm = memory_env.manager

    # Leaf primary source — a directly-ingested fact citing only an external URL.
    leaf_prov = make_provenance(
        written_by="agent:ingest",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.95,
        derived_from=[CitationRef.external("https://wiki/leaf-source")],
    )
    leaf_id = _add_with_provenance(
        mm,
        "Walker test leaf: this is the original ingested fact.",
        leaf_prov,
        kind="external_doc",
    )

    # Mid node — a summarisation of the leaf.
    mid_prov = make_provenance(
        written_by="agent:summarizer",
        derivation_kind=DerivationKind.SUMMARIZATION,
        confidence=0.85,
        derived_from=[CitationRef.memory(leaf_id)],
    )
    mid_id = _add_with_provenance(
        mm,
        "Walker test mid: summary of the leaf fact.",
        mid_prov,
        kind="external_doc",
    )

    # Root — synthesis of the mid + a fresh external citation.
    root_prov = make_provenance(
        written_by="agent:search",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.78,
        derived_from=[
            CitationRef.memory(mid_id),
            CitationRef.external("https://wiki/root-extra"),
        ],
    )
    root_id = _add_with_provenance(
        mm,
        "Walker test root: synthesised answer.",
        root_prov,
        kind="entity_fact",
    )

    walker = ProvenanceWalker(mm)
    graph = walker.walk(root_id, tenant_id=TENANT)

    chain_ids = {n.memory_id for n in graph.nodes}
    assert root_id in chain_ids
    assert mid_id in chain_ids
    assert leaf_id in chain_ids

    primary_keys = {(r.ref_kind, r.ref_id) for r in graph.primary_sources}
    # Both external URLs surface as primary sources in the walked graph.
    assert ("url", "https://wiki/root-extra") in primary_keys
    assert ("url", "https://wiki/leaf-source") in primary_keys
    assert graph.truncated_at_max_depth is False


def test_walker_reuses_bfs_records_without_per_node_get(memory_env, monkeypatch):
    """The walker populates each node's provenance from the records already
    fetched during the BFS — it issues no per-node ``store.get``."""
    mm = memory_env.manager

    leaf_prov = make_provenance(
        written_by="agent:ingest",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.95,
        derived_from=[CitationRef.external("https://wiki/reuse-leaf")],
    )
    leaf_id = _add_with_provenance(
        mm, "Reuse test leaf fact.", leaf_prov, kind="external_doc"
    )
    mid_prov = make_provenance(
        written_by="agent:summarizer",
        derivation_kind=DerivationKind.SUMMARIZATION,
        confidence=0.85,
        derived_from=[CitationRef.memory(leaf_id)],
    )
    mid_id = _add_with_provenance(
        mm, "Reuse test mid summary.", mid_prov, kind="external_doc"
    )
    root_prov = make_provenance(
        written_by="agent:search",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.78,
        derived_from=[CitationRef.memory(mid_id)],
    )
    root_id = _add_with_provenance(
        mm, "Reuse test root synthesis.", root_prov, kind="entity_fact"
    )

    walker = ProvenanceWalker(mm)

    get_calls: list[str] = []
    real_get = walker._store.get

    def spy_get(memory_id):
        get_calls.append(memory_id)
        return real_get(memory_id)

    monkeypatch.setattr(walker._store, "get", spy_get)

    graph = walker.walk(root_id, tenant_id=TENANT)

    assert {n.memory_id for n in graph.nodes} == {root_id, mid_id, leaf_id}
    by_id = {n.memory_id: n for n in graph.nodes}
    assert by_id[root_id].provenance is not None
    assert by_id[root_id].provenance.derivation_kind == DerivationKind.SYNTHESIS
    assert by_id[leaf_id].provenance is not None
    assert by_id[leaf_id].provenance.derivation_kind == DerivationKind.DIRECT_INGEST
    assert get_calls == []


@pytest.mark.asyncio
async def test_synthesis_provenance_refusal_removes_new_primary(memory_env):
    """A refused provenance feed cannot return a successful synthesis id."""
    mm = memory_env.manager
    agent = MultiDocumentSynthesisAgent(
        deps=MultiDocSynthesisDeps(tenant_id=TENANT),
        config_manager=memory_env.config_manager,
    )
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = "multi_document_synthesis_agent"

    memory_env.proxy.arm(
        lambda method, path, _body: (
            method in {"POST", "PUT"} and "/document/v1/content/provenance_" in path
        ),
        failure=True,
    )
    memory_env.proxy.release.set()

    with pytest.raises(ProvenanceWriteError, match="provenance") as exc_info:
        persisted_id = await agent._persist_synthesis(
            tenant_id=TENANT,
            answer="A synthesis whose provenance index write is refused.",
            citation_refs=[CitationRef.external("https://source.test/refused")],
        )
        pytest.fail(
            f"persistence returned {persisted_id}; proxy paths="
            f"{[path for _, path, _ in memory_env.proxy.requests]}"
        )

    failed_memory_id = exc_info.value.memory_id
    assert mm.memory.get(failed_memory_id) is None
    assert mm.provenance_store.fetch([failed_memory_id]) == {}


def test_add_compensation_removes_a_provenance_row_that_landed(memory_env):
    """A feed that landed but answered as failed leaves no orphan row."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:landed-row",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.71,
        derived_from=[CitationRef.external("https://source.test/landed-row")],
    )
    memory_env.proxy.arm(
        lambda method, path, _body: (
            method in {"POST", "PUT"} and "/document/v1/content/provenance_" in path
        ),
        failure=True,
        after_upstream=True,
    )
    memory_env.proxy.release.set()
    first_request = len(memory_env.proxy.requests)

    with pytest.raises(ProvenanceWriteError) as exc_info:
        mm.add_memory(
            content="A primary whose provenance row landed under a refused answer.",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
            infer=False,
        )

    error = exc_info.value
    memory_id = error.memory_id
    row_id = f"prov-{mm._storage_tenant_id}-{memory_id}"
    assert error.compensation_error is None
    assert mm.memory.get(memory_id) is None
    assert mm.provenance_store.fetch([memory_id]) == {}
    assert [
        method
        for method, path, _ in memory_env.proxy.requests[first_request:]
        if "/provenance_" in path
        and unquote(path.split("?", 1)[0]).endswith(f"/docid/{row_id}")
    ] == ["POST", "DELETE"]


def test_provenance_row_delete_refused_by_vespa_raises_and_keeps_the_row(
    memory_env,
):
    """A refused indexed-row delete is an error, never a reported deletion."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:refused-delete",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.66,
        derived_from=[CitationRef.external("https://source.test/refused-delete")],
    )
    memory_id = _add_with_provenance(
        mm, "A primary whose indexed row delete is refused.", provenance
    )
    row_id = f"prov-{mm._storage_tenant_id}-{memory_id}"
    indexed = mm.provenance_store.fetch([memory_id])
    assert list(indexed) == [memory_id]

    memory_env.proxy.arm(
        lambda method, path, _body: (
            method == "DELETE" and "/provenance_" in path and row_id in unquote(path)
        ),
        failure=True,
    )
    memory_env.proxy.release.set()

    with pytest.raises(ProvenanceWriteError) as exc_info:
        mm.provenance_store.delete(memory_id)

    error = exc_info.value
    assert error.memory_id == memory_id
    assert error.row_id == row_id
    assert error.result is None
    assert type(error.cause) is RuntimeError
    assert "HTTP 400" in str(error.cause)
    assert mm.provenance_store.fetch([memory_id]) == indexed

    assert mm.provenance_store.delete(memory_id) is True
    assert mm.provenance_store.fetch([memory_id]) == {}
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_walker_rejects_primary_with_missing_indexed_provenance(memory_env):
    """A process failure before attach cannot turn declared sources into a leaf."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:interrupted",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.76,
        derived_from=[CitationRef.external("https://source.test/interrupted")],
    )
    result = mm.memory.add(
        "A primary persisted before its provenance attach was interrupted.",
        user_id=mm._storage_tenant_id,
        agent_id=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    assert result["results"][0]["event"] == "ADD"
    memory_id = result["results"][0]["id"]
    assert mm.provenance_store.fetch([memory_id]) == {}

    Mem0MemoryManager._instances.pop(TENANT, None)
    restarted = Mem0MemoryManager(tenant_id=TENANT)
    restarted.initialize(
        backend_host="http://127.0.0.1",
        backend_port=memory_env.proxy.port,
        backend_config_port=memory_env.proxy.port,
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=memory_env.denseon,
        auto_create_schema=False,
        config_manager=memory_env.config_manager,
        schema_loader=memory_env.schema_loader,
    )

    with pytest.raises(
        ProvenanceConsistencyError, match="indexed provenance is missing"
    ):
        ProvenanceWalker(restarted).walk(memory_id, tenant_id=TENANT)

    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_walker_rejects_indexed_orphan_but_keeps_both_absent_leaf(memory_env):
    """Only a genuinely absent primary and index row is a normal leaf."""
    mm = memory_env.manager
    orphan_id = "indexed-orphan"
    provenance = make_provenance(
        written_by="agent:orphan",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.73,
        derived_from=[CitationRef.external("https://source.test/orphan")],
    )
    assert mm.provenance_store.attach(orphan_id, provenance) == (
        f"prov-{mm._storage_tenant_id}-{orphan_id}"
    )
    assert mm.memory.get(orphan_id) is None

    with pytest.raises(ProvenanceConsistencyError, match="primary memory is missing"):
        ProvenanceWalker(mm).walk(orphan_id, tenant_id=TENANT)

    missing_id = "primary-and-index-both-absent"
    graph = ProvenanceWalker(mm).walk(missing_id, tenant_id=TENANT)
    assert [node.memory_id for node in graph.nodes] == [missing_id]
    assert [(ref.ref_kind, ref.ref_id) for ref in graph.primary_sources] == [
        ("memory", missing_id)
    ]


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "written_by": "agent:malformed",
            "written_at": "2026-09-19T00:00:00+00:00",
            "derivation_kind": "synthesis",
            "confidence": 0.5,
            "derived_from": [None],
            "trace_id": None,
        },
        {
            "written_by": "agent:malformed",
            "written_at": "2026-09-19T00:00:00+00:00",
            "derivation_kind": "synthesis",
            "confidence": None,
            "derived_from": [
                {"ref_kind": "url", "ref_id": "https://source.test/malformed"}
            ],
            "trace_id": None,
        },
    ],
)
def test_malformed_provenance_uses_typed_write_and_read_errors(
    memory_env, invalid_payload
):
    """Nested ref and confidence type failures never leak raw TypeError."""
    mm = memory_env.manager
    with pytest.raises(ProvenanceWriteError, match="invalid_provenance"):
        mm.add_memory(
            content="Malformed provenance rejected before its primary write.",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "entity_fact", "provenance": invalid_payload},
            infer=False,
        )

    raw = mm.memory.add(
        "Malformed provenance persisted by an interrupted external writer.",
        user_id=mm._storage_tenant_id,
        agent_id=AGENT,
        metadata={"kind": "entity_fact", "provenance": invalid_payload},
        infer=False,
    )
    memory_id = raw["results"][0]["id"]
    with pytest.raises(
        ProvenanceConsistencyError, match="primary provenance payload is malformed"
    ):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)
    with pytest.raises(
        ProvenanceConsistencyError, match="primary provenance payload is malformed"
    ) as repair_error:
        mm.repair_provenance(memory_id)
    assert repair_error.value.memory_id == memory_id
    assert mm.provenance_store.fetch([memory_id]) == {}
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_real_mem0_update_attach_failure_preserves_existing_primary(memory_env):
    """An UPDATE attach fault keeps the existing ID and exposes mismatch."""
    mm = memory_env.manager
    agent_name = "provenance_update_agent"
    old_content = "The launch date is 20 September."
    new_content = "The launch date is 21 September."
    old_provenance = make_provenance(
        written_by="agent:update-old",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.8,
        derived_from=[CitationRef.external("https://source.test/update-old")],
    )
    memory_id = mm.add_memory(
        content=old_content,
        tenant_id=TENANT,
        agent_name=agent_name,
        metadata=attach_to_metadata({"kind": "entity_fact"}, old_provenance),
        infer=False,
    )
    old_indexed = mm.provenance_store.fetch([memory_id])[memory_id]
    new_provenance = make_provenance(
        written_by="agent:update-new",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.91,
        derived_from=[CitationRef.external("https://source.test/update-new")],
    )
    old_llm = mm.memory.llm
    mm.memory.llm = _UpdateDecisionLLM(old_content, new_content)
    memory_env.proxy.arm(
        lambda method, path, _body: (
            method in {"POST", "PUT"} and "/document/v1/content/provenance_" in path
        ),
        failure=True,
    )
    memory_env.proxy.release.set()
    try:
        with pytest.raises(ProvenanceWriteError) as exc_info:
            mm.add_memory(
                content=new_content,
                tenant_id=TENANT,
                agent_name=agent_name,
                metadata=attach_to_metadata({"kind": "entity_fact"}, new_provenance),
                infer=True,
            )
    finally:
        mm.memory.llm = old_llm

    assert exc_info.value.memory_id == memory_id
    assert exc_info.value.compensation_error is None
    primary = mm.memory.get(memory_id)
    assert primary["id"] == memory_id
    assert primary["memory"] == new_content
    assert extract_from_memory(primary) == new_provenance
    assert mm.provenance_store.fetch([memory_id]) == {memory_id: old_indexed}
    with pytest.raises(
        ProvenanceConsistencyError,
        match="indexed provenance does not match primary provenance",
    ):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)

    assert mm.repair_provenance(memory_id) == (
        f"prov-{mm._storage_tenant_id}-{memory_id}"
    )
    repaired = ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)
    assert [(ref.ref_kind, ref.ref_id) for ref in repaired.primary_sources] == [
        ("url", "https://source.test/update-new"),
        ("memory", memory_id),
    ]
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_repair_serializes_with_manager_primary_update(memory_env, monkeypatch):
    """A manager update cannot cross repair's verified success point."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:serialized-repair",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.83,
        derived_from=[CitationRef.external("https://source.test/serialized")],
    )
    memory_id = mm.add_memory(
        content="Primary content before serialized repair.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    entered = threading.Event()
    release = threading.Event()
    real_attach = mm.provenance_store.attach
    old_embedder = mm.memory.embedding_model
    mm.memory.embedding_model = _StaticEmbedder()

    def blocked_attach(target_memory_id, target_provenance, **kwargs):
        row_id = real_attach(target_memory_id, target_provenance, **kwargs)
        entered.set()
        assert release.wait(10) is True
        return row_id

    monkeypatch.setattr(mm.provenance_store, "attach", blocked_attach)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            repair_future = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(10) is True
            update_future = pool.submit(
                mm.update_memory,
                memory_id,
                "Primary content after serialized repair.",
                TENANT,
                AGENT,
            )
            threading.Event().wait(0.5)
            assert update_future.done() is False
            release.set()
            assert repair_future.result(timeout=20) == (
                f"prov-{mm._storage_tenant_id}-{memory_id}"
            )
            assert update_future.result(timeout=20) is True
    finally:
        release.set()
        mm.memory.embedding_model = old_embedder

    assert mm.memory.get(memory_id)["memory"] == (
        "Primary content after serialized repair."
    )
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_repair_serializes_with_second_manager_for_same_storage_tenant(
    memory_env, monkeypatch
):
    """Canonical tenant peers share repair/write ownership across instances."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:cross-instance",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.82,
        derived_from=[CitationRef.external("https://source.test/cross-instance")],
    )
    memory_id = mm.add_memory(
        content="Primary before a cross-instance repair.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    canonical_tenant = mm._storage_tenant_id
    Mem0MemoryManager._instances.pop(canonical_tenant, None)
    peer = Mem0MemoryManager(canonical_tenant)
    peer.initialize(
        backend_host="http://127.0.0.1",
        backend_port=memory_env.proxy.port,
        backend_config_port=memory_env.proxy.port,
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=memory_env.denseon,
        auto_create_schema=False,
        config_manager=memory_env.config_manager,
        schema_loader=memory_env.schema_loader,
    )
    old_embedder = peer.memory.embedding_model
    peer.memory.embedding_model = _StaticEmbedder()
    entered = threading.Event()
    release = threading.Event()
    real_attach = mm.provenance_store.attach

    def blocked_attach(target_memory_id, target_provenance, **kwargs):
        row_id = real_attach(target_memory_id, target_provenance, **kwargs)
        entered.set()
        assert release.wait(10) is True
        return row_id

    monkeypatch.setattr(mm.provenance_store, "attach", blocked_attach)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            repair_future = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(10) is True
            update_future = pool.submit(
                peer.update_memory,
                memory_id,
                "Primary after the cross-instance repair.",
                canonical_tenant,
                AGENT,
            )
            threading.Event().wait(0.5)
            assert update_future.done() is False
            release.set()
            assert repair_future.result(timeout=20) == (
                f"prov-{canonical_tenant}-{memory_id}"
            )
            assert update_future.result(timeout=20) is True
    finally:
        release.set()
        peer.memory.embedding_model = old_embedder

    assert peer.memory.get(memory_id)["memory"] == (
        "Primary after the cross-instance repair."
    )
    peer.memory.delete(memory_id)
    assert peer.memory.get(memory_id) is None
    Mem0MemoryManager._instances.pop(canonical_tenant, None)


def test_provenance_dropping_update_serializes_with_a_peer_repair(
    memory_env, monkeypatch
):
    """Explicit provenance-free metadata on a provenance-bearing primary still
    rewrites what the indexed row must agree with, so it waits for repair."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:dropping-update",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.79,
        derived_from=[CitationRef.external("https://source.test/dropping-update")],
    )
    memory_id = mm.add_memory(
        content="Primary before a provenance-dropping update.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    canonical_tenant = mm._storage_tenant_id
    Mem0MemoryManager._instances.pop(canonical_tenant, None)
    peer = Mem0MemoryManager(canonical_tenant)
    peer.initialize(
        backend_host="http://127.0.0.1",
        backend_port=memory_env.proxy.port,
        backend_config_port=memory_env.proxy.port,
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=memory_env.denseon,
        auto_create_schema=False,
        config_manager=memory_env.config_manager,
        schema_loader=memory_env.schema_loader,
    )
    old_embedder = peer.memory.embedding_model
    peer.memory.embedding_model = _StaticEmbedder()
    entered = threading.Event()
    release = threading.Event()
    real_attach = mm.provenance_store.attach

    def blocked_attach(target_memory_id, target_provenance, **kwargs):
        row_id = real_attach(target_memory_id, target_provenance, **kwargs)
        entered.set()
        assert release.wait(10) is True
        return row_id

    monkeypatch.setattr(mm.provenance_store, "attach", blocked_attach)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            repair_future = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(10) is True
            update_future = pool.submit(
                peer.update_memory,
                memory_id,
                "Primary after a provenance-dropping update.",
                canonical_tenant,
                AGENT,
                {"kind": "note"},
            )
            threading.Event().wait(0.5)
            assert update_future.done() is False
            release.set()
            assert repair_future.result(timeout=20) == (
                f"prov-{canonical_tenant}-{memory_id}"
            )
            assert update_future.result(timeout=20) is True
    finally:
        release.set()
        peer.memory.embedding_model = old_embedder

    updated = peer.memory.get(memory_id)
    assert updated["memory"] == "Primary after a provenance-dropping update."
    assert updated["metadata"]["kind"] == "note"
    assert "provenance" not in updated["metadata"]
    assert peer.provenance_store.delete(memory_id) is True
    peer.memory.delete(memory_id)
    assert peer.memory.get(memory_id) is None
    Mem0MemoryManager._instances.pop(canonical_tenant, None)


def test_raw_mutation_after_repair_final_read_is_detected_by_reader(
    memory_env, monkeypatch
):
    """Repair linearizes before a raw mutation whose digest later mismatches."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:final-window",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.86,
        derived_from=[CitationRef.external("https://source.test/final-window")],
    )
    memory_id = mm.add_memory(
        content="Primary before the repair final-read window.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    entered = threading.Event()
    release = threading.Event()
    real_get = mm.provenance_store.get
    old_embedder = mm.memory.embedding_model
    mm.memory.embedding_model = _StaticEmbedder()

    def blocked_index_verification(target_memory_id):
        record = real_get(target_memory_id)
        entered.set()
        assert release.wait(10) is True
        return record

    monkeypatch.setattr(mm.provenance_store, "get", blocked_index_verification)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            repair_future = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(10) is True
            mm.memory.update(
                memory_id,
                data="Primary mutated after repair's final primary read.",
                metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
            )
            release.set()
            assert repair_future.result(timeout=20) == (
                f"prov-{mm._storage_tenant_id}-{memory_id}"
            )
    finally:
        release.set()
        mm.memory.embedding_model = old_embedder

    assert mm.memory.get(memory_id)["memory"] == (
        "Primary mutated after repair's final primary read."
    )
    with pytest.raises(
        ProvenanceConsistencyError, match="primary digest does not match indexed"
    ):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)

    monkeypatch.setattr(mm.provenance_store, "get", real_get)
    assert mm.repair_provenance(memory_id) == (
        f"prov-{mm._storage_tenant_id}-{memory_id}"
    )
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def _peer_manager(memory_env, canonical_tenant):
    """A second manager instance for the same storage tenant.

    Distinct instances share no in-process lock, so anything they
    serialize against each other is serialized by the store-backed
    provenance lease alone.
    """
    Mem0MemoryManager._instances.pop(canonical_tenant, None)
    peer = Mem0MemoryManager(canonical_tenant)
    peer.initialize(
        backend_host="http://127.0.0.1",
        backend_port=memory_env.proxy.port,
        backend_config_port=memory_env.proxy.port,
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=memory_env.denseon,
        auto_create_schema=False,
        config_manager=memory_env.config_manager,
        schema_loader=memory_env.schema_loader,
    )
    peer.memory.embedding_model = _StaticEmbedder()
    return peer


def test_stalled_holder_is_fenced_after_a_peer_takes_the_expired_lease(
    memory_env, monkeypatch
):
    """A write that outlives its lease must not mutate or report success.

    The lease expires after a fixed hold time and a peer is entitled to
    take it over. Repair stalls inside its first primary read for longer
    than that hold time; the peer takes the lease and rewrites the
    primary. The stalled holder must be fenced before its indexed write
    rather than resuming against a primary it no longer owns.
    """
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:stale-holder",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.87,
        derived_from=[CitationRef.external("https://source.test/stale-holder")],
    )
    memory_id = mm.add_memory(
        content="Primary before the stale holder loses its lease.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    canonical_tenant = mm._storage_tenant_id
    peer = _peer_manager(memory_env, canonical_tenant)

    # The provenance write lease carries its own sizing rather than the
    # deploy lease's defaults. Scaled down here, preserving the contract the
    # real values keep: the wait outlasts the hold, so a stalled holder is
    # always waitable-out within one wait.
    from cogniverse_core.memory import manager as manager_module

    monkeypatch.setattr(manager_module, "PROVENANCE_LEASE_SECONDS", 5.0)
    monkeypatch.setattr(manager_module, "PROVENANCE_WAIT_SECONDS", 30.0)

    entered = threading.Event()
    release = threading.Event()
    real_primary_get = mm.memory.get
    real_attach = mm.provenance_store.attach
    primary_reads = 0
    stale_attach_calls = 0

    def stalled_primary_get(target_memory_id):
        nonlocal primary_reads
        primary_reads += 1
        if primary_reads == 1:
            entered.set()
            assert release.wait(30) is True
        return real_primary_get(target_memory_id)

    def counted_attach(target_memory_id, target_provenance, **kwargs):
        nonlocal stale_attach_calls
        stale_attach_calls += 1
        return real_attach(target_memory_id, target_provenance, **kwargs)

    monkeypatch.setattr(mm.memory, "get", stalled_primary_get)
    monkeypatch.setattr(mm.provenance_store, "attach", counted_attach)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            stale_repair = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(30) is True
            takeover = pool.submit(
                peer.update_memory,
                memory_id,
                "Primary written by the owner that took the lease over.",
                canonical_tenant,
                AGENT,
            )
            assert takeover.result(timeout=60) is True
            release.set()
            with pytest.raises(DeploymentLeaseLost):
                stale_repair.result(timeout=60)
    finally:
        release.set()

    assert stale_attach_calls == 0
    assert peer.memory.get(memory_id)["memory"] == (
        "Primary written by the owner that took the lease over."
    )
    assert peer.delete_memory(memory_id, canonical_tenant, AGENT) is True
    assert peer.memory.get(memory_id) is None
    Mem0MemoryManager._instances.pop(canonical_tenant, None)


def test_supported_delete_cannot_cross_repairs_verification_window(
    memory_env, monkeypatch
):
    """A supported delete linearizes outside repair's verified window.

    Repair reads the primary a final time, verifies the indexed row and
    returns. A delete landing inside that window would make repair report
    success for a primary that is already gone. The delete runs on a
    second manager instance, so only the store-backed lease can hold it
    back. It then removes the primary and the indexed row together.
    """
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:delete-race",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.89,
        derived_from=[CitationRef.external("https://source.test/delete-race")],
    )
    memory_id = mm.add_memory(
        content="Primary deleted after repair verified it.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    canonical_tenant = mm._storage_tenant_id
    peer = _peer_manager(memory_env, canonical_tenant)

    entered = threading.Event()
    release = threading.Event()
    real_index_get = mm.provenance_store.get

    def stalled_index_verification(target_memory_id):
        record = real_index_get(target_memory_id)
        entered.set()
        assert release.wait(30) is True
        return record

    monkeypatch.setattr(mm.provenance_store, "get", stalled_index_verification)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            repair_future = pool.submit(mm.repair_provenance, memory_id)
            assert entered.wait(30) is True
            delete_future = pool.submit(
                peer.delete_memory, memory_id, canonical_tenant, AGENT
            )
            threading.Event().wait(0.5)
            assert delete_future.done() is False
            assert mm.memory.get(memory_id) is not None
            release.set()
            assert repair_future.result(timeout=60) == (
                f"prov-{canonical_tenant}-{memory_id}"
            )
            assert delete_future.result(timeout=60) is True
    finally:
        release.set()

    assert peer.memory.get(memory_id) is None
    assert peer.provenance_store.fetch([memory_id]) == {}
    Mem0MemoryManager._instances.pop(canonical_tenant, None)


def test_repair_reports_bounded_conflict_for_external_primary_changes(
    memory_env, monkeypatch
):
    """A raw writer changing every attempt exhausts repair without success."""
    mm = memory_env.manager
    initial = make_provenance(
        written_by="agent:conflict-initial",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.6,
        derived_from=[CitationRef.external("https://source.test/conflict-initial")],
    )
    memory_id = mm.add_memory(
        content="Primary revision zero.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, initial),
        infer=False,
    )
    old_embedder = mm.memory.embedding_model
    mm.memory.embedding_model = _StaticEmbedder()
    real_attach = mm.provenance_store.attach
    mutations = 0

    def attach_then_change_primary(target_memory_id, target_provenance, **kwargs):
        nonlocal mutations
        row_id = real_attach(target_memory_id, target_provenance, **kwargs)
        mutations += 1
        changed = make_provenance(
            written_by=f"agent:conflict-{mutations}",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.6,
            derived_from=[
                CitationRef.external(f"https://source.test/conflict-{mutations}")
            ],
        )
        mm.memory.update(
            target_memory_id,
            data=f"Primary revision {mutations}.",
            metadata=attach_to_metadata({"kind": "entity_fact"}, changed),
        )
        return row_id

    monkeypatch.setattr(mm.provenance_store, "attach", attach_then_change_primary)
    try:
        with pytest.raises(
            ProvenanceRepairConflictError,
            match="primary changed during 3 repair attempts",
        ):
            mm.repair_provenance(memory_id, max_attempts=3)
    finally:
        mm.memory.embedding_model = old_embedder
    assert mutations == 3
    with pytest.raises(ProvenanceConsistencyError):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)

    monkeypatch.setattr(mm.provenance_store, "attach", real_attach)
    assert mm.repair_provenance(memory_id) == (
        f"prov-{mm._storage_tenant_id}-{memory_id}"
    )
    assert mm.memory.get(memory_id)["memory"] == "Primary revision 3."
    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


@pytest.mark.asyncio
async def test_failed_attach_and_compensation_remains_repairable(memory_env):
    """Both boundary errors remain visible and explicit repair converges."""
    mm = memory_env.manager
    agent = MultiDocumentSynthesisAgent(
        deps=MultiDocSynthesisDeps(tenant_id=TENANT),
        config_manager=memory_env.config_manager,
    )
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = "multi_document_synthesis_agent"

    memory_env.proxy.arm(
        lambda method, path, _body: (
            method in {"POST", "PUT"} and "/document/v1/content/provenance_" in path
        ),
        failure=True,
    )
    persist_task = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(
                agent._persist_synthesis(
                    tenant_id=TENANT,
                    answer="A synthesis left for explicit provenance repair.",
                    citation_refs=[
                        CitationRef.external("https://source.test/repair-after-fault")
                    ],
                )
            )
        )
    )
    assert await asyncio.to_thread(memory_env.proxy.entered.wait, 10) is True
    memory_env.proxy.arm(
        lambda method, path, _body: (
            method == "DELETE" and "/document/v1/memory_content/agent_memories_" in path
        ),
        failure=True,
    )
    memory_env.proxy.release.set()

    with pytest.raises(ProvenanceWriteError) as exc_info:
        await persist_task
    error = exc_info.value
    assert type(error.compensation_error) is RuntimeError
    assert "HTTP 400" in str(error.compensation_error)
    memory_id = error.memory_id
    primary = mm.memory.get(memory_id)
    assert primary["memory"] == "A synthesis left for explicit provenance repair."
    assert primary["metadata"]["provenance"]["derived_from"] == [
        {
            "ref_kind": "url",
            "ref_id": "https://source.test/repair-after-fault",
            "label": None,
        }
    ]
    assert mm.provenance_store.fetch([memory_id]) == {}
    with pytest.raises(ProvenanceConsistencyError):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)

    assert mm.repair_provenance(memory_id) == (
        f"prov-{mm._storage_tenant_id}-{memory_id}"
    )
    graph = ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)
    assert [(ref.ref_kind, ref.ref_id) for ref in graph.primary_sources] == [
        ("url", "https://source.test/repair-after-fault"),
        ("memory", memory_id),
    ]

    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_concurrent_repair_upserts_one_stable_row(memory_env, monkeypatch):
    """Concurrent repair of one primary converges on one stable index row."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:concurrent-repair",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.79,
        derived_from=[CitationRef.external("https://source.test/concurrent-repair")],
    )
    result = mm.memory.add(
        "A primary repaired concurrently.",
        user_id=mm._storage_tenant_id,
        agent_id=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    assert result["results"][0]["event"] == "ADD"
    memory_id = result["results"][0]["id"]

    start_barrier = threading.Barrier(2)
    attach_calls = 0
    active_attaches = 0
    max_active_attaches = 0
    attach_lock = threading.Lock()
    real_attach = mm.provenance_store.attach

    def synchronized_attach(target_memory_id, target_provenance, **kwargs):
        nonlocal attach_calls, active_attaches, max_active_attaches
        with attach_lock:
            attach_calls += 1
            active_attaches += 1
            max_active_attaches = max(max_active_attaches, active_attaches)
        try:
            threading.Event().wait(0.1)
            return real_attach(target_memory_id, target_provenance, **kwargs)
        finally:
            with attach_lock:
                active_attaches -= 1

    def repair_from_barrier(_index):
        start_barrier.wait(timeout=10)
        return mm.repair_provenance(memory_id)

    monkeypatch.setattr(mm.provenance_store, "attach", synchronized_attach)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(repair_from_barrier, range(2)))

    expected_row_id = f"prov-{mm._storage_tenant_id}-{memory_id}"
    assert rows == [expected_row_id, expected_row_id]
    assert attach_calls == 2
    assert max_active_attaches == 1
    fetched = mm.provenance_store.fetch([memory_id])
    assert list(fetched) == [memory_id]
    assert fetched[memory_id].derived_from_other == [
        {
            "ref_kind": "url",
            "ref_id": "https://source.test/concurrent-repair",
            "label": None,
        }
    ]

    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_explicit_repair_restores_exact_citation_graph(memory_env):
    """Repair reattaches canonical primary provenance under the stable row id."""
    mm = memory_env.manager
    provenance = make_provenance(
        written_by="agent:repair",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.81,
        derived_from=[
            CitationRef.external("https://source.test/repair"),
            CitationRef.memory("repair-source-memory"),
        ],
        trace_id="repair-trace",
    )
    result = mm.memory.add(
        "A recoverable primary whose secondary write did not run.",
        user_id=mm._storage_tenant_id,
        agent_id=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, provenance),
        infer=False,
    )
    assert result["results"][0]["event"] == "ADD"
    memory_id = result["results"][0]["id"]

    row_id = mm.repair_provenance(memory_id)
    assert row_id == f"prov-{mm._storage_tenant_id}-{memory_id}"
    record = mm.provenance_store.fetch([memory_id])[memory_id]
    assert record.memory_id == memory_id
    assert record.written_by == "agent:repair"
    assert record.derivation_kind == "synthesis"
    assert record.confidence == pytest.approx(0.81)
    assert record.derived_from_memory_ids == ["repair-source-memory"]
    assert record.derived_from_other == [
        {
            "ref_kind": "url",
            "ref_id": "https://source.test/repair",
            "label": None,
        }
    ]
    assert record.trace_id == "repair-trace"

    graph = ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)
    assert [(ref.ref_kind, ref.ref_id) for ref in graph.primary_sources] == [
        ("url", "https://source.test/repair"),
        ("memory", "repair-source-memory"),
    ]
    assert mm.repair_provenance(memory_id) == row_id
    assert mm.provenance_store.fetch([memory_id]) == {memory_id: record}

    mm.memory.delete(memory_id)
    assert mm.memory.get(memory_id) is None


def test_repair_preserves_unrelated_memory_and_distinct_tenant(memory_env):
    """Repair changes only the stable row in its tenant partition."""
    mm = memory_env.manager
    unrelated_provenance = make_provenance(
        written_by="agent:unrelated",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.88,
        derived_from=[CitationRef.external("https://source.test/unrelated")],
    )
    unrelated_id = mm.add_memory(
        content="An unrelated memory in the repaired tenant.",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "external_doc"}, unrelated_provenance),
        infer=False,
    )
    unrelated_primary = mm.memory.get(unrelated_id)
    unrelated_indexed = mm.provenance_store.fetch([unrelated_id])

    target_provenance = make_provenance(
        written_by="agent:isolation-target",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.84,
        derived_from=[CitationRef.external("https://source.test/isolation-target")],
    )
    raw = mm.memory.add(
        "A partial primary repaired without touching its neighbors.",
        user_id=mm._storage_tenant_id,
        agent_id=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, target_provenance),
        infer=False,
    )
    target_id = raw["results"][0]["id"]

    other_tenant = "a2_provenance_other"
    Mem0MemoryManager._instances.pop(other_tenant, None)
    other = Mem0MemoryManager(other_tenant)
    other.initialize(
        backend_host="http://127.0.0.1",
        backend_port=memory_env.proxy.port,
        backend_config_port=memory_env.proxy.port,
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=memory_env.denseon,
        auto_create_schema=True,
        config_manager=memory_env.config_manager,
        schema_loader=memory_env.schema_loader,
    )
    other_provenance = make_provenance(
        written_by="agent:other-tenant",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.93,
        derived_from=[CitationRef.external("https://source.test/other-tenant")],
    )
    other_id = other.add_memory(
        content="A memory in a distinct tenant.",
        tenant_id=other_tenant,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "external_doc"}, other_provenance),
        infer=False,
    )
    other_primary = other.memory.get(other_id)
    other_indexed = other.provenance_store.fetch([other_id])

    assert mm.repair_provenance(target_id) == (
        f"prov-{mm._storage_tenant_id}-{target_id}"
    )
    assert mm.memory.get(unrelated_id) == unrelated_primary
    assert mm.provenance_store.fetch([unrelated_id]) == unrelated_indexed
    assert other.memory.get(other_id) == other_primary
    assert other.provenance_store.fetch([other_id]) == other_indexed
    assert mm.provenance_store.fetch([other_id]) == {}
    assert other.provenance_store.fetch([target_id]) == {}

    for manager, memory_id in (
        (mm, unrelated_id),
        (mm, target_id),
        (other, other_id),
    ):
        manager.memory.delete(memory_id)
        assert manager.memory.get(memory_id) is None
    Mem0MemoryManager._instances.pop(other_tenant, None)


def test_mounted_synthesis_and_citation_routes_reject_torn_provenance(
    memory_env, monkeypatch
):
    """Mounted routes return non-success for write and read inconsistency."""
    mm = memory_env.manager
    monkeypatch.setattr(knowledge_router, "_build_factory", lambda _tenant_id: mm)
    monkeypatch.setattr(
        knowledge_router,
        "_runtime_config_manager",
        lambda: memory_env.config_manager,
    )
    monkeypatch.setattr(
        knowledge_router, "_bind_graph", lambda _agent, _tenant_id: None
    )
    monkeypatch.setattr(
        MultiDocumentSynthesisAgent,
        "_synthesise_without_rlm",
        lambda _self, _query, _documents: "A deterministic mounted synthesis.",
    )
    app = FastAPI()
    app.include_router(knowledge_router.router, prefix="/admin")
    app.dependency_overrides[knowledge_router._get_config_manager] = lambda: (
        memory_env.config_manager
    )

    memory_env.proxy.arm(
        lambda method, path, _body: (
            method in {"POST", "PUT"} and "/document/v1/content/provenance_" in path
        ),
        failure=True,
    )
    memory_env.proxy.release.set()
    with TestClient(app, raise_server_exceptions=False) as client:
        synthesis = client.post(
            f"/admin/tenants/{TENANT}/knowledge/synthesis/multi_doc",
            json={
                "query": "What is the mounted-route result?",
                "documents": [
                    {
                        "content": "Mounted route source content.",
                        "label": "https://source.test/mounted-route",
                    }
                ],
            },
        )
        assert synthesis.status_code == 500
        assert synthesis.text == "Internal Server Error"
        remaining = mm.memory.get_all(
            user_id=mm._storage_tenant_id,
            agent_id="multi_document_synthesis_agent",
        )
        assert remaining == {"results": []}

        partial_provenance = make_provenance(
            written_by="agent:mounted-partial",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.77,
            derived_from=[CitationRef.external("https://source.test/mounted-partial")],
        )
        raw = mm.memory.add(
            "A mounted citation read must reject this partial primary.",
            user_id=mm._storage_tenant_id,
            agent_id=AGENT,
            metadata=attach_to_metadata({"kind": "entity_fact"}, partial_provenance),
            infer=False,
        )
        partial_id = raw["results"][0]["id"]
        citation = client.post(
            f"/admin/tenants/{TENANT}/knowledge/citations/trace",
            json={"memory_id": partial_id},
        )
        assert citation.status_code == 500
        assert citation.text == "Internal Server Error"

    mm.memory.delete(partial_id)
    assert mm.memory.get(partial_id) is None
