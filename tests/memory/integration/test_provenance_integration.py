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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_agents.multi_document_synthesis_agent import (
    MultiDocSynthesisDeps,
    MultiDocumentSynthesisAgent,
)
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    ProvenanceConsistencyError,
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
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "a2_provenance_tenant"
AGENT = "a2_provenance_agent"


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
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )

        yield SimpleNamespace(manager=mm, proxy=proxy, config_manager=cm)

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

    with pytest.raises(
        ProvenanceConsistencyError, match="indexed provenance is missing"
    ):
        ProvenanceWalker(mm).walk(memory_id, tenant_id=TENANT)

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

    barrier = threading.Barrier(2)
    attach_calls = 0
    attach_lock = threading.Lock()
    real_attach = mm.provenance_store.attach

    def synchronized_attach(target_memory_id, target_provenance):
        nonlocal attach_calls
        with attach_lock:
            attach_calls += 1
        barrier.wait(timeout=10)
        return real_attach(target_memory_id, target_provenance)

    monkeypatch.setattr(mm.provenance_store, "attach", synchronized_attach)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _index: mm.repair_provenance(memory_id), range(2)))

    expected_row_id = f"prov-{mm._storage_tenant_id}-{memory_id}"
    assert rows == [expected_row_id, expected_row_id]
    assert attach_calls == 2
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
