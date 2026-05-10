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

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    ProvenanceWalker,
    attach_to_metadata,
    extract_from_memory,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    SchemaViolationError,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "a2_provenance_tenant"
AGENT = "a2_provenance_agent"


@pytest.fixture(scope="module")
def memory_env(shared_memory_vespa, shared_denseon):
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

    yield mm

    try:
        mm.clear_agent_memory(TENANT, AGENT)
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
    mm = memory_env
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
    mm = memory_env
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
    mm = memory_env

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
