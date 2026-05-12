"""CitationTracingAgent integration against real Mem0+Vespa."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_agents.citation_tracing_agent import (
    CitationTracingAgent,
    CitationTracingDeps,
    CitationTracingInput,
)
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import KnowledgeSchema, build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "test_tenant"  # matches provenance_test_tenant schema deployed by shared_memory_vespa
AGENT_NAME = "citation_int_agent"


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
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )

    yield mm

    try:
        mm.clear_agent_memory(TENANT, AGENT_NAME)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _add(mm, content: str, prov, kind: str = "entity_fact") -> str:
    schema = KnowledgeSchema(kind=kind, provenance_required=True)
    schema.validate_provenance(prov)
    return mm.add_memory(
        content=content,
        tenant_id=TENANT,
        agent_name=AGENT_NAME,
        metadata=attach_to_metadata({"kind": kind}, prov),
        infer=False,
    )


@pytest.mark.asyncio
async def test_agent_walks_real_chain_back_to_primary_sources(memory_env):
    mm = memory_env

    leaf_id = _add(
        mm,
        "leaf — direct ingest of an external doc.",
        make_provenance(
            written_by="agent:ingest",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external("https://wiki/citation-leaf")],
        ),
        kind="external_doc",
    )
    mid_id = _add(
        mm,
        "mid — summary of the leaf.",
        make_provenance(
            written_by="agent:summarizer",
            derivation_kind=DerivationKind.SUMMARIZATION,
            confidence=0.82,
            derived_from=[CitationRef.memory(leaf_id)],
        ),
        kind="external_doc",
    )
    root_id = _add(
        mm,
        "root — synthesised answer.",
        make_provenance(
            written_by="agent:search",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.75,
            derived_from=[
                CitationRef.memory(mid_id),
                CitationRef.external("https://wiki/citation-extra"),
            ],
        ),
        kind="entity_fact",
    )

    # Build the agent and stamp the live memory_manager onto it the way the
    # dispatcher would (via auto-init).
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id=TENANT))
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = AGENT_NAME

    out = await agent._process_impl(
        CitationTracingInput(
            memory_id=root_id,
            tenant_id=TENANT,
            max_depth=5,
            max_nodes=20,
        )
    )

    assert out.root_memory_id == root_id
    chain_ids = {n.memory_id for n in out.nodes}
    assert root_id in chain_ids
    assert mid_id in chain_ids
    assert leaf_id in chain_ids

    primary_keys = {(r.ref_kind, r.ref_id) for r in out.primary_sources}
    assert ("url", "https://wiki/citation-extra") in primary_keys
    assert ("url", "https://wiki/citation-leaf") in primary_keys

    assert out.truncated is False
    assert out.metadata["nodes_visited"] == len(out.nodes)
    assert out.metadata["primary_source_count"] == len(out.primary_sources)


@pytest.mark.asyncio
async def test_agent_returns_empty_when_target_missing(memory_env):
    mm = memory_env
    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id=TENANT))
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = AGENT_NAME

    out = await agent._process_impl(
        CitationTracingInput(memory_id="m_does_not_exist", tenant_id=TENANT)
    )
    assert out.root_memory_id == "m_does_not_exist"
    # Indexed-walker behaviour: the unknown root is still emitted as a
    # placeholder node (depth 0, no provenance, empty excerpt) so callers
    # see exactly which id was unresolvable. The same id is also surfaced
    # as a primary source ref so an audit UI can link back to it.
    assert len(out.nodes) == 1, out.nodes
    placeholder = out.nodes[0]
    assert placeholder.memory_id == "m_does_not_exist"
    assert placeholder.depth == 0
    assert placeholder.written_by is None
    assert placeholder.derivation_kind is None
    assert placeholder.content_excerpt == ""
    keys = {(r.ref_kind, r.ref_id) for r in out.primary_sources}
    assert ("memory", "m_does_not_exist") in keys
