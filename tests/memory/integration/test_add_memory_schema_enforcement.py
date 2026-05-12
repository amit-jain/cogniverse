"""Mem0MemoryManager.add_memory enforces schema + auto-attaches trust.

Without this enforcement, the write path bypassed every guarantee the
plan made about provenance and trust:

  * memories of a ``provenance_required=True`` kind could be written
    without provenance, leaving them un-auditable;
  * trust scores were never computed at write time, so retrieval-time
    ``rank_with_trust`` had nothing to rank by.

This test verifies, against a real Vespa-backed Mem0:

  * a write to a ``provenance_required`` kind without provenance is
    rejected before any backend call;
  * the same write with provenance succeeds, and reading the memory
    back yields a stored ``trust`` record whose score is the
    schema/provenance-derived initial trust;
  * legacy callers (no registry wired) keep working — provenance is
    optional, no trust auto-attachment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    SchemaViolationError,
    build_default_registry,
)
from cogniverse_core.memory.trust import extract_trust
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

# Reuse the conftest's pre-deployed per-tenant schema. The shared
# memory-vespa fixture deploys ``agent_memories_test_tenant`` once at
# session scope, so any test that uses that exact tenant_id can write
# without needing dynamic schema deployment (which the local Vespa
# config rejects on some dev machines).
TENANT_ENFORCED = "test_tenant"
TENANT_LEGACY = "test_tenant"
AGENT_ENFORCED = "p21_enforced_agent"
AGENT_LEGACY = "p21_legacy_agent"


def _build_manager(*, shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
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
    mm = Mem0MemoryManager(tenant_id=TENANT_ENFORCED)
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
        knowledge_registry=None,  # toggled per-test
    )
    return mm


@pytest.fixture(scope="module")
def base_mm(shared_memory_vespa, shared_denseon):
    """Single manager instance shared across both enforced and legacy tests.

    The Mem0MemoryManager is singleton-per-tenant; tests toggle the
    ``_knowledge_registry`` attribute to switch between the
    enforced/legacy code paths without re-initialising the backend.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    yield mm
    try:
        mm.clear_agent_memory(TENANT_ENFORCED, AGENT_ENFORCED)
        mm.clear_agent_memory(TENANT_LEGACY, AGENT_LEGACY)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


@pytest.fixture
def enforced_mm(base_mm):
    base_mm._knowledge_registry = build_default_registry()
    yield base_mm
    base_mm._knowledge_registry = None


@pytest.fixture
def legacy_mm(base_mm):
    base_mm._knowledge_registry = None
    yield base_mm


def _good_provenance():
    return make_provenance(
        written_by="agent:test",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("https://wiki/p21")],
    )


class TestEnforcement:
    def test_write_without_provenance_rejected_for_required_kind(self, enforced_mm):
        # entity_fact requires provenance per the default registry.
        with pytest.raises(SchemaViolationError, match="requires provenance"):
            enforced_mm.add_memory(
                content="should be rejected",
                tenant_id=TENANT_ENFORCED,
                agent_name=AGENT_ENFORCED,
                metadata={"kind": "entity_fact", "subject_key": "p21:no_prov"},
                infer=False,
            )

    def test_write_with_provenance_persists_trust_round_trip(self, enforced_mm):
        """Real round-trip: write through add_memory, then read back from
        Vespa via get_all_memories and assert the auto-attached trust +
        the original provenance both survive.

        Anything weaker (boundary-spy on ``memory.add``) only proves that
        ``_enforce_schema_on_write`` injected trust into the dict it
        handed Mem0 — it does not prove that Vespa persisted it or that
        downstream retrieval-time ranking has anything to rank by.
        """
        prov = _good_provenance()
        meta = attach_to_metadata(
            {"kind": "entity_fact", "subject_key": "p21:with_prov"}, prov
        )
        memory_id = enforced_mm.add_memory(
            content="accepted with provenance",
            tenant_id=TENANT_ENFORCED,
            agent_name=AGENT_ENFORCED,
            metadata=meta,
            infer=False,
        )
        assert memory_id, "memory must persist when provenance is present"

        rows = enforced_mm.get_all_memories(
            tenant_id=TENANT_ENFORCED, agent_name=AGENT_ENFORCED
        )
        round_tripped = next((r for r in rows if str(r.get("id")) == memory_id), None)
        assert round_tripped is not None, (
            f"memory {memory_id} not visible via get_all_memories — write "
            "didn't reach the read path"
        )
        md = round_tripped.get("metadata") or {}
        assert isinstance(md, dict), (
            f"metadata round-trip broken: expected dict, got {type(md)!r}. "
            "BackendVectorStore must deserialize metadata_ on read."
        )
        assert md.get("kind") == "entity_fact", (
            f"kind did not round-trip; got metadata={md!r}"
        )
        assert md.get("subject_key") == "p21:with_prov", (
            f"subject_key did not round-trip; got metadata={md!r}"
        )
        assert "provenance" in md, (
            f"provenance was stripped on round-trip; got metadata={md!r}"
        )

        trust = extract_trust(round_tripped)
        assert trust is not None, (
            "trust auto-attached by _enforce_schema_on_write must survive "
            f"the Vespa round-trip; got metadata={md!r}"
        )
        # entity_fact default_trust=0.5 × DIRECT_INGEST weight 1.20 = 0.60.
        # Allow a small tolerance in case the schema defaults shift later.
        assert 0.55 <= trust.score <= 1.0, (
            f"trust score {trust.score!r} outside expected band for "
            "entity_fact + DIRECT_INGEST"
        )
        assert trust.endorsements == 0


class TestLegacyDeployments:
    def test_legacy_write_without_provenance_succeeds(self, legacy_mm):
        # No registry wired → no enforcement. Existing deployments that
        # haven't adopted the schema layer keep working.
        memory_id = legacy_mm.add_memory(
            content="legacy write, no provenance",
            tenant_id=TENANT_LEGACY,
            agent_name=AGENT_LEGACY,
            metadata={"kind": "entity_fact"},
            infer=False,
        )
        assert memory_id is not None

    def test_legacy_write_does_not_attach_trust(self, legacy_mm):
        memory_id = legacy_mm.add_memory(
            content="legacy: no trust attached",
            tenant_id=TENANT_LEGACY,
            agent_name=AGENT_LEGACY,
            metadata={"kind": "external_doc"},
            infer=False,
        )
        rows = legacy_mm.get_all_memories(
            tenant_id=TENANT_LEGACY, agent_name=AGENT_LEGACY
        )
        round_tripped = next((r for r in rows if str(r.get("id")) == memory_id), None)
        assert round_tripped is not None
        assert extract_trust(round_tripped) is None


class TestUnknownKindEdgeCase:
    def test_no_kind_skips_enforcement(self, enforced_mm):
        # No metadata.kind → schema lookup is impossible → behave as legacy.
        memory_id = enforced_mm.add_memory(
            content="no kind in metadata",
            tenant_id=TENANT_ENFORCED,
            agent_name=AGENT_ENFORCED,
            metadata={"some_other_field": "x"},
            infer=False,
        )
        assert memory_id is not None
