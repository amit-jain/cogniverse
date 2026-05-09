"""Knowledge-schema validation gates real Mem0 writes (integration).

The unit tests cover schema validation in isolation. This integration test
proves the validation gate composes with the real Mem0+Vespa write path:
schema-violating writes are rejected before they touch Vespa, schema-valid
writes persist and are searchable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    Pinnable,
    SchemaViolationError,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory_with_registry(shared_memory_vespa, shared_denseon):
    """Real Mem0 manager + the seeded KnowledgeRegistry for schema-aware writes."""
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
    mm = Mem0MemoryManager(tenant_id="schema_test_tenant")
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

    yield mm, build_default_registry()

    try:
        mm.clear_agent_memory("schema_test_tenant", "schema_int_agent")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _validated_add(
    manager,
    registry,
    *,
    kind: str,
    content: str,
    tenant_id: str,
    agent_name: str,
    provenance,
    pinned_by=None,
    extra_meta=None,
) -> str:
    """Schema-aware Mem0 add — validates against registry before persisting."""
    schema = registry.get(kind)
    schema.validate_write(provenance=provenance, pinned_by=pinned_by)

    metadata = dict(extra_meta or {})
    metadata["kind"] = kind
    if provenance is not None:
        metadata["derived_from"] = list(provenance.derived_from)
    if pinned_by is not None:
        metadata["pinned_by"] = pinned_by.value

    return manager.add_memory(
        content=content,
        tenant_id=tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )


class _Provenance:
    def __init__(self, derived_from):
        self.derived_from = list(derived_from)


def test_provenance_required_blocks_write_before_vespa(memory_with_registry):
    """A required-provenance kind without provenance must NOT touch Vespa."""
    mm, registry = memory_with_registry

    # entity_fact requires provenance per the seed registry.
    with pytest.raises(SchemaViolationError):
        _validated_add(
            mm,
            registry,
            kind="entity_fact",
            content="The capital of France is Paris.",
            tenant_id="schema_test_tenant",
            agent_name="schema_int_agent",
            provenance=None,
        )

    # Confirm the rejected write left no trace in Vespa.
    found = mm.search_memory(
        query="capital France",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        top_k=5,
    )
    assert all("capital of France" not in m.get("memory", "") for m in found), (
        "entity_fact without provenance must not have persisted; "
        f"found: {[m.get('memory', '') for m in found]}"
    )


def test_provenance_present_persists_and_carries_derived_from(memory_with_registry):
    """A provenance-required kind with provenance persists with derived_from."""
    mm, registry = memory_with_registry
    prov = _Provenance(derived_from=["src://wiki/paris", "src://atlas/eu"])

    _validated_add(
        mm,
        registry,
        kind="entity_fact",
        content="Schema integration: Paris is the capital of France.",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        provenance=prov,
    )

    found = mm.search_memory(
        query="Schema integration Paris capital",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        top_k=5,
    )
    matched = [m for m in found if "Schema integration" in m.get("memory", "")]
    assert matched, "schema-valid write must round-trip through Vespa search"

    metadata = matched[0].get("metadata", {}) or {}
    # The metadata may be JSON-encoded depending on backend layer; tolerate both.
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)
    assert metadata.get("kind") == "entity_fact"
    assert metadata.get("derived_from") == [
        "src://wiki/paris",
        "src://atlas/eu",
    ]


def test_unauthorized_pin_request_blocks_write(memory_with_registry):
    """Pin requests below the schema's required role are rejected."""
    mm, registry = memory_with_registry

    # tenant_instruction requires TENANT_ADMIN to pin; USER must be denied.
    with pytest.raises(SchemaViolationError):
        _validated_add(
            mm,
            registry,
            kind="tenant_instruction",
            content="this should be rejected",
            tenant_id="schema_test_tenant",
            agent_name="schema_int_agent",
            provenance=None,  # not required
            pinned_by=Pinnable.USER,
        )


def test_admin_can_pin_tenant_instruction(memory_with_registry):
    """A TENANT_ADMIN-level pin request on tenant_instruction succeeds."""
    mm, registry = memory_with_registry

    _validated_add(
        mm,
        registry,
        kind="tenant_instruction",
        content="Schema integration: tenant directive about routing.",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        provenance=None,
        pinned_by=Pinnable.TENANT_ADMIN,
    )

    # Confirm round-trip and pin metadata.
    found = mm.search_memory(
        query="tenant directive routing",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        top_k=5,
    )
    matched = [m for m in found if "tenant directive" in m.get("memory", "")]
    assert matched
    metadata = matched[0].get("metadata", {}) or {}
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)
    assert metadata.get("kind") == "tenant_instruction"
    assert metadata.get("pinned_by") == "tenant_admin"


def test_unknown_kind_falls_back_to_safe_default(memory_with_registry):
    """An unregistered kind is treated as private + provenance-required."""
    mm, registry = memory_with_registry

    # Default fallback requires provenance — a write without it must fail.
    with pytest.raises(SchemaViolationError):
        _validated_add(
            mm,
            registry,
            kind="totally_made_up_kind",
            content="should not persist",
            tenant_id="schema_test_tenant",
            agent_name="schema_int_agent",
            provenance=None,
        )


def test_custom_schema_overrides_default(memory_with_registry):
    """Registering a permissive custom kind allows writes that would otherwise fail."""
    mm, registry = memory_with_registry

    registry.register(
        KnowledgeSchema(
            kind="ephemeral_note",
            provenance_required=False,
            pinnable_by=Pinnable.USER,
        ),
        replace=True,
    )

    _validated_add(
        mm,
        registry,
        kind="ephemeral_note",
        content="Schema integration: a casual note",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        provenance=None,
    )

    found = mm.search_memory(
        query="casual note",
        tenant_id="schema_test_tenant",
        agent_name="schema_int_agent",
        top_k=5,
    )
    matched = [m for m in found if "casual note" in m.get("memory", "")]
    assert matched
