"""Curated memory text survives storage verbatim.

``POST /admin/tenant/{tenant_id}/memories`` (the web client's Add memory)
stores exactly what the user typed: it calls ``Mem0MemoryManager.add_memory``
with ``infer=False``. Mem0's extraction pass (``infer=True``) may distil that
text down to no facts at all and store no row, which
``Mem0MemoryManager.add_memory`` reports as ``None``.

These tests drive the real manager against a real backend with the call
shape the route uses, and pin that the stored text is byte-identical to the
submitted text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager, affirm_memory_profile
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

pytestmark = pytest.mark.integration

TENANT = "test_tenant"
# Own our state: a per-module agent so the assertions below describe exactly
# the rows this module wrote, not whatever else the shared tenant holds.
AGENT = "curated_memory_agent"

# A short, already curated text carrying no extractable "fact" -- the case
# the extraction pass discards.
CURATED_TEXT = "E2E test memory curated-roundtrip"


@pytest.fixture(scope="module")
def curated_mm(shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
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
    affirm_memory_profile(cm)
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
        knowledge_registry=None,
    )
    yield mm
    for row in mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT) or []:
        if row.get("id"):
            mm.delete_memory(memory_id=row["id"], tenant_id=TENANT, agent_name=AGENT)


def _stored_texts(mm: Mem0MemoryManager) -> list[str]:
    rows = mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT) or []
    return [r.get("memory") or r.get("content") or "" for r in rows]


def test_curated_text_is_stored_byte_identical(curated_mm) -> None:
    """The route's call shape stores the submitted text unaltered."""
    memory_id = curated_mm.add_memory(
        content=CURATED_TEXT,
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata={"topic": "hobbies"},
        infer=False,
    )

    rows = curated_mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)

    # The id the route returns to the user is the id of the row that
    # actually persisted -- not a value invented by the return path.
    assert [r.get("id") for r in rows] == [memory_id]

    # The whole content of this agent's memory, written out: the submitted
    # string and nothing else. An extraction pass that reworded, split or
    # dropped it fails here.
    assert _stored_texts(curated_mm) == [CURATED_TEXT]


def test_stored_metadata_survives_the_write(curated_mm) -> None:
    """Metadata sent with the text is attached to the stored row."""
    rows = curated_mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)
    assert [r.get("metadata", {}).get("topic") for r in rows] == ["hobbies"]


def test_curated_text_is_retrievable_by_its_own_words(curated_mm) -> None:
    """Search returns the row, so an "add then search" flow works."""
    hits = curated_mm.search_memory(
        query=CURATED_TEXT,
        tenant_id=TENANT,
        agent_name=AGENT,
        top_k=5,
    )
    assert [h.get("memory") for h in hits] == [CURATED_TEXT]
