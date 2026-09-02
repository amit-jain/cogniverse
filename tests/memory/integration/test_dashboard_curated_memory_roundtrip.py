"""Curated text submitted from the dashboard survives storage verbatim.

The Add Memory form in libs/dashboard/.../tabs/memory_management.py stores
exactly what the user typed. Mem0's extraction pass (``infer=True``) may
distil that text down to no facts at all and store no row, which
``Mem0MemoryManager.add_memory`` reports as ``None`` (manager.py:680-693)
and the form can only render as "Failed to add memory".

These tests drive the real manager against a real backend with the call
shape the form uses, and pin that the stored text is byte-identical to the
submitted text. The companion unit test
(tests/dashboard/unit/test_memory_management_forms.py) pins that the form
actually sends ``infer=False``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

pytestmark = pytest.mark.integration

TENANT = "test_tenant"
# Own our state: a per-module agent so the assertions below describe exactly
# the rows this module wrote, not whatever else the shared tenant holds.
AGENT = "dashboard_curated_memory_agent"

# The dashboard form's own placeholder shape. Short, already curated, and
# carrying no extractable "fact" -- the case the extraction pass discards.
CURATED_TEXT = "E2E test memory dashboard-curated-roundtrip"


@pytest.fixture(scope="module")
def dashboard_mm(shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
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
        knowledge_registry=None,
    )
    yield mm
    for row in mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT) or []:
        if row.get("id"):
            mm.delete_memory(memory_id=row["id"], tenant_id=TENANT, agent_name=AGENT)


def _stored_texts(mm: Mem0MemoryManager) -> list[str]:
    rows = mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT) or []
    return [r.get("memory") or r.get("content") or "" for r in rows]


def test_curated_text_is_stored_byte_identical(dashboard_mm) -> None:
    """The form's call shape stores the submitted text unaltered."""
    memory_id = dashboard_mm.add_memory(
        content=CURATED_TEXT,
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata={"topic": "hobbies"},
        infer=False,
    )

    rows = dashboard_mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)

    # The id the form renders back to the user is the id of the row that
    # actually persisted -- not a value invented by the return path.
    assert [r.get("id") for r in rows] == [memory_id]

    # The whole content of this agent's memory, written out: the submitted
    # string and nothing else. An extraction pass that reworded, split or
    # dropped it fails here.
    assert _stored_texts(dashboard_mm) == [CURATED_TEXT]


def test_stored_metadata_survives_the_write(dashboard_mm) -> None:
    """Metadata typed into the form is attached to the stored row."""
    rows = dashboard_mm.get_all_memories(tenant_id=TENANT, agent_name=AGENT)
    assert [r.get("metadata", {}).get("topic") for r in rows] == ["hobbies"]


def test_curated_text_is_retrievable_by_its_own_words(dashboard_mm) -> None:
    """Search returns the row, so the form's "add then search" flow works."""
    hits = dashboard_mm.search_memory(
        query=CURATED_TEXT,
        tenant_id=TENANT,
        agent_name=AGENT,
        top_k=5,
    )
    assert [h.get("memory") for h in hits] == [CURATED_TEXT]
