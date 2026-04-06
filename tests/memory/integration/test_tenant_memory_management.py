"""
Real integration tests for tenant memory management API.

Uses shared_memory_vespa fixture — dedicated Vespa Docker container
with agent_memories schema deployed. Tests the actual add → search → clear
round-trip that the tenant extensibility API exposes.
"""

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def tenant_mem_manager(shared_memory_vespa):
    """Mem0MemoryManager for tenant memory management tests."""
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
        )
    )

    mm = Mem0MemoryManager(tenant_id="test_tenant")
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="nomic-embed-text",
        llm_base_url="http://localhost:11434",
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )

    # Pre-register schemas so _get_or_create_ingestion_client skips redeployment
    backend = mm.memory.vector_store.backend
    if backend.schema_registry:
        from cogniverse_core.registries.schema_registry import SchemaInfo

        for schema_name, base_name in [
            ("agent_memories_test_tenant", "agent_memories"),
            ("wiki_pages_test_tenant", "wiki_pages"),
        ]:
            key = ("test_tenant", base_name)
            if key not in backend.schema_registry._schemas:
                backend.schema_registry._schemas[key] = SchemaInfo(
                    tenant_id="test_tenant",
                    base_schema_name=base_name,
                    full_schema_name=schema_name,
                    schema_definition="{}",
                    config={},
                    deployment_time="2026-04-06T00:00:00",
                )

    yield mm

    try:
        mm.clear_agent_memory("test_tenant", "_tenant_mgmt_rt")
        mm.clear_agent_memory("test_tenant", "_tenant_mgmt_iso_a")
        mm.clear_agent_memory("test_tenant", "_tenant_mgmt_iso_b")
        mm.clear_agent_memory("test_tenant", "_tenant_mgmt_clear")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


@pytest.mark.integration
class TestTenantMemoryRoundTrip:
    """Full add → search → verify content round-trip with real Vespa + Ollama."""

    def test_add_then_search(self, tenant_mem_manager):
        """Store a memory, search it back, verify content."""
        mm = tenant_mem_manager

        mm.add_memory(
            content="I always prefer using ColPali model for video searches",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_rt",
        )

        results = mm.search_memory(
            query="ColPali video search preference",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_rt",
            top_k=5,
        )

        assert len(results) >= 1, (
            "Should find the memory we just added"
        )
        all_text = " ".join(r.get("memory", "") for r in results).lower()
        assert "colpali" in all_text or "video" in all_text or "search" in all_text, (
            f"Memory should reference ColPali/video/search, got: {all_text}"
        )


@pytest.mark.integration
class TestTenantMemoryIsolation:
    """Agent namespace isolation — memories under agent_a not visible to agent_b."""

    def test_agent_namespacing_isolates(self, tenant_mem_manager, shared_memory_vespa):
        from tests.utils.async_polling import wait_for_vespa_indexing

        mm = tenant_mem_manager
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        mm.add_memory(
            content="I prefer quantum computing approaches for all optimization tasks",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_iso_a",
        )

        wait_for_vespa_indexing(backend_url=vespa_url, delay=3, description="memory add for isolation")

        results_a = mm.search_memory(
            query="quantum computing optimization preference",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_iso_a",
            top_k=5,
        )

        results_b = mm.search_memory(
            query="quantum computing optimization preference",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_iso_b",
            top_k=5,
        )

        assert len(results_a) >= 1, "Agent A should find its own memory"
        assert len(results_b) == 0, (
            f"Agent B should NOT see Agent A's memories, got {len(results_b)}"
        )


@pytest.mark.integration
class TestTenantMemoryClear:
    """Clear all memories for an agent namespace."""

    def test_clear_removes_searchable_memories(self, tenant_mem_manager, shared_memory_vespa):
        """Add memory, verify searchable, clear, verify gone."""
        from tests.utils.async_polling import wait_for_vespa_indexing

        mm = tenant_mem_manager
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        mm.add_memory(
            content="I prefer using TensorFlow over PyTorch for all deep learning tasks",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_clear",
        )

        wait_for_vespa_indexing(backend_url=vespa_url, delay=3, description="memory add")

        before = mm.search_memory(
            query="TensorFlow PyTorch deep learning preference",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_clear",
            top_k=5,
        )
        assert len(before) >= 1, "Memory should exist before clear"

        success = mm.clear_agent_memory(
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_clear",
        )
        assert success is True

        wait_for_vespa_indexing(backend_url=vespa_url, delay=3, description="memory clear")

        after = mm.search_memory(
            query="TensorFlow PyTorch deep learning preference",
            tenant_id="test_tenant",
            agent_name="_tenant_mgmt_clear",
            top_k=5,
        )
        assert len(after) == 0, (
            f"After clear, should find 0 memories, got {len(after)}"
        )
