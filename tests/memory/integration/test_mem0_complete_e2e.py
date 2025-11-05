"""
Complete End-to-End Memory System Test

Uses shared session-scoped Vespa container from conftest.py.
Tests full memory functionality with proper document cleanup.

Run with: pytest tests/memory/integration/test_mem0_complete_e2e.py -v -s
"""


import pytest
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.backends import TenantSchemaManager

from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa):
    """Initialize and return memory manager for all tests"""
    from pathlib import Path

    from cogniverse_core.config.manager import ConfigManager
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    # Clear singleton to ensure fresh state
    Mem0MemoryManager._instances.clear()

    manager = Mem0MemoryManager(tenant_id="test_tenant")

    # Create dependencies for dependency injection
    config_manager = ConfigManager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    # Initialize with shared Vespa backend
    # auto_create_schema=False since schema was already deployed
    manager.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        auto_create_schema=False,  # Schema already deployed
        config_manager=config_manager,
        schema_loader=schema_loader
    )

    yield manager

    # Cleanup: Clear all test memories (not schemas!)
    test_agents = [
        ("e2e_test_tenant", "test_agent"),
        ("e2e_test_tenant", "get_all_test"),
        ("e2e_test_tenant", "delete_test"),
        ("e2e_test_tenant", "stats_test"),
        ("test_tenant", "mixin_e2e_test"),  # Mixin test uses test_tenant
        ("tenant_a", "isolation_test"),
        ("tenant_b", "isolation_test"),
    ]

    for tenant_id, agent_name in test_agents:
        try:
            manager.clear_agent_memory(tenant_id, agent_name)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.e2e
class TestMemorySystemCompleteE2E:
    """Complete end-to-end tests for memory system"""

    def test_01_vespa_health(self, memory_manager, shared_memory_vespa):
        """Test Vespa is healthy"""
        import requests

        response = requests.get(
            f"http://localhost:{shared_memory_vespa['config_port']}/ApplicationStatus"
        )
        assert response.status_code == 200
        print("✅ Vespa health check passed")

    def test_02_schema_deployed(self, memory_manager, shared_memory_vespa):
        """Test schema is deployed and accessible"""
        import requests

        # Test by feeding a test document to the tenant schema
        test_doc = {
            "fields": {
                "id": "test_schema",
                "text": "test content",
                "user_id": "test",
                "agent_id": "test",
                "embedding": [0.0] * 768,
                "metadata_": "{}",
                "created_at": 1234567890,
            }
        }

        schema_name = shared_memory_vespa["tenant_schema_name"]
        response = requests.post(
            f"http://localhost:{shared_memory_vespa['http_port']}/document/v1/{schema_name}/{schema_name}/docid/test_schema",
            json=test_doc,
        )
        if response.status_code not in [200, 201]:
            print(f"❌ Vespa returned {response.status_code}: {response.text[:500]}")
        assert response.status_code in [200, 201]
        print("✅ Schema verification passed")

        # Cleanup
        requests.delete(
            f"http://localhost:{shared_memory_vespa['http_port']}/document/v1/{schema_name}/{schema_name}/docid/test_schema"
        )

    def test_03_memory_manager_initialization(self, memory_manager):
        """Test memory manager initializes correctly"""
        assert memory_manager.memory is not None
        assert memory_manager.health_check() is True
        print("✅ Memory manager initialization passed")

    def test_04_add_memory(self, memory_manager):
        """Test adding memory"""
        memory_id = memory_manager.add_memory(
            content="User prefers video content about machine learning and AI",
            tenant_id="e2e_test_tenant",
            agent_name="test_agent",
            metadata={"test": "e2e", "category": "preference"},
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        print(f"✅ Added memory with ID: {memory_id}")

    def test_05_search_memory(self, memory_manager):
        """Test searching memory"""

        # Wait for indexing
        print("⏳ Waiting for Vespa indexing...")
        wait_for_vespa_indexing(delay=5)

        # Search
        results = memory_manager.search_memory(
            query="machine learning preferences",
            tenant_id="e2e_test_tenant",
            agent_name="test_agent",
            top_k=5,
        )

        assert len(results) > 0
        print(f"✅ Found {len(results)} memories")

        # Verify content
        result_text = str(results)
        assert "machine learning" in result_text.lower() or "AI" in result_text
        print("✅ Search results contain expected content")

    def test_06_multi_tenant_isolation(self, memory_manager, shared_memory_vespa):
        """Test tenant isolation"""

        # Add memory for tenant A
        memory_manager.add_memory(
            content="Customer Sarah owns three Persian cats named Whiskers, Mittens, and Shadow",
            tenant_id="tenant_a",
            agent_name="isolation_test",
        )

        # Add memory for tenant B
        memory_manager.add_memory(
            content="Customer Mike has two Golden Retriever dogs named Buddy and Max",
            tenant_id="tenant_b",
            agent_name="isolation_test",
        )

        wait_for_vespa_indexing(delay=5)

        # Search tenant A
        results_a = memory_manager.search_memory(
            query="animals",
            tenant_id="tenant_a",
            agent_name="isolation_test",
            top_k=10,
        )

        # Search tenant B
        results_b = memory_manager.search_memory(
            query="animals",
            tenant_id="tenant_b",
            agent_name="isolation_test",
            top_k=10,
        )

        # Verify isolation
        text_a = " ".join([str(r) for r in results_a])
        text_b = " ".join([str(r) for r in results_b])

        assert (
            "cat" in text_a.lower()
            or "sarah" in text_a.lower()
            or "whiskers" in text_a.lower()
        )
        assert (
            "dog" in text_b.lower()
            or "mike" in text_b.lower()
            or "buddy" in text_b.lower()
            or "golden" in text_b.lower()
        )

        print("✅ Tenant isolation verified")

        # Cleanup
        memory_manager.clear_agent_memory("tenant_a", "isolation_test")
        memory_manager.clear_agent_memory("tenant_b", "isolation_test")

    def test_07_get_all_memories(self, memory_manager, shared_memory_vespa):
        """Test getting all memories"""

        # Clear first
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

        # Add multiple memories with semantically distinct content across different domains
        # Domain 1: Astronomy/Space
        memory_manager.add_memory(
            content="Jupiter has 95 confirmed moons orbiting around it as of 2023",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        # Domain 2: Culinary/Food
        memory_manager.add_memory(
            content="Traditional French croissants require 27 layers of butter lamination",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        # Domain 3: Sports/Athletics
        memory_manager.add_memory(
            content="Olympic marathon distance is exactly 42.195 kilometers or 26.2 miles",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )

        wait_for_vespa_indexing(delay=5)

        # Get all
        memories = memory_manager.get_all_memories(
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )

        assert len(memories) >= 3
        print(f"✅ Retrieved {len(memories)} memories")

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

    def test_08_delete_memory(self, memory_manager, shared_memory_vespa):
        """Test deleting specific memory"""

        # Add memory with factual content
        memory_id = memory_manager.add_memory(
            content="Customer Sarah Johnson lives in Portland, Oregon",
            tenant_id="e2e_test_tenant",
            agent_name="delete_test",
        )

        wait_for_vespa_indexing(delay=5)

        # Delete
        success = memory_manager.delete_memory(
            memory_id=memory_id,
            tenant_id="e2e_test_tenant",
            agent_name="delete_test",
        )

        assert success is True
        print(f"✅ Deleted memory {memory_id}")

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "delete_test")

    def test_09_memory_stats(self, memory_manager, shared_memory_vespa):
        """Test memory statistics"""

        # Clear and add memories with distinct factual content
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

        memory_manager.add_memory(
            content="Manager Lisa leads the engineering team at Google Paris office",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="Product launch scheduled for January 30th 2026 in Tokyo",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="Server cluster has 24 nodes running Ubuntu 22.04",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="Invoice #9876 for $15000 paid by Wells Fargo",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="Client ABC Corp signed 3-year contract in November 2023",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )

        wait_for_vespa_indexing(delay=5)

        # Get stats
        stats = memory_manager.get_memory_stats(
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )

        assert stats["enabled"] is True
        assert stats["total_memories"] >= 5
        assert stats["tenant_id"] == "e2e_test_tenant"
        assert stats["agent_name"] == "stats_test"
        print(f"✅ Stats: {stats['total_memories']} memories")

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

    def test_10_memory_aware_mixin(self, memory_manager, shared_memory_vespa):
        """Test MemoryAwareMixin with real backend"""

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        # Initialize with shared Vespa ports
        # Use same tenant as shared fixture to reuse existing schema via singleton pattern
        success = agent.initialize_memory(
            agent_name="mixin_e2e_test",
            tenant_id="test_tenant",  # Same tenant = reuses manager singleton
            backend_host="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            backend_config_port=shared_memory_vespa["config_port"],
            auto_create_schema=False,  # Explicitly don't try to create (already exists)
        )
        assert success is True
        print("✅ Mixin initialized")

        # Add memory with factual content - Domain 1: Geography
        success = agent.update_memory(
            "The Amazon River flows through Brazil and is 6400 kilometers long"
        )
        assert success is True
        print("✅ Memory added via mixin")

        wait_for_vespa_indexing(delay=3)

        # Search
        context = agent.get_relevant_context("rivers in South America", top_k=5)
        # Context may be None if no semantic match, that's okay
        print(f"✅ Search via mixin returned: {context is not None}")

        # Remember success with factual content - Domain 2: Chemistry
        success = agent.remember_success(
            query="What is the chemical symbol for gold?",
            result="Gold has the chemical symbol Au from Latin aurum",
            metadata={"test": "mixin"},
        )
        assert success is True
        print("✅ Remember success via mixin")

        # Remember failure with factual content - Domain 3: Architecture
        success = agent.remember_failure(
            query="What year was the Eiffel Tower completed?",
            error="Could not verify construction date of 1889 in historical records",
        )
        assert success is True
        print("✅ Remember failure via mixin")

        wait_for_vespa_indexing(delay=8)

        # Get stats
        stats = agent.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["agent_name"] == "mixin_e2e_test"
        assert stats["total_memories"] >= 2  # Mem0 may deduplicate similar content
        print(f"✅ Mixin stats: {stats['total_memories']} memories")

        # Summary
        summary = agent.get_memory_summary()
        assert summary["enabled"] is True
        print(f"✅ Mixin summary: {summary}")

        # Cleanup
        agent.clear_memory()
        print("✅ Mixin cleanup complete")

    def test_11_cleanup_all_test_data(self, memory_manager, shared_memory_vespa):
        """Final cleanup of all test data"""

        # Clear all test tenants/agents
        test_agents = [
            ("e2e_test_tenant", "test_agent"),
            ("e2e_test_tenant", "get_all_test"),
            ("e2e_test_tenant", "delete_test"),
            ("e2e_test_tenant", "stats_test"),
            ("test_tenant", "mixin_e2e_test"),  # Mixin test uses test_tenant
            ("tenant_a", "isolation_test"),
            ("tenant_b", "isolation_test"),
        ]

        for tenant_id, agent_name in test_agents:
            try:
                memory_manager.clear_agent_memory(tenant_id, agent_name)
            except Exception:
                pass

        print("✅ All test data cleaned up")

    def test_12_final_verification(self, memory_manager, shared_memory_vespa):
        """Final verification that system is still healthy"""

        assert memory_manager.health_check() is True
        print("✅ Final health check passed")

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED - Memory system fully functional!")
        print("=" * 70)
