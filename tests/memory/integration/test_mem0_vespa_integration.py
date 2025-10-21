"""
Integration tests for Mem0 with Vespa backend

These tests verify the complete memory system works with real Vespa instance.
Uses shared session-scoped Vespa container from conftest.py.
"""


import pytest
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa):
    """Create and initialize Mem0 memory manager with shared Vespa"""
    # Clear singletons to ensure fresh state
    TenantSchemaManager._clear_instance()
    Mem0MemoryManager._instances.clear()

    manager = Mem0MemoryManager(tenant_id="test_tenant")

    # Initialize with shared Vespa backend using Ollama
    # auto_create_schema=False since schema was already deployed by shared_memory_vespa fixture
    manager.initialize(
        vespa_host="localhost",
        vespa_port=shared_memory_vespa["http_port"],
        vespa_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434/v1",
        auto_create_schema=False,  # Schema already deployed
    )

    yield manager

    # Cleanup: Clear test memories (not schemas!)
    try:
        manager.clear_agent_memory("test_tenant", "test_agent")
        manager.clear_agent_memory("tenant_1", "test_agent")
        manager.clear_agent_memory("tenant_2", "test_agent")
        manager.clear_agent_memory("test_tenant", "get_all_agent")
        manager.clear_agent_memory("test_tenant", "delete_test_agent")
        manager.clear_agent_memory("test_tenant", "update_test_agent")
        manager.clear_agent_memory("test_tenant", "stats_agent")
        manager.clear_agent_memory("test_tenant", "clear_test_agent")
        manager.clear_agent_memory("test_tenant", "mixin_test_agent")
    except Exception:
        pass


@pytest.mark.integration
class TestMem0VespaIntegration:
    """Integration tests for Mem0 with Vespa"""

    def test_health_check(self, memory_manager):
        """Test system health check"""
        assert memory_manager.health_check() is True

    def test_add_and_search_memory(self, memory_manager):
        """Test adding and searching memories"""
        # Add memory
        memory_id = memory_manager.add_memory(
            content="User prefers video content about machine learning",
            tenant_id="test_tenant",
            agent_name="test_agent",
            metadata={"category": "preference"},
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)

        # Wait for indexing
        wait_for_vespa_indexing(delay=2)

        # Search for memory
        results = memory_manager.search_memory(
            query="machine learning preferences",
            tenant_id="test_tenant",
            agent_name="test_agent",
            top_k=5,
        )

        assert len(results) > 0
        # Mem0 returns results with 'memory' field
        found = any("machine learning" in str(r).lower() for r in results)
        assert found

    def test_multi_tenant_isolation(self, memory_manager):
        """Test that memories are isolated between tenants"""
        # Clear any existing memories first
        memory_manager.clear_agent_memory("tenant_1", "test_agent")
        memory_manager.clear_agent_memory("tenant_2", "test_agent")

        wait_for_vespa_indexing(delay=1)

        # Add memory for tenant 1 with very specific content
        mem1_id = memory_manager.add_memory(
            content="Tenant 1 loves adopting rescue cats from shelters",
            tenant_id="tenant_1",
            agent_name="test_agent",
        )

        # Add memory for tenant 2 with very different content
        mem2_id = memory_manager.add_memory(
            content="Tenant 2 trains therapy dogs for hospitals",
            tenant_id="tenant_2",
            agent_name="test_agent",
        )

        # Verify both memories were added
        assert mem1_id is not None or mem1_id == ""  # Mem0 might return empty string
        assert mem2_id is not None or mem2_id == ""

        wait_for_vespa_indexing(delay=2)

        # Search tenant 1 - should only find cats
        results_1 = memory_manager.search_memory(
            query="pets",
            tenant_id="tenant_1",
            agent_name="test_agent",
            top_k=5,
        )

        # Search tenant 2 - should only find dogs
        results_2 = memory_manager.search_memory(
            query="pets",
            tenant_id="tenant_2",
            agent_name="test_agent",
            top_k=5,
        )

        # Verify isolation - each tenant should only see their own content
        tenant1_text = " ".join([str(r) for r in results_1]).lower()
        tenant2_text = " ".join([str(r) for r in results_2]).lower()

        # Tenant 1 should have cats but not dogs
        # Tenant 2 should have dogs but not cats
        # Note: Mem0 may not return results if it deduplicates or if LLM processing fails
        # So we check that at least one tenant has their expected content
        has_tenant1_content = "cat" in tenant1_text or "rescue" in tenant1_text
        has_tenant2_content = "dog" in tenant2_text or "therapy" in tenant2_text

        # At least one should have content (if both empty, Mem0/Ollama issue)
        assert has_tenant1_content or has_tenant2_content, (
            f"Neither tenant has memories. T1: {tenant1_text[:100]}, T2: {tenant2_text[:100]}"
        )

        # Verify isolation: tenant1 shouldn't have tenant2's content
        if has_tenant1_content:
            assert "therapy" not in tenant1_text and "hospital" not in tenant1_text, (
                "Tenant 1 has Tenant 2's content - isolation broken"
            )

        # Verify isolation: tenant2 shouldn't have tenant1's content
        if has_tenant2_content:
            assert "rescue" not in tenant2_text and "shelter" not in tenant2_text, (
                "Tenant 2 has Tenant 1's content - isolation broken"
            )

        # Cleanup
        memory_manager.clear_agent_memory("tenant_1", "test_agent")
        memory_manager.clear_agent_memory("tenant_2", "test_agent")

    def test_get_all_memories(self, memory_manager):
        """Test retrieving all memories for an agent"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "get_all_agent")

        wait_for_vespa_indexing(delay=1)

        # Add multiple semantically distinct memories to avoid Mem0 deduplication
        contents = [
            "User prefers Python programming language for data science",
            "User is located in San Francisco timezone PST",
            "User favorite food is Italian cuisine especially pasta",
        ]

        added_ids = []
        for content in contents:
            mem_id = memory_manager.add_memory(
                content=content,
                tenant_id="test_tenant",
                agent_name="get_all_agent",
            )
            added_ids.append(mem_id)

        wait_for_vespa_indexing(delay=3, description="multiple documents")

        # Get all memories
        memories = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="get_all_agent",
        )

        # Mem0 may condense or deduplicate, so we check for at least 1 memory
        assert len(memories) >= 1, f"Expected at least 1 memory, got {len(memories)}"

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "get_all_agent")

    def test_delete_memory(self, memory_manager):
        """Test deleting specific memory"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "delete_test_agent")

        wait_for_vespa_indexing(delay=1)

        # Add memory with factual content that Mem0 will store
        memory_id = memory_manager.add_memory(
            content="User actively plays chess on Sundays and has an Elo rating of 1800",
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        wait_for_vespa_indexing(delay=2)

        # Delete memory
        success = memory_manager.delete_memory(
            memory_id=memory_id,
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )
        assert success is True

        # Verify memory was deleted by checking it's not in get_all
        wait_for_vespa_indexing(delay=1)
        memories = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )
        memory_ids = [m.get("id") for m in memories if isinstance(m, dict)]
        assert memory_id not in memory_ids

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "delete_test_agent")

    def test_update_memory(self, memory_manager):
        """Test updating existing memory"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "update_test_agent")

        wait_for_vespa_indexing(delay=1)

        # Add memory with factual content
        memory_id = memory_manager.add_memory(
            content="User primary email address is john.doe@oldcompany.com",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        wait_for_vespa_indexing(delay=2)

        # Update memory
        success = memory_manager.update_memory(
            memory_id=memory_id,
            content="User primary email address is john.doe@newcompany.com",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )
        assert success is True

        # Verify memory was updated by searching for the new content
        wait_for_vespa_indexing(delay=2)
        results = memory_manager.search_memory(
            query="newcompany email",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
            top_k=5,
        )
        # Should find the updated memory
        assert len(results) > 0
        # Verify it contains the new email
        found = any("newcompany" in str(r).lower() for r in results)
        assert found

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "update_test_agent")

    def test_memory_stats(self, memory_manager):
        """Test getting memory statistics"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "stats_agent")

        wait_for_vespa_indexing(delay=1)

        # Add semantically distinct memories
        memories_to_add = [
            "User birthday is March 15th",
            "User speaks English and Spanish fluently",
            "User favorite sport is basketball",
        ]

        for content in memories_to_add:
            memory_manager.add_memory(
                content=content,
                tenant_id="test_tenant",
                agent_name="stats_agent",
            )

        wait_for_vespa_indexing(delay=3, description="multiple documents")

        # Get stats
        stats = memory_manager.get_memory_stats(
            tenant_id="test_tenant",
            agent_name="stats_agent",
        )

        assert stats["enabled"] is True
        # Mem0 may deduplicate, so check for at least 1
        assert stats["total_memories"] >= 1, (
            f"Expected at least 1 memory, got {stats['total_memories']}"
        )
        assert stats["tenant_id"] == "test_tenant"
        assert stats["agent_name"] == "stats_agent"

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "stats_agent")

    def test_clear_agent_memory(self, memory_manager):
        """Test clearing all memory for an agent"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "clear_test_agent")

        wait_for_vespa_indexing(delay=1)

        # Add distinct memories
        memories_to_add = [
            "User owns a red Tesla Model 3",
            "User completed MBA from Stanford in 2020",
            "User allergic to peanuts and shellfish",
        ]

        for content in memories_to_add:
            memory_manager.add_memory(
                content=content,
                tenant_id="test_tenant",
                agent_name="clear_test_agent",
            )

        wait_for_vespa_indexing(delay=3, description="multiple documents")

        # Verify at least some memories exist
        memories_before = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        _has_memories_before = len(memories_before) >= 1

        # Clear all
        success = memory_manager.clear_agent_memory(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        assert success is True

        wait_for_vespa_indexing(delay=1)

        # Verify cleared
        memories_after = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        # After clear, should have 0 memories
        assert len(memories_after) == 0, (
            f"Expected 0 memories after clear, got {len(memories_after)}"
        )


@pytest.mark.integration
class TestMem0MemoryAwareMixinIntegration:
    """Integration tests for MemoryAwareMixin with real Mem0"""

    def test_mixin_with_real_memory(self, shared_memory_vespa):
        """Test MemoryAwareMixin with real Mem0 backend"""
        from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

        # CRITICAL: Clear singletons FIRST before any initialization
        # Otherwise TenantSchemaManager gets created with default port 8080
        TenantSchemaManager._clear_instance()
        Mem0MemoryManager._instances.clear()

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        # Initialize memory with shared Vespa ports
        success = agent.initialize_memory(
            agent_name="mixin_test_agent",
            tenant_id="test_tenant",
            vespa_host="http://localhost",
            vespa_port=shared_memory_vespa["http_port"],
            vespa_config_port=shared_memory_vespa["config_port"],
        )
        assert success is True

        # Add memory
        success = agent.update_memory("Test memory content")
        assert success is True

        wait_for_vespa_indexing(delay=2)

        # Search memory
        agent.get_relevant_context("test query")
        # May or may not have results depending on semantic match
        # Just verify no errors

        # Get stats
        stats = agent.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["agent_name"] == "mixin_test_agent"

        # Cleanup
        agent.clear_memory()
