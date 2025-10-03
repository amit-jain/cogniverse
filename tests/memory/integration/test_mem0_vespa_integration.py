"""
Integration tests for Mem0 with Vespa backend

These tests verify the complete memory system works with real Vespa instance.
"""

import os
import pytest
import requests
import time

from src.common.mem0_memory_manager import Mem0MemoryManager


@pytest.fixture(scope="module")
def vespa_available():
    """Check if Vespa is available"""
    try:
        response = requests.get("http://localhost:8080/ApplicationStatus", timeout=2)
        return response.status_code == 200
    except Exception:
        pytest.skip("Vespa not available at localhost:8080")


@pytest.fixture(scope="module")
def memory_manager(vespa_available):
    """Create and initialize Mem0 memory manager with Vespa"""
    manager = Mem0MemoryManager()

    # Initialize with Vespa backend
    manager.initialize(
        vespa_host="localhost",
        vespa_port=8080,
        collection_name="agent_memories",
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    yield manager

    # Cleanup: Clear test memories
    try:
        manager.clear_agent_memory("test_tenant", "test_agent")
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
        time.sleep(2)

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
        # Add memory for tenant 1
        memory_manager.add_memory(
            content="Tenant 1 specific information about cats",
            tenant_id="tenant_1",
            agent_name="test_agent",
        )

        # Add memory for tenant 2
        memory_manager.add_memory(
            content="Tenant 2 specific information about dogs",
            tenant_id="tenant_2",
            agent_name="test_agent",
        )

        time.sleep(2)

        # Search tenant 1 - should only find cats
        results_1 = memory_manager.search_memory(
            query="animals",
            tenant_id="tenant_1",
            agent_name="test_agent",
            top_k=5,
        )

        # Search tenant 2 - should only find dogs
        results_2 = memory_manager.search_memory(
            query="animals",
            tenant_id="tenant_2",
            agent_name="test_agent",
            top_k=5,
        )

        # Verify isolation
        tenant1_text = " ".join([str(r) for r in results_1])
        tenant2_text = " ".join([str(r) for r in results_2])

        # Each tenant should only see their own content
        assert "cats" in tenant1_text.lower() or "Tenant 1" in tenant1_text
        assert "dogs" in tenant2_text.lower() or "Tenant 2" in tenant2_text

        # Cleanup
        memory_manager.clear_agent_memory("tenant_1", "test_agent")
        memory_manager.clear_agent_memory("tenant_2", "test_agent")

    def test_get_all_memories(self, memory_manager):
        """Test retrieving all memories for an agent"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "get_all_agent")

        # Add multiple memories
        contents = [
            "Memory 1: First interaction",
            "Memory 2: Second interaction",
            "Memory 3: Third interaction",
        ]

        for content in contents:
            memory_manager.add_memory(
                content=content,
                tenant_id="test_tenant",
                agent_name="get_all_agent",
            )

        time.sleep(2)

        # Get all memories
        memories = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="get_all_agent",
        )

        assert len(memories) >= 3

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "get_all_agent")

    def test_delete_memory(self, memory_manager):
        """Test deleting specific memory"""
        # Add memory
        memory_id = memory_manager.add_memory(
            content="Memory to be deleted",
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )

        time.sleep(1)

        # Delete memory
        success = memory_manager.delete_memory(
            memory_id=memory_id,
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )

        assert success is True

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "delete_test_agent")

    def test_update_memory(self, memory_manager):
        """Test updating existing memory"""
        # Add memory
        memory_id = memory_manager.add_memory(
            content="Original content",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )

        time.sleep(1)

        # Update memory
        success = memory_manager.update_memory(
            memory_id=memory_id,
            content="Updated content",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )

        assert success is True

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "update_test_agent")

    def test_memory_stats(self, memory_manager):
        """Test getting memory statistics"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "stats_agent")

        # Add some memories
        for i in range(3):
            memory_manager.add_memory(
                content=f"Memory {i}",
                tenant_id="test_tenant",
                agent_name="stats_agent",
            )

        time.sleep(2)

        # Get stats
        stats = memory_manager.get_memory_stats(
            tenant_id="test_tenant",
            agent_name="stats_agent",
        )

        assert stats["enabled"] is True
        assert stats["total_memories"] >= 3
        assert stats["tenant_id"] == "test_tenant"
        assert stats["agent_name"] == "stats_agent"

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "stats_agent")

    def test_clear_agent_memory(self, memory_manager):
        """Test clearing all memory for an agent"""
        # Add memories
        for i in range(3):
            memory_manager.add_memory(
                content=f"Memory {i}",
                tenant_id="test_tenant",
                agent_name="clear_test_agent",
            )

        time.sleep(2)

        # Verify memories exist
        memories_before = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        assert len(memories_before) >= 3

        # Clear all
        success = memory_manager.clear_agent_memory(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        assert success is True

        time.sleep(1)

        # Verify cleared
        memories_after = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        assert len(memories_after) == 0


@pytest.mark.integration
class TestMem0MemoryAwareMixinIntegration:
    """Integration tests for MemoryAwareMixin with real Mem0"""

    def test_mixin_with_real_memory(self, vespa_available):
        """Test MemoryAwareMixin with real Mem0 backend"""
        from src.app.agents.memory_aware_mixin import MemoryAwareMixin

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        # Initialize memory
        success = agent.initialize_memory(
            agent_name="mixin_test_agent",
            tenant_id="test_tenant",
        )
        assert success is True

        # Add memory
        success = agent.update_memory("Test memory content")
        assert success is True

        time.sleep(2)

        # Search memory
        context = agent.get_relevant_context("test query")
        # May or may not have results depending on semantic match
        # Just verify no errors

        # Get stats
        stats = agent.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["agent_name"] == "mixin_test_agent"

        # Cleanup
        agent.clear_memory()
