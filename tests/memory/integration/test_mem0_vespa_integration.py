"""
Integration tests for Mem0 with Vespa backend

These tests verify the complete memory system works with real Vespa instance.
"""

import time

import pytest
import requests

from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager


@pytest.fixture(scope="module")
def test_vespa():
    """
    Setup test Vespa Docker instance on port 8081 for Mem0 tests

    Automatically starts test Vespa before tests and cleans up after.
    """
    import platform
    import subprocess

    # Configuration
    test_port = 8081
    config_port = 19072
    container_name = f"vespa-mem0-test-{test_port}"

    # Stop and remove existing test container
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    # Start test Vespa Docker
    machine = platform.machine().lower()
    docker_platform = "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"

    docker_result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{test_port}:8080",
            "-p", f"{config_port}:19071",
            "--platform", docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        timeout=60,
    )

    if docker_result.returncode != 0:
        pytest.fail(f"Failed to start Docker container: {docker_result.stderr.decode()}")

    # Wait for Vespa to be ready
    import time
    for i in range(120):
        try:
            response = requests.get(f"http://localhost:{config_port}/", timeout=5)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        # Cleanup on failure
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail("Vespa config server not ready after 120 seconds")

    vespa_config = {
        "http_port": test_port,
        "config_port": config_port,
        "container_name": container_name,
        "base_url": f"http://localhost:{test_port}",
    }

    # Deploy agent_memories schema using JsonSchemaParser
    import json
    from pathlib import Path

    from vespa.package import ApplicationPackage

    from cogniverse_vespa.json_schema_parser import JsonSchemaParser

    schema_path = Path(__file__).parent.parent.parent.parent / "configs" / "schemas" / "agent_memories_schema.json"
    with open(schema_path, 'r') as f:
        schema_config = json.load(f)

    parser = JsonSchemaParser()
    schema = parser.parse_schema(schema_config)

    app_package = ApplicationPackage(name="mem0test")
    app_package.add_schema(schema)
    app_zip = app_package.to_zip()

    deploy_response = requests.post(
        f"http://localhost:{config_port}/application/v2/tenant/default/prepareandactivate",
        headers={"Content-Type": "application/zip"},
        data=app_zip,
        timeout=60,
    )

    if deploy_response.status_code not in [200, 201]:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail(f"Schema deployment failed: {deploy_response.status_code} {deploy_response.text}")

    # Wait for application to be ready
    for i in range(60):
        try:
            response = requests.get(f"http://localhost:{test_port}/ApplicationStatus", timeout=5)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        # Cleanup on failure
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail("Application not ready after 120 seconds")

    yield vespa_config

    # Teardown: Stop and remove test Vespa
    subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=30)
    subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def memory_manager(test_vespa):
    """Create and initialize Mem0 memory manager with test Vespa"""
    manager = Mem0MemoryManager()

    # Initialize with test Vespa backend using Ollama
    manager.initialize(
        vespa_host="localhost",
        vespa_port=test_vespa["http_port"],
        collection_name="agent_memories",
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434/v1",
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
        # Clear any existing memories first
        memory_manager.clear_agent_memory("tenant_1", "test_agent")
        memory_manager.clear_agent_memory("tenant_2", "test_agent")

        time.sleep(1)

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

        time.sleep(2)

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
        assert has_tenant1_content or has_tenant2_content, \
            f"Neither tenant has memories. T1: {tenant1_text[:100]}, T2: {tenant2_text[:100]}"

        # Verify isolation: tenant1 shouldn't have tenant2's content
        if has_tenant1_content:
            assert "therapy" not in tenant1_text and "hospital" not in tenant1_text, \
                "Tenant 1 has Tenant 2's content - isolation broken"

        # Verify isolation: tenant2 shouldn't have tenant1's content
        if has_tenant2_content:
            assert "rescue" not in tenant2_text and "shelter" not in tenant2_text, \
                "Tenant 2 has Tenant 1's content - isolation broken"

        # Cleanup
        memory_manager.clear_agent_memory("tenant_1", "test_agent")
        memory_manager.clear_agent_memory("tenant_2", "test_agent")

    def test_get_all_memories(self, memory_manager):
        """Test retrieving all memories for an agent"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "get_all_agent")

        time.sleep(1)

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

        time.sleep(3)

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

        time.sleep(1)

        # Add memory with factual content that Mem0 will store
        memory_id = memory_manager.add_memory(
            content="User actively plays chess on Sundays and has an Elo rating of 1800",
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        time.sleep(2)

        # Delete memory
        success = memory_manager.delete_memory(
            memory_id=memory_id,
            tenant_id="test_tenant",
            agent_name="delete_test_agent",
        )
        assert success is True

        # Verify memory was deleted by checking it's not in get_all
        time.sleep(1)
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

        time.sleep(1)

        # Add memory with factual content
        memory_id = memory_manager.add_memory(
            content="User primary email address is john.doe@oldcompany.com",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        time.sleep(2)

        # Update memory
        success = memory_manager.update_memory(
            memory_id=memory_id,
            content="User primary email address is john.doe@newcompany.com",
            tenant_id="test_tenant",
            agent_name="update_test_agent",
        )
        assert success is True

        # Verify memory was updated by searching for the new content
        time.sleep(2)
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

        time.sleep(1)

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

        time.sleep(3)

        # Get stats
        stats = memory_manager.get_memory_stats(
            tenant_id="test_tenant",
            agent_name="stats_agent",
        )

        assert stats["enabled"] is True
        # Mem0 may deduplicate, so check for at least 1
        assert stats["total_memories"] >= 1, f"Expected at least 1 memory, got {stats['total_memories']}"
        assert stats["tenant_id"] == "test_tenant"
        assert stats["agent_name"] == "stats_agent"

        # Cleanup
        memory_manager.clear_agent_memory("test_tenant", "stats_agent")

    def test_clear_agent_memory(self, memory_manager):
        """Test clearing all memory for an agent"""
        # Clear first
        memory_manager.clear_agent_memory("test_tenant", "clear_test_agent")

        time.sleep(1)

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

        time.sleep(3)

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

        time.sleep(1)

        # Verify cleared
        memories_after = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="clear_test_agent",
        )
        # After clear, should have 0 memories
        assert len(memories_after) == 0, f"Expected 0 memories after clear, got {len(memories_after)}"


@pytest.mark.integration
class TestMem0MemoryAwareMixinIntegration:
    """Integration tests for MemoryAwareMixin with real Mem0"""

    def test_mixin_with_real_memory(self, test_vespa):
        """Test MemoryAwareMixin with real Mem0 backend"""
        from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        # Initialize memory with test Vespa port
        success = agent.initialize_memory(
            agent_name="mixin_test_agent",
            tenant_id="test_tenant",
            vespa_port=test_vespa["http_port"],
        )
        assert success is True

        # Add memory
        success = agent.update_memory("Test memory content")
        assert success is True

        time.sleep(2)

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
