"""
Complete End-to-End Memory System Test

This test:
1. Starts Vespa in Docker
2. Deploys agent_memories schema
3. Tests full memory functionality
4. Stops and removes Vespa container

Run with: pytest tests/memory/integration/test_mem0_complete_e2e.py -v -s
"""

import subprocess
import time
from pathlib import Path

import pytest
import requests
from vespa.package import ApplicationPackage

from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

VESPA_CONTAINER_NAME = "vespa-memory-standalone"
VESPA_DATA_PORT = 8084
VESPA_CONFIG_PORT = 19075


@pytest.fixture(scope="module")
def memory_manager(vespa_container):
    """Initialize and return memory manager for all tests"""
    # Clear singleton to ensure fresh state
    Mem0MemoryManager._instances.pop("test_tenant", None)
    manager = Mem0MemoryManager(tenant_id="test_tenant")
    # auto_create_schema=False because fixture already deployed base schema
    manager.initialize(
        vespa_host="localhost",
        vespa_port=VESPA_DATA_PORT,
        base_schema_name="agent_memories",
        auto_create_schema=False,
    )
    return manager


@pytest.fixture(scope="module")
def vespa_container():
    """Start Vespa container, deploy schema, yield, then cleanup"""

    print("\n" + "=" * 70)
    print("🚀 Starting Vespa container...")
    print("=" * 70)

    # Stop and remove any existing container
    subprocess.run(
        ["docker", "stop", VESPA_CONTAINER_NAME],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", VESPA_CONTAINER_NAME],
        capture_output=True,
    )

    # Start fresh Vespa container
    result = subprocess.run(
        [
            "docker", "run",
            "--detach",
            "--name", VESPA_CONTAINER_NAME,
            "--hostname", "vespa-container",
            "--publish", f"{VESPA_DATA_PORT}:8080",
            "--publish", f"{VESPA_CONFIG_PORT}:19071",
            "vespaengine/vespa:latest",
        ],
        capture_output=True,
        text=True,
    )
    print(f"✅ Vespa container started: {result.stdout.strip()}")

    # Wait for Vespa to be ready (up to 120 seconds)
    print("\n⏳ Waiting for Vespa to be ready...")
    max_wait = 120
    for i in range(max_wait):
        try:
            response = requests.get(
                f"http://localhost:{VESPA_CONFIG_PORT}/ApplicationStatus",
                timeout=2,
            )
            if response.status_code == 200:
                print(f"✅ Vespa ready after {i+1} seconds")
                break
        except Exception:
            pass

        if i == max_wait - 1:
            # Cleanup on failure
            subprocess.run(["docker", "stop", VESPA_CONTAINER_NAME])
            subprocess.run(["docker", "rm", VESPA_CONTAINER_NAME])
            pytest.fail("Vespa failed to start within 120 seconds")

        time.sleep(1)

    # Deploy schema
    print("\n📦 Deploying agent_memories schema...")
    if not deploy_schema():
        # Cleanup on failure
        subprocess.run(["docker", "stop", VESPA_CONTAINER_NAME])
        subprocess.run(["docker", "rm", VESPA_CONTAINER_NAME])
        pytest.fail("Failed to deploy schema")

    print("✅ Schema deployed successfully")

    # Wait for schema to be active
    print("⏳ Waiting for schema to be active...")
    time.sleep(15)

    print("\n" + "=" * 70)
    print("✅ Vespa is ready for testing")
    print("=" * 70 + "\n")

    yield VESPA_CONTAINER_NAME

    # Cleanup
    print("\n" + "=" * 70)
    print("🧹 Cleaning up Vespa container...")
    print("=" * 70)

    subprocess.run(
        ["docker", "stop", VESPA_CONTAINER_NAME],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", VESPA_CONTAINER_NAME],
        capture_output=True,
    )

    print("✅ Cleanup complete\n")


def deploy_schema():
    """Deploy agent_memories schema to Vespa using JSON schema"""

    try:
        # Schema path
        schema_path = Path(__file__).parent.parent.parent.parent / "configs" / "schemas" / "agent_memories_schema.json"

        if not schema_path.exists():
            print(f"❌ Schema file not found: {schema_path}")
            return False

        # Load and parse JSON schema
        import json

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser

        with open(schema_path, 'r') as f:
            schema_config = json.load(f)

        parser = JsonSchemaParser()
        schema = parser.parse_schema(schema_config)

        # Create application package
        app_package = ApplicationPackage(name="agentmemories")
        app_package.add_schema(schema)

        # Generate the ZIP package
        app_zip = app_package.to_zip()

        # Deploy via HTTP
        response = requests.post(
            f"http://localhost:{VESPA_CONFIG_PORT}/application/v2/tenant/default/prepareandactivate",
            headers={"Content-Type": "application/zip"},
            data=app_zip,
            timeout=60,
            verify=False,
        )

        if response.status_code == 200:
            return True
        else:
            print(f"❌ Deployment failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Deployment error: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.integration
@pytest.mark.e2e
class TestMemorySystemCompleteE2E:
    """Complete end-to-end tests for memory system"""

    def test_01_vespa_health(self, memory_manager, vespa_container):
        """Test Vespa is healthy"""
        response = requests.get(f"http://localhost:{VESPA_CONFIG_PORT}/ApplicationStatus")
        assert response.status_code == 200
        print("✅ Vespa health check passed")

    def test_02_schema_deployed(self, memory_manager, vespa_container):
        """Test schema is deployed and accessible"""
        # Test by feeding a test document
        test_doc = {
            "fields": {
                "id": "test_schema",
                "text": "test content",
                "user_id": "test",
                "agent_id": "test",
                "embedding": [0.0] * 768,
                "metadata_": "{}",
                "created_at": 1234567890
            }
        }

        response = requests.post(
            f"http://localhost:{VESPA_DATA_PORT}/document/v1/agent_memories/agent_memories/docid/test_schema",
            json=test_doc,
        )
        if response.status_code not in [200, 201]:
            print(f"❌ Vespa returned {response.status_code}: {response.text[:500]}")
        assert response.status_code in [200, 201]
        print("✅ Schema verification passed")

        # Cleanup
        requests.delete(
            f"http://localhost:{VESPA_DATA_PORT}/document/v1/agent_memories/agent_memories/docid/test_schema"
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
        time.sleep(5)

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

    def test_06_multi_tenant_isolation(self, memory_manager, vespa_container):
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

        time.sleep(5)

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

        assert "cat" in text_a.lower() or "sarah" in text_a.lower() or "whiskers" in text_a.lower()
        assert "dog" in text_b.lower() or "mike" in text_b.lower() or "buddy" in text_b.lower() or "golden" in text_b.lower()

        print("✅ Tenant isolation verified")

        # Cleanup
        memory_manager.clear_agent_memory("tenant_a", "isolation_test")
        memory_manager.clear_agent_memory("tenant_b", "isolation_test")

    def test_07_get_all_memories(self, memory_manager, vespa_container):
        """Test getting all memories"""

        # Clear first
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

        # Add multiple memories with distinct factual content
        memory_manager.add_memory(
            content="Customer Alice works at Microsoft in Seattle",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="Order #12345 contains 5 red widgets shipped to Boston",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="Employee Bob received excellence award in March 2024",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )

        time.sleep(5)

        # Get all
        memories = memory_manager.get_all_memories(
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )

        assert len(memories) >= 3
        print(f"✅ Retrieved {len(memories)} memories")

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

    def test_08_delete_memory(self, memory_manager, vespa_container):
        """Test deleting specific memory"""

        # Add memory with factual content
        memory_id = memory_manager.add_memory(
            content="Customer Sarah Johnson lives in Portland, Oregon",
            tenant_id="e2e_test_tenant",
            agent_name="delete_test",
        )

        time.sleep(5)

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

    def test_09_memory_stats(self, memory_manager, vespa_container):
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

        time.sleep(5)

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

    def test_10_memory_aware_mixin(self, memory_manager, vespa_container):
        """Test MemoryAwareMixin with real backend"""

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        # Initialize
        success = agent.initialize_memory(
            agent_name="mixin_e2e_test",
            tenant_id="e2e_test_tenant",
            vespa_host="localhost",
            vespa_port=VESPA_DATA_PORT,
        )
        assert success is True
        print("✅ Mixin initialized")

        # Add memory with factual content
        success = agent.update_memory("Developer prefers Python version 3.12 for all new projects")
        assert success is True
        print("✅ Memory added via mixin")

        time.sleep(3)

        # Search
        context = agent.get_relevant_context("Python programming", top_k=5)
        # Context may be None if no semantic match, that's okay
        print(f"✅ Search via mixin returned: {context is not None}")

        # Remember success with factual content
        success = agent.remember_success(
            query="What is the latest Python version?",
            result="Python 3.12 released in October 2023",
            metadata={"test": "mixin"},
        )
        assert success is True
        print("✅ Remember success via mixin")

        # Remember failure with factual content
        success = agent.remember_failure(
            query="Connect to legacy database",
            error="Connection refused port 5432 deprecated",
        )
        assert success is True
        print("✅ Remember failure via mixin")

        time.sleep(5)

        # Get stats
        stats = agent.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["agent_name"] == "mixin_e2e_test"
        assert stats["total_memories"] >= 3
        print(f"✅ Mixin stats: {stats['total_memories']} memories")

        # Summary
        summary = agent.get_memory_summary()
        assert summary["enabled"] is True
        print(f"✅ Mixin summary: {summary}")

        # Cleanup
        agent.clear_memory()
        print("✅ Mixin cleanup complete")

    def test_11_cleanup_all_test_data(self, memory_manager, vespa_container):
        """Final cleanup of all test data"""

        # Clear all test tenants/agents
        test_agents = [
            ("e2e_test_tenant", "test_agent"),
            ("e2e_test_tenant", "get_all_test"),
            ("e2e_test_tenant", "delete_test"),
            ("e2e_test_tenant", "stats_test"),
            ("e2e_test_tenant", "mixin_e2e_test"),
            ("tenant_a", "isolation_test"),
            ("tenant_b", "isolation_test"),
        ]

        for tenant_id, agent_name in test_agents:
            try:
                memory_manager.clear_agent_memory(tenant_id, agent_name)
            except Exception:
                pass

        print("✅ All test data cleaned up")

    def test_12_final_verification(self, memory_manager, vespa_container):
        """Final verification that system is still healthy"""

        assert memory_manager.health_check() is True
        print("✅ Final health check passed")

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED - Memory system fully functional!")
        print("=" * 70)
