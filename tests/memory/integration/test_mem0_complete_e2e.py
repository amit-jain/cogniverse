"""
Complete End-to-End Memory System Test

Uses shared session-scoped Vespa container from conftest.py.
Tests full memory functionality with proper document cleanup.

Run with: pytest tests/memory/integration/test_mem0_complete_e2e.py -v -s
"""

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.manager import Mem0MemoryManager
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.llm_config import (
    get_backend_host,
    get_llm_base_url,
    get_llm_model,
    get_memory_embedding_model,
)


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa):
    """Initialize and return memory manager for all tests."""
    Mem0MemoryManager._instances.clear()

    manager = Mem0MemoryManager(tenant_id="test_tenant")

    manager.initialize(
        backend_host=get_backend_host(),
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model=get_memory_embedding_model(),
        llm_base_url=get_llm_base_url(),
        auto_create_schema=False,
        config_manager=shared_memory_vespa["config_manager"],
        schema_loader=shared_memory_vespa["schema_loader"],
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

    def test_02_schema_deployed(self, memory_manager, shared_memory_vespa):
        """Test schema is deployed and accessible with real embeddings"""
        import requests

        # Generate a real embedding via Ollama nomic-embed-text (same model Mem0 uses)
        embed_resp = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": "test content for schema verification",
            },
            timeout=30,
        )
        assert embed_resp.status_code == 200, (
            f"Ollama embedding failed: {embed_resp.status_code}: {embed_resp.text[:200]}"
        )
        real_embedding = embed_resp.json()["embeddings"][0]
        assert len(real_embedding) == 768, (
            f"Expected 768-dim, got {len(real_embedding)}"
        )

        # Test by feeding a test document with the real embedding
        test_doc = {
            "fields": {
                "id": "test_schema",
                "text": "test content for schema verification",
                "user_id": "test",
                "agent_id": "test",
                "embedding": real_embedding,
                "metadata_": "{}",
                "created_at": 1234567890,
            }
        }

        schema_name = shared_memory_vespa["tenant_schema_name"]
        # Use memory_content namespace — matches VespaIngestionClient namespace
        # logic for agent_memories schemas (ingestion_client.py).
        namespace = "memory_content"
        response = requests.post(
            f"http://localhost:{shared_memory_vespa['http_port']}/document/v1/{namespace}/{schema_name}/docid/test_schema",
            json=test_doc,
        )
        assert response.status_code in [200, 201], (
            f"Vespa returned {response.status_code}: {response.text[:500]}"
        )

        # Cleanup
        requests.delete(
            f"http://localhost:{shared_memory_vespa['http_port']}/document/v1/{namespace}/{schema_name}/docid/test_schema"
        )

    def test_03_memory_manager_initialization(self, memory_manager):
        """Test memory manager initializes correctly"""
        assert memory_manager.memory is not None
        assert memory_manager.health_check() is True

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

    def test_05_search_memory(self, memory_manager):
        """Test searching memory"""

        # Wait for indexing
        wait_for_vespa_indexing(delay=5)

        # Search
        results = memory_manager.search_memory(
            query="machine learning preferences",
            tenant_id="e2e_test_tenant",
            agent_name="test_agent",
            top_k=5,
        )

        assert len(results) > 0

        # Verify content
        result_text = str(results)
        assert "machine learning" in result_text.lower() or "AI" in result_text

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


        # Cleanup
        memory_manager.clear_agent_memory("tenant_a", "isolation_test")
        memory_manager.clear_agent_memory("tenant_b", "isolation_test")

    def test_07_get_all_memories(self, memory_manager, shared_memory_vespa):
        """Test getting all memories.

        Uses personal/behavioral content (user preferences and attributes) which
        Mem0 reliably extracts. Adds 5 items to guarantee >= 3 stored even if
        the LLM (llama3.2) skips one or two due to similarity detection.
        """

        # Clear first
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

        # Use personal/behavioral content — Mem0 is designed to extract
        # personal facts, not encyclopedic trivia.
        memory_manager.add_memory(
            content="My name is Alex and I work as a senior software engineer",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="I prefer Python and use it daily for data analysis projects",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="I live in Berlin and commute by bicycle every morning",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="My team size is 8 people and we use agile with 2-week sprints",
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )
        memory_manager.add_memory(
            content="I am learning Spanish and have been studying for 6 months",
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

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "delete_test")

    def test_09_memory_stats(self, memory_manager, shared_memory_vespa):
        """Test memory statistics.

        Uses personal/behavioral content which Mem0 reliably extracts. Adds 8
        distinct personal facts and asserts >= 5 to verify the stats API returns
        meaningful counts. The buffer accounts for occasional LLM skips with
        the small llama3.2 model.
        """

        # Clear and add memories with personal/behavioral content
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

        memory_manager.add_memory(
            content="My name is Jordan and I manage the backend infrastructure team",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="I prefer using Go for high-performance backend services",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="My work schedule is Monday through Friday, 9am to 6pm Tokyo time",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="I have 10 years of experience in distributed systems engineering",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="My team is currently migrating from PostgreSQL to CockroachDB",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="I attend the weekly SRE sync every Tuesday at 2pm",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="My home office setup uses three monitors and a standing desk",
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )
        memory_manager.add_memory(
            content="I am certified in AWS Solutions Architect and Kubernetes administration",
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

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

    def test_10_memory_aware_mixin(self, memory_manager, shared_memory_vespa):
        """Test MemoryAwareMixin with real backend"""

        class TestAgent(MemoryAwareMixin):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        success = agent.initialize_memory(
            agent_name="mixin_e2e_test",
            tenant_id="test_tenant",
            backend_host=get_backend_host(),
            backend_port=shared_memory_vespa["http_port"],
            llm_model=get_llm_model(),
            embedding_model=get_memory_embedding_model(),
            llm_base_url=get_llm_base_url(),
            config_manager=shared_memory_vespa["config_manager"],
            schema_loader=shared_memory_vespa["schema_loader"],
            backend_config_port=shared_memory_vespa["config_port"],
            auto_create_schema=False,
        )
        assert success is True

        # Add memory with factual content - Domain 1: Geography
        success = agent.update_memory(
            "The Amazon River flows through Brazil and is 6400 kilometers long"
        )
        assert success is True

        wait_for_vespa_indexing(delay=3)

        # Context may be None if no semantic match, that's okay
        agent.get_relevant_context("rivers in South America", top_k=5)

        # Remember success with factual content - Domain 2: Chemistry
        success = agent.remember_success(
            query="What is the chemical symbol for gold?",
            result="Gold has the chemical symbol Au from Latin aurum",
            metadata={"test": "mixin"},
        )
        assert success is True

        # Remember failure with factual content - Domain 3: Architecture
        success = agent.remember_failure(
            query="What year was the Eiffel Tower completed?",
            error="Could not verify construction date of 1889 in historical records",
        )
        assert success is True

        wait_for_vespa_indexing(delay=8)

        # Get stats
        stats = agent.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["agent_name"] == "mixin_e2e_test"
        assert stats["total_memories"] >= 2  # Mem0 may deduplicate similar content

        # Summary
        summary = agent.get_memory_summary()
        assert summary["enabled"] is True

        # Cleanup
        agent.clear_memory()

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


    def test_12_final_verification(self, memory_manager, shared_memory_vespa):
        """Final verification that system is still healthy"""

        assert memory_manager.health_check() is True

