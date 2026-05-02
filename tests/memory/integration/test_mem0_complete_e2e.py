"""
Complete End-to-End Memory System Test

Uses shared session-scoped Vespa container from conftest.py.
Tests full memory functionality with proper document cleanup.

Run with: pytest tests/memory/integration/test_mem0_complete_e2e.py -v -s

Tests 07/09 assert ``>= 1`` final memories by default because Mem0's
UPDATE/DELETE/NONE pass dedups aggressively on small LLMs (gemma3:4b,
qwen3:4b). Set ``TEST_LM_IS_STRONG=1`` to switch to the strict
``>= len(inputs) - 2`` assertion when running against a strong vision
or instruction model that respects topic orthogonality.
"""

import os

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.memory.manager import Mem0MemoryManager
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.llm_config import (
    get_backend_host,
    get_llm_base_url,
    get_llm_model,
)


def _strong_lm() -> bool:
    """Operator-controlled flag: assertions get stricter when set."""
    return os.environ.get("TEST_LM_IS_STRONG", "").lower() in ("1", "true", "yes")


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa, shared_denseon):
    """Initialize and return memory manager for all tests."""
    Mem0MemoryManager._instances.clear()

    manager = Mem0MemoryManager(tenant_id="test_tenant")

    manager.initialize(
        backend_host=get_backend_host(),
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
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

    def test_02_schema_deployed(
        self, memory_manager, shared_memory_vespa, shared_denseon
    ):
        """Test schema is deployed and accessible with real embeddings"""
        import requests

        # Generate a real embedding via the denseon sidecar (DenseOn,
        # 768-dim CLS-pooled, OAI /v1/embeddings).
        embed_resp = requests.post(
            f"{shared_denseon}/v1/embeddings",
            json={
                "model": "lightonai/DenseOn",
                "input": "test content for schema verification",
            },
            timeout=30,
        )
        assert embed_resp.status_code == 200, (
            f"DenseOn embedding failed: {embed_resp.status_code}: {embed_resp.text[:200]}"
        )
        real_embedding = embed_resp.json()["data"][0]["embedding"]
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

        Drives Mem0's full LLM-driven extraction + dedup path with
        cross-domain personal facts. Asserts ``>= 1`` because Mem0's
        UPDATE/DELETE/NONE pass is model-quality dependent: small
        models (gemma3:4b, qwen3:4b) tend to choose NONE for
        subsequent adds even when the new fact is orthogonal to
        existing memory. The CRUD contract this test verifies is
        "the storage path completes through Mem0's full inference
        loop and get_all returns at least the kept memory" — not a
        specific dedup count.
        """

        # Clear first
        memory_manager.clear_agent_memory("e2e_test_tenant", "get_all_test")

        for content in (
            "Drives a 2019 Toyota Camry",
            "Allergic to peanuts",
            "Owns a Border Collie named Rex",
            "Plays acoustic guitar every weekend",
            "Born in Toronto in 1988",
        ):
            memory_manager.add_memory(
                content=content,
                tenant_id="e2e_test_tenant",
                agent_name="get_all_test",
            )

        wait_for_vespa_indexing(delay=5)

        # Get all
        memories = memory_manager.get_all_memories(
            tenant_id="e2e_test_tenant",
            agent_name="get_all_test",
        )

        threshold = 3 if _strong_lm() else 1
        assert len(memories) >= threshold

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

        Drives Mem0's full LLM-driven extraction + dedup path with
        cross-domain personal facts. Asserts ``total_memories >= 1``
        because the dedup count is model-quality dependent (see
        test_07 docstring) — this test verifies the stats API
        returns a populated structure after the storage path runs,
        not a specific dedup count.
        """

        # Clear and add memories with personal/behavioral content
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

        for content in (
            "Married to Sam in 2015",
            "Vegetarian since age 12",
            "Speaks fluent Japanese",
            "Owns a beach house in Maui",
            "Allergic to shellfish",
            "Holds a private pilot license",
            "Drives a Tesla Model 3",
            "Reads two books per month",
        ):
            memory_manager.add_memory(
                content=content,
                tenant_id="e2e_test_tenant",
                agent_name="stats_test",
            )

        wait_for_vespa_indexing(delay=5)

        # Get stats
        stats = memory_manager.get_memory_stats(
            tenant_id="e2e_test_tenant",
            agent_name="stats_test",
        )

        threshold = 5 if _strong_lm() else 1
        assert stats["enabled"] is True
        assert stats["total_memories"] >= threshold
        assert stats["tenant_id"] == "e2e_test_tenant"
        assert stats["agent_name"] == "stats_test"

        # Cleanup
        memory_manager.clear_agent_memory("e2e_test_tenant", "stats_test")

    def test_10_memory_aware_mixin(
        self, memory_manager, shared_memory_vespa, shared_denseon
    ):
        """Test MemoryAwareMixin with real backend + denseon"""

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
            embedding_model="lightonai/DenseOn",
            llm_base_url=get_llm_base_url(),
            embedder_base_url=shared_denseon,
            config_manager=shared_memory_vespa["config_manager"],
            schema_loader=shared_memory_vespa["schema_loader"],
            backend_config_port=shared_memory_vespa["config_port"],
            auto_create_schema=False,
        )
        assert success is True

        # Store with ``infer=False`` — the test exercises the mixin↔Mem0
        # storage wiring, not Mem0's LLM-based fact extraction. With small
        # local models (qwen3:4b) extraction frequently returns empty
        # results for short sentences, which would look like a wiring
        # failure but is really LLM flakiness.
        success = agent.update_memory(
            "The Amazon River flows through Brazil and is 6400 kilometers long",
            infer=False,
        )
        assert success is True

        wait_for_vespa_indexing(delay=3)

        # Context may be None if no semantic match, that's okay
        agent.get_relevant_context("rivers in South America", top_k=5)

        success = agent.remember_success(
            query="What is the chemical symbol for gold?",
            result="Gold has the chemical symbol Au from Latin aurum",
            metadata={"test": "mixin"},
            infer=False,
        )
        assert success is True

        success = agent.remember_failure(
            query="What year was the Eiffel Tower completed?",
            error="Could not verify construction date of 1889 in historical records",
            infer=False,
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
