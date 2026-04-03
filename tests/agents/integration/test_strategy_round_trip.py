"""
Integration tests for strategy learning round-trip.

Full flow: distill strategies from trigger dataset → store in Vespa memory
→ retrieve via StrategyLearner → format for agent context injection.

Requires Docker for Vespa (Mem0 backend) and optionally Ollama for LLM distillation.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cogniverse_agents.optimizer.strategy_learner import (
    StrategyLearner,
)
from cogniverse_core.registries.backend_registry import BackendRegistry
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.fixture(scope="module")
def vespa_instance():
    """Start Vespa Docker for memory storage integration tests."""
    manager = VespaDockerManager()
    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()

    try:
        container_info = manager.start_container(
            module_name="strategy_learner_integration",
            use_module_ports=True,
        )
        manager.wait_for_config_ready(container_info, timeout=180)

        import time

        time.sleep(15)

        # Deploy metadata schemas (required for Mem0's backend initialization)
        from vespa.package import ApplicationPackage

        import cogniverse_vespa  # noqa: F401
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        # Include agent_memories schema for Mem0 — Vespa requires all schemas
        # in a single deployment package.
        from cogniverse_vespa.json_schema_parser import JsonSchemaParser

        memory_schema_file = SCHEMAS_DIR / "agent_memories_schema.json"
        with open(memory_schema_file) as f:
            memory_schema_json = json.load(f)
        # Create tenant-specific version
        memory_schema_json["name"] = "agent_memories_acme_testuser"
        memory_schema_json["document"]["name"] = "agent_memories_acme_testuser"
        parser = JsonSchemaParser()
        memory_schema = parser.parse_schema(memory_schema_json)

        all_schemas = metadata_schemas + [memory_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)

        manager.wait_for_application_ready(container_info, timeout=120)
        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()
        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()


def _is_ollama_available() -> bool:
    import httpx

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _is_ollama_available(),
    reason="Ollama not available for Mem0 initialization",
)


@pytest.fixture(scope="module")
def memory_manager(vespa_instance):
    """Real Mem0MemoryManager backed by Vespa Docker.

    Requires Ollama for Mem0's LLM extraction.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )

    schema_loader = FilesystemSchemaLoader(SCHEMAS_DIR)

    # Clear singleton cache
    Mem0MemoryManager._instances.clear()

    mm = Mem0MemoryManager(tenant_id="acme:testuser")
    mm.initialize(
        backend_host="http://localhost",
        backend_port=vespa_instance["http_port"],
        llm_model="qwen3:4b",
        embedding_model="nomic-embed-text",
        llm_base_url="http://localhost:11434",
        config_manager=cm,
        schema_loader=schema_loader,
        auto_create_schema=False,  # Pre-deployed in vespa_instance fixture
    )

    yield mm

    Mem0MemoryManager._instances.clear()


@pytest.fixture
def trigger_df():
    """Sample trigger dataset with clear high/low scoring patterns."""
    return pd.DataFrame(
        [
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "man lifting weights in gym",
                "score": 0.95,
                "output": '{"results": [{"video_id": "v1"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "person running on track",
                "score": 0.88,
                "output": '{"results": [{"video_id": "v2"}, {"video_id": "v3"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "what is the dog doing",
                "score": 0.82,
                "output": '{"results": [{"video_id": "v4"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "show me the red car",
                "score": 0.90,
                "output": '{"results": [{"video_id": "v5"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "find the tall building",
                "score": 0.85,
                "output": '{"results": [{"video_id": "v6"}]}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "when did the event happen after the explosion",
                "score": 0.15,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "timeline of events before the crash",
                "score": 0.10,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "sequence during the performance",
                "score": 0.20,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "what happened after the goal",
                "score": 0.18,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "before the sunrise over mountains",
                "score": 0.12,
                "output": '{"results": []}',
            },
        ]
    )


@pytest.mark.integration
@skip_if_no_ollama
class TestStrategyRoundTrip:
    """Full round-trip: distill → store in Vespa → retrieve."""

    @pytest.mark.asyncio
    async def test_distill_store_retrieve(self, memory_manager, trigger_df):
        """Distill strategies from trigger data, store in Vespa, retrieve back."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="acme:testuser",
        )

        # Distill and store
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert len(strategies) >= 1, "Should distill at least one strategy"

        # All strategies should be org-level (pattern extraction default)
        assert all(s.level == "org" for s in strategies)
        assert all(s.source == "pattern_extraction" for s in strategies)

        # Retrieve strategies for search agent
        retrieved = learner.get_strategies_for_agent(
            query="find videos of people exercising",
            agent_name="search",
            top_k=5,
        )

        # Should find at least one strategy
        assert len(retrieved) >= 1, (
            f"Should retrieve stored strategies, got {len(retrieved)}"
        )

        # Verify metadata is correct
        for s in retrieved:
            meta = s.get("metadata", {})
            assert meta.get("type") == "strategy"
            assert float(meta.get("confidence", 0)) >= 0.5

    @pytest.mark.asyncio
    async def test_format_retrieved_strategies(self, memory_manager, trigger_df):
        """Retrieved strategies format correctly for agent context."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="acme:testuser",
        )

        # Store strategies first
        await learner.learn_from_trigger_dataset(trigger_df)

        # Retrieve and format
        retrieved = learner.get_strategies_for_agent(
            query="search for objects in video",
            agent_name="search",
        )

        formatted = StrategyLearner.format_strategies_for_context(retrieved)

        if retrieved:
            assert "## Learned Strategies" in formatted
            assert "confidence:" in formatted

    @pytest.mark.asyncio
    async def test_org_level_scoping(self, memory_manager, trigger_df):
        """Strategies stored at org level are retrievable with org tenant."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="acme:testuser",
        )

        strategies = await learner.learn_from_trigger_dataset(trigger_df)

        # Strategies should be stored with org_id="acme"
        org_strategies = [s for s in strategies if s.tenant_id == "acme"]
        assert len(org_strategies) >= 1

    @pytest.mark.asyncio
    async def test_deduplication_on_repeated_learning(self, memory_manager, trigger_df):
        """Running learning twice doesn't duplicate strategies."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="acme:testuser",
        )

        first_run = await learner.learn_from_trigger_dataset(trigger_df)
        await learner.learn_from_trigger_dataset(trigger_df)  # second run

        # Second run should produce same strategies but dedup should prevent storage
        # (exact count depends on overlap detection, but shouldn't double)
        all_retrieved = learner.get_strategies_for_agent(
            query="search video content",
            agent_name="search",
            top_k=20,
        )

        # Should not have 2x the strategies
        assert len(all_retrieved) <= len(first_run) + 2  # Allow small margin


@pytest.mark.integration
class TestStrategyRoundTripWithMockMemory:
    """Round-trip tests using mock Mem0 (no Ollama dependency)."""

    @pytest.mark.asyncio
    async def test_distill_and_store_calls_memory_manager(self, trigger_df):
        """Verify StrategyLearner calls memory manager with correct args."""
        mm = MagicMock()
        mm.memory = MagicMock()
        mm.add_memory.return_value = "mem_123"
        mm.search_memory.return_value = []

        learner = StrategyLearner(
            memory_manager=mm,
            tenant_id="acme:alice",
        )

        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert len(strategies) >= 1

        # Verify add_memory was called with strategy content and metadata
        assert mm.add_memory.call_count >= 1
        call_args = mm.add_memory.call_args
        assert "[STRATEGY]" in call_args.kwargs.get(
            "content", call_args[1].get("content", "")
        )
        metadata = call_args.kwargs.get("metadata", call_args[1].get("metadata", {}))
        assert metadata.get("type") == "strategy"

    @pytest.mark.asyncio
    async def test_memory_mixin_integration(self, trigger_df):
        """Verify MemoryAwareMixin.get_strategies() calls StrategyLearner."""
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        mixin = MemoryAwareMixin()
        mm = MagicMock()
        mm.memory = MagicMock()
        mm.search_memory.return_value = [
            {
                "memory": "[STRATEGY] Use ColPali for object queries | Applies when: object keywords",
                "metadata": {
                    "type": "strategy",
                    "agent": "search",
                    "confidence": 0.9,
                    "level": "org",
                    "trace_count": 20,
                },
            }
        ]

        mixin.memory_manager = mm
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "acme:alice"
        mixin._memory_initialized = True

        strategies = mixin.get_strategies("find the red car")

        assert strategies is not None
        assert "## Learned Strategies" in strategies
        assert "ColPali" in strategies
        assert "0.90" in strategies

    def test_inject_context_includes_strategies(self):
        """inject_context_into_prompt includes both strategies and memories."""
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        mixin = MemoryAwareMixin()
        mm = MagicMock()
        mm.memory = MagicMock()

        # search_memory returns both regular memories and strategies
        def mock_search(query, tenant_id, agent_name, top_k=5):
            if "[STRATEGY]" in query:
                return [
                    {
                        "memory": "[STRATEGY] Prefer frame search for objects",
                        "metadata": {
                            "type": "strategy",
                            "agent": "search",
                            "confidence": 0.85,
                            "level": "org",
                            "trace_count": 15,
                        },
                    }
                ]
            return [
                {"memory": "Previous query about cars returned good results"},
            ]

        mm.search_memory.side_effect = mock_search

        mixin.memory_manager = mm
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "acme:alice"
        mixin._memory_initialized = True

        result = mixin.inject_context_into_prompt(
            "You are a search agent.", "find the blue car"
        )

        assert "You are a search agent." in result
        assert "## Learned Strategies" in result
        assert "Prefer frame search" in result
        assert "## Relevant Context from Memory:" in result
        assert "Previous query about cars" in result
        assert "## Current Query:" in result
