"""
Integration tests for strategy learning round-trip with real Vespa + Ollama.

Uses shared_memory_vespa + memory_manager fixtures from tests/memory/conftest.py.
Full flow: distill strategies → store in real Vespa memory → retrieve → format.
"""

import logging

import pandas as pd
import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.optimizer.strategy_learner import StrategyLearner

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa):
    """Mem0MemoryManager using shared Vespa — same pattern as test_mem0_vespa_integration."""
    from pathlib import Path

    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    Mem0MemoryManager._instances.clear()

    manager = Mem0MemoryManager(tenant_id="test_tenant")
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    config_manager = ConfigManager(store=config_store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
        )
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    manager.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        llm_base_url="http://localhost:11434",
        auto_create_schema=False,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    yield manager

    try:
        manager.clear_agent_memory("test_tenant", "_strategy_store")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


@pytest.fixture
def trigger_df():
    """Trigger dataset with clear high/low scoring patterns."""
    return pd.DataFrame(
        [
            {"agent": "search", "category": "high_scoring", "query": "man lifting weights in gym", "score": 0.95, "output": '{"results": [{"video_id": "v1"}]}'},
            {"agent": "search", "category": "high_scoring", "query": "person running on track", "score": 0.88, "output": '{"results": [{"video_id": "v2"}, {"video_id": "v3"}]}'},
            {"agent": "search", "category": "high_scoring", "query": "what is the dog doing", "score": 0.82, "output": '{"results": [{"video_id": "v4"}]}'},
            {"agent": "search", "category": "high_scoring", "query": "show me the red car", "score": 0.90, "output": '{"results": [{"video_id": "v5"}]}'},
            {"agent": "search", "category": "high_scoring", "query": "find the tall building", "score": 0.85, "output": '{"results": [{"video_id": "v6"}]}'},
            {"agent": "search", "category": "low_scoring", "query": "when did the event happen after the explosion", "score": 0.15, "output": '{"results": []}'},
            {"agent": "search", "category": "low_scoring", "query": "timeline of events before the crash", "score": 0.10, "output": '{"results": []}'},
            {"agent": "search", "category": "low_scoring", "query": "sequence during the performance", "score": 0.20, "output": '{"results": []}'},
            {"agent": "search", "category": "low_scoring", "query": "what happened after the goal", "score": 0.18, "output": '{"results": []}'},
            {"agent": "search", "category": "low_scoring", "query": "before the sunrise over mountains", "score": 0.12, "output": '{"results": []}'},
        ]
    )


@pytest.mark.integration
class TestStrategyRoundTrip:
    """Full round-trip with real Vespa: distill → store → retrieve."""

    @pytest.mark.asyncio
    async def test_distill_store_retrieve(self, memory_manager, trigger_df):
        """Distill strategies, store in real Vespa, retrieve back."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )

        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert len(strategies) >= 1

        retrieved = learner.get_strategies_for_agent(
            query="find videos of people exercising",
            agent_name="search",
            top_k=5,
        )

        assert len(retrieved) >= 1, (
            f"Should retrieve stored strategies from real Vespa, got {len(retrieved)}"
        )

        for s in retrieved:
            # Mem0 returns memory text in 'memory' key
            assert "memory" in s, f"Expected 'memory' key in result: {s}"
            assert len(s["memory"]) > 0

    @pytest.mark.asyncio
    async def test_format_retrieved_strategies(self, memory_manager, trigger_df):
        """Retrieved strategies format correctly for agent context."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )

        await learner.learn_from_trigger_dataset(trigger_df)

        retrieved = learner.get_strategies_for_agent(
            query="search for objects in video",
            agent_name="search",
        )

        formatted = StrategyLearner.format_strategies_for_context(retrieved)

        if retrieved:
            assert "## Learned Strategies" in formatted
            assert "confidence:" in formatted

    @pytest.mark.asyncio
    async def test_memory_mixin_retrieves_strategies(self, memory_manager, trigger_df):
        """MemoryAwareMixin.get_strategies() returns real strategies from Vespa."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )
        await learner.learn_from_trigger_dataset(trigger_df)

        mixin = MemoryAwareMixin()
        mixin.memory_manager = memory_manager
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "test_tenant"
        mixin._memory_initialized = True

        strategies = mixin.get_strategies("find the red car in video")

        assert strategies is not None
        assert "## Learned Strategies" in strategies

    @pytest.mark.asyncio
    async def test_inject_context_includes_strategies(self, memory_manager, trigger_df):
        """inject_context_into_prompt includes real strategies from Vespa."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )
        await learner.learn_from_trigger_dataset(trigger_df)

        mixin = MemoryAwareMixin()
        mixin.memory_manager = memory_manager
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "test_tenant"
        mixin._memory_initialized = True

        result = mixin.inject_context_into_prompt(
            "You are a search agent.", "find objects in video"
        )

        assert "You are a search agent." in result
        assert "## Learned Strategies" in result
