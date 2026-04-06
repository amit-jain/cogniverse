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
from tests.utils.llm_config import get_llm_model

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
        llm_model=get_llm_model(),
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


@pytest.mark.integration
class TestTwoLevelScoping:
    """Verify org:user two-level strategy scoping.

    org:user scoping works via Mem0's user_id field — strategies stored with
    user_id=org_id are retrievable by any user in that org.
    """

    @pytest.mark.asyncio
    async def test_org_strategies_stored_with_org_tenant(
        self, memory_manager, trigger_df
    ):
        """Pattern-extracted strategies use org_id as tenant_id for storage."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert len(strategies) >= 1

        # All pattern-extracted strategies should have level=org
        # and tenant_id=org_id (which is "test_tenant" for simple tenant)
        for s in strategies:
            assert s.level == "org"
            assert s.tenant_id == "test_tenant"

    @pytest.mark.asyncio
    async def test_org_id_parsed_from_colon_format(self):
        """Verify org:user parsing extracts org correctly."""
        from unittest.mock import MagicMock

        mm = MagicMock()
        mm.memory = MagicMock()

        learner1 = StrategyLearner(memory_manager=mm, tenant_id="acme:alice")
        assert learner1.org_id == "acme"
        assert learner1.tenant_id == "acme:alice"

        learner2 = StrategyLearner(memory_manager=mm, tenant_id="acme:bob")
        assert learner2.org_id == "acme"

        # Same org_id means both share org-level strategies
        assert learner1.org_id == learner2.org_id

        learner3 = StrategyLearner(memory_manager=mm, tenant_id="simple_tenant")
        assert learner3.org_id == "simple_tenant"

    @pytest.mark.asyncio
    async def test_user_and_org_retrieval_paths(self):
        """Verify get_strategies_for_agent calls search for both levels."""
        from unittest.mock import MagicMock

        mm = MagicMock()
        mm.memory = MagicMock()
        mm.search_memory.return_value = []

        learner = StrategyLearner(
            memory_manager=mm,
            tenant_id="acme:alice",
        )
        learner.get_strategies_for_agent("test query", "search")

        # Should call search_memory twice: once for user, once for org
        assert mm.search_memory.call_count == 2
        calls = mm.search_memory.call_args_list

        # First call: user-level (tenant_id="acme:alice")
        assert calls[0].kwargs["tenant_id"] == "acme:alice"

        # Second call: org-level (tenant_id="acme")
        assert calls[1].kwargs["tenant_id"] == "acme"


def _is_ollama_available() -> bool:
    import httpx

    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=5.0).status_code == 200
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _is_ollama_available(),
    reason="Ollama not available for LLM distillation",
)


@pytest.mark.integration
@skip_if_no_ollama
class TestLLMDistillation:
    """Test contrastive LLM distillation with real Ollama."""

    @pytest.mark.asyncio
    async def test_llm_distillation_produces_strategies(
        self, memory_manager, trigger_df
    ):
        """Run LLM distillation with real Ollama, verify output quality."""
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        llm_config = LLMEndpointConfig(
            model="ollama_chat/llama3.2",
            api_base="http://localhost:11434",
            temperature=0.1,
            max_tokens=200,
        )

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
            llm_config=llm_config,
        )

        strategies = await learner.learn_from_trigger_dataset(trigger_df)

        # Should have both pattern-extracted AND LLM-distilled strategies
        pattern_strategies = [s for s in strategies if s.source == "pattern_extraction"]
        llm_strategies = [s for s in strategies if s.source == "llm_distillation"]

        assert len(pattern_strategies) >= 1
        assert len(llm_strategies) >= 1, (
            f"LLM distillation should produce at least 1 strategy, got {len(llm_strategies)}"
        )

        for s in llm_strategies:
            assert len(s.text) >= 10, f"Strategy text too short: '{s.text}'"
            assert len(s.applies_when) >= 5, f"Applies_when too short: '{s.applies_when}'"
            assert s.agent == "search"
            assert s.confidence == 0.6  # Default for LLM-distilled
