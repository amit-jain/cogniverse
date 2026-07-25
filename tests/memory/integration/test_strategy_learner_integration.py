"""
Integration tests for strategy learning round-trip with real Vespa + configured LM.

Uses shared_memory_vespa + memory_manager fixtures from tests/memory/conftest.py.
Full flow: distill strategies → store in real Vespa memory → retrieve → format.
"""

import logging

import pandas as pd
import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.optimizer.strategy_learner import StrategyLearner
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)

EXPECTED_PATTERN_PROJECTIONS = [
    (
        "High-scoring search queries average 0.88 while low-scoring average "
        "0.15. Focus on improving queries similar to the low-scoring patterns.",
        "Processing search requests",
        "search",
        "org",
        0.2,
        "pattern_extraction",
        "test_tenant",
        10,
    ),
    (
        "Temporal queries (when, before, after) consistently score poorly "
        "(0% above threshold). Consider alternative search strategy or profile "
        "for these query types.",
        "Query contains temporal keywords",
        "search",
        "org",
        0.25,
        "pattern_extraction",
        "test_tenant",
        5,
    ),
    (
        "Object queries (what, show, find) perform well with current search "
        "configuration (75% score above threshold).",
        "Query contains object keywords",
        "search",
        "org",
        0.2,
        "pattern_extraction",
        "test_tenant",
        4,
    ),
    (
        "High-scoring searches return an average of 1 results. Adjust top_k "
        "accordingly.",
        "Configuring search result count",
        "search",
        "org",
        0.7,
        "pattern_extraction",
        "test_tenant",
        5,
    ),
]


def _strategy_projection(strategy):
    return (
        strategy.text,
        strategy.applies_when,
        strategy.agent,
        strategy.level,
        strategy.confidence,
        strategy.source,
        strategy.tenant_id,
        strategy.trace_count,
    )


def _expected_pattern_contents():
    return {
        (
            f"I prefer the following approach for search: {text} "
            f"I use this when {applies_when}."
        )
        for text, applies_when, *_ in EXPECTED_PATTERN_PROJECTIONS
    }


def _expected_formatted_lines():
    return {
        (
            "- I prefer the following approach for search: "
            f"{text} I use this when {applies_when}. "
            f"(confidence: {confidence:.2f}, from {trace_count} traces, user-level)"
        )
        for (
            text,
            applies_when,
            _agent,
            _level,
            confidence,
            _source,
            _tenant_id,
            trace_count,
        ) in EXPECTED_PATTERN_PROJECTIONS
    }


@pytest.fixture(scope="module")
def memory_manager(shared_memory_vespa, shared_denseon):
    """Mem0MemoryManager using shared Vespa + denseon."""
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
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    manager.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=True,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    yield manager

    try:
        manager.clear_agent_memory("test_tenant", "_strategy_store")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


@pytest.fixture(autouse=True)
def clear_strategy_namespace(memory_manager):
    memory_manager.clear_agent_memory("test_tenant", "_strategy_store_search")
    wait_for_vespa_indexing(delay=1)
    yield
    memory_manager.clear_agent_memory("test_tenant", "_strategy_store_search")


@pytest.fixture
def trigger_df():
    """Trigger dataset with clear high/low scoring patterns."""
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
        assert [_strategy_projection(strategy) for strategy in strategies] == (
            EXPECTED_PATTERN_PROJECTIONS
        )

        retrieved = learner.get_strategies_for_agent(
            query="find videos of people exercising",
            agent_name="search",
            top_k=5,
        )

        assert {row["memory"] for row in retrieved} == _expected_pattern_contents()
        assert {
            (
                row["_level"],
                (row.get("metadata") or {})["agent"],
                (row.get("metadata") or {})["source"],
            )
            for row in retrieved
        } == {("user", "search", "pattern_extraction")}

    @pytest.mark.asyncio
    async def test_format_retrieved_strategies(self, memory_manager, trigger_df):
        """Retrieved strategies format correctly for agent context."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )

        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert [_strategy_projection(strategy) for strategy in strategies] == (
            EXPECTED_PATTERN_PROJECTIONS
        )

        retrieved = learner.get_strategies_for_agent(
            query="search for objects in video",
            agent_name="search",
        )

        formatted = StrategyLearner.format_strategies_for_context(retrieved)
        lines = formatted.splitlines()
        assert lines[0] == "## Learned Strategies"
        assert set(lines[1:]) == _expected_formatted_lines()

    @pytest.mark.asyncio
    async def test_memory_mixin_retrieves_strategies(self, memory_manager, trigger_df):
        """MemoryAwareMixin.get_strategies() returns real strategies from Vespa."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert [_strategy_projection(strategy) for strategy in strategies] == (
            EXPECTED_PATTERN_PROJECTIONS
        )

        mixin = MemoryAwareMixin()
        mixin.memory_manager = memory_manager
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "test_tenant"
        mixin._memory_initialized = True

        strategies = mixin.get_strategies("find the red car in video")

        lines = strategies.splitlines()
        assert lines[0] == "## Learned Strategies"
        assert set(lines[1:]) == _expected_formatted_lines()

    @pytest.mark.asyncio
    async def test_inject_context_includes_strategies(self, memory_manager, trigger_df):
        """inject_context_into_prompt includes real strategies from Vespa."""
        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test_tenant",
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert [_strategy_projection(strategy) for strategy in strategies] == (
            EXPECTED_PATTERN_PROJECTIONS
        )

        mixin = MemoryAwareMixin()
        mixin.memory_manager = memory_manager
        mixin._memory_agent_name = "search"
        mixin._memory_tenant_id = "test_tenant"
        mixin._memory_initialized = True

        result = mixin.inject_context_into_prompt(
            "You are a search agent.", "find objects in video"
        )

        assert result.startswith("You are a search agent.")
        assert "## Learned Strategies" in result
        for line in _expected_formatted_lines():
            assert line in result


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
        assert [_strategy_projection(strategy) for strategy in strategies] == (
            EXPECTED_PATTERN_PROJECTIONS
        )

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


# Runtime LM gate: the requires_lm marker is enforced per test by
# ``pytest_runtest_setup`` in tests/conftest.py (an import-time skipif
# latches the pre-session-fixture endpoint state).
skip_if_no_lm = pytest.mark.requires_lm


@pytest.mark.integration
@skip_if_no_lm
class TestLLMDistillation:
    """Test contrastive LLM distillation against the configured LM."""

    @pytest.mark.asyncio
    async def test_llm_distillation_produces_strategies(
        self, memory_manager, trigger_df
    ):
        """Run LLM distillation with real test LM, verify output quality."""
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from tests.fixtures.llm import (
            resolve_api_key,
            resolve_base_url,
            resolve_prefixed_model,
        )

        llm_config = LLMEndpointConfig(
            model=resolve_prefixed_model(),
            api_base=resolve_base_url(),
            api_key=resolve_api_key(),
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

        assert [
            _strategy_projection(strategy) for strategy in pattern_strategies
        ] == EXPECTED_PATTERN_PROJECTIONS
        assert len(llm_strategies) == 5
        assert {
            (
                strategy.agent,
                strategy.level,
                strategy.confidence,
                strategy.source,
                strategy.tenant_id,
                strategy.trace_count,
                strategy.confirmation_count,
            )
            for strategy in llm_strategies
        } == {("search", "org", 0.6, "llm_distillation", "test_tenant", 2, 1)}
        assert all(10 <= len(strategy.text) <= 500 for strategy in llm_strategies)
        assert all(
            5 <= len(strategy.applies_when) <= 200 for strategy in llm_strategies
        )

        wait_for_vespa_indexing(delay=2)
        stored = memory_manager.get_all_memories(
            tenant_id="test_tenant",
            agent_name="_strategy_store_search",
        )
        stored_llm = [
            memory
            for memory in stored
            if (memory.get("metadata") or {}).get("source") == "llm_distillation"
        ]
        expected_contents = {
            (
                f"I prefer the following approach for search: {strategy.text} "
                f"I use this when {strategy.applies_when}."
            )
            for strategy in llm_strategies
        }
        stored_by_content = {memory["memory"]: memory for memory in stored_llm}
        assert set(stored_by_content) == expected_contents
        for memory in stored_by_content.values():
            assert memory["metadata"]["agent"] == "search"
            assert memory["metadata"]["source"] == "llm_distillation"
            assert memory["metadata"]["confidence"] == 0.6
