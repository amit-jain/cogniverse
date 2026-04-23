"""Per-agent strategy filter + agent strategy injection round-trip tests.

Audit fixes #6, #8, #9 — all three broken in the same way: strategies were
stored with a metadata.agent tag at write time but the tag was never read
at retrieval time, so every agent received every strategy regardless of
which agent it was learned for.

Fix #6: StrategyLearner.get_strategies_for_agent() now filters post-fetch by
  metadata.agent, so search_agent strategies are invisible to coding_agent.

Fix #8: RoutingAgent._process_impl() calls inject_context_into_prompt() which
  calls get_strategies() → get_strategies_for_agent(agent_name="routing_agent").
  Before the fix, RoutingAgent lacked MemoryAwareMixin entirely.

Fix #9: Same pattern for CodingAgent — it lacked MemoryAwareMixin before the fix.

These tests use real Mem0 backed by real Vespa (via memory_manager fixture).
No mocks at the Mem0/Vespa boundary.
"""

import time

import pytest

_MEM0_INDEX_WAIT_S = 8  # seconds to wait after add_memory before searching

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.optimizer.strategy_learner import (
    Strategy,
    StrategyLearner,
)


def _seed_strategy(
    memory_manager,
    agent_name: str,
    tenant_id: str,
    text: str,
    applies_when: str = "testing strategy filter",
) -> None:
    """Store one strategy entry tagged for agent_name via real StrategyLearner."""
    learner = StrategyLearner(
        memory_manager=memory_manager,
        tenant_id=tenant_id,
    )
    strategy = Strategy(
        text=text,
        applies_when=applies_when,
        agent=agent_name,
        level="user",
        confidence=0.9,
        source="test",
        tenant_id=tenant_id,
        trace_count=10,
    )
    learner._store_strategy(strategy)


class _MemoryProxy(MemoryAwareMixin):
    """Minimal MemoryAwareMixin carrier used to test inject_context_into_prompt
    without constructing a full agent (which requires telemetry, LLM, etc.)."""

    pass


@pytest.mark.integration
class TestPerAgentStrategyFilter:
    """Fix #6 — strategies are filtered by metadata.agent at retrieval time."""

    def test_search_strategy_not_visible_to_coding_agent(self, memory_manager):
        """Strategy seeded for search_agent must NOT appear in coding_agent results.

        Before fix #6, get_strategies_for_agent ignored the agent_name parameter
        and returned all strategies — this test would have caught it."""
        tenant_id = f"filter_test_{int(time.time() * 1000)}"
        search_text = "use ColBERT reranking when video relevance is ambiguous"

        _seed_strategy(memory_manager, "search_agent", tenant_id, search_text)

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )
        coding_results = learner.get_strategies_for_agent(
            query="how to search for code snippets",
            agent_name="coding_agent",
        )
        texts = [r.get("memory", "") for r in coding_results]
        assert not any("ColBERT" in t or "colbert" in t.lower() for t in texts), (
            f"search_agent strategy leaked to coding_agent — fix #6 per-agent "
            f"filter has regressed. Returned: {texts}"
        )

    def test_search_strategy_visible_to_search_agent(self, memory_manager):
        """Strategy seeded for search_agent MUST appear in search_agent results.

        Fix #6 should not over-filter: agents must still see their own strategies."""
        tenant_id = f"filter_test_pos_{int(time.time() * 1000)}"
        search_text = "use ColBERT reranking when video relevance is ambiguous"

        _seed_strategy(memory_manager, "search_agent", tenant_id, search_text)

        # Allow Vespa indexing to complete — Mem0 LLM extraction + embedding
        # + Vespa write is async in the Vespa distributor. Under load this may
        # take a few seconds before search returns results.
        time.sleep(_MEM0_INDEX_WAIT_S)

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )
        results = learner.get_strategies_for_agent(
            query="how to search for videos",
            agent_name="search_agent",
        )
        assert len(results) > 0, (
            f"search_agent strategy is missing from search_agent retrieval — "
            f"fix #6 filter is too aggressive, blocking the agent's own strategies. "
            f"Got: {results}"
        )

    def test_cross_agent_isolation_both_directions(self, memory_manager):
        """Two distinct strategies for two agents — each sees only its own.

        Seeding search_agent + routing_agent strategies into the same tenant,
        then verifying neither agent sees the other's strategy."""
        tenant_id = f"cross_agent_test_{int(time.time() * 1000)}"

        search_text = "filter video frames by temporal proximity to query keywords"
        routing_text = "escalate to orchestration when query spans multiple modalities"

        _seed_strategy(
            memory_manager,
            "search_agent",
            tenant_id,
            search_text,
            applies_when="video search",
        )
        _seed_strategy(
            memory_manager,
            "routing_agent",
            tenant_id,
            routing_text,
            applies_when="query routing",
        )

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )

        search_results = learner.get_strategies_for_agent(
            query="search for videos about machine learning",
            agent_name="search_agent",
            top_k=10,
        )
        routing_results = learner.get_strategies_for_agent(
            query="route this complex multimodal query",
            agent_name="routing_agent",
            top_k=10,
        )

        search_texts = [r.get("memory", "") for r in search_results]
        routing_texts = [r.get("memory", "") for r in routing_results]

        assert not any("escalate to orchestration" in t for t in search_texts), (
            f"routing_agent strategy leaked into search_agent results: {search_texts}"
        )
        assert not any("temporal proximity" in t for t in routing_texts), (
            f"search_agent strategy leaked into routing_agent results: {routing_texts}"
        )


@pytest.mark.integration
class TestRoutingAgentStrategyInjection:
    """Fix #8 — RoutingAgent.inject_context_into_prompt returns strategies
    from Mem0 when memory is initialized for routing_agent."""

    def test_routing_agent_receives_strategies_via_mixin(self, memory_manager):
        """inject_context_into_prompt must include strategies seeded for routing_agent.

        Before fix #8, RoutingAgent lacked MemoryAwareMixin entirely — this
        method didn't exist on the class, so no strategy injection happened."""
        tenant_id = f"routing_strat_test_{int(time.time() * 1000)}"
        routing_text = "escalate to orchestration when query spans multiple modalities"

        _seed_strategy(
            memory_manager,
            "routing_agent",
            tenant_id,
            routing_text,
            applies_when="multi-modal routing",
        )

        # Allow Vespa to index the new strategy before querying.
        time.sleep(_MEM0_INDEX_WAIT_S)

        # Wire up the mixin directly — no need to construct a full RoutingAgent
        # (which requires telemetry, LLM config, etc.). The mixin is the unit
        # under test here.
        proxy = _MemoryProxy()
        proxy.memory_manager = memory_manager
        proxy._memory_initialized = True
        proxy._memory_agent_name = "routing_agent"
        proxy._memory_tenant_id = tenant_id

        enriched = proxy.inject_context_into_prompt(
            "route this complex multimodal query",
            "route this complex multimodal query",
        )

        assert enriched != "route this complex multimodal query", (
            f"routing_agent strategy not injected into prompt — "
            f"inject_context_into_prompt returned the raw prompt unchanged. "
            f"Fix #8 (RoutingAgent MemoryAwareMixin) may have regressed. "
            f"Enriched prompt: {enriched[:300]}"
        )

    def test_routing_agent_memory_init_sets_agent_name(
        self, memory_manager, vespa_instance, config_manager, schema_loader
    ):
        """initialize_memory() must set _memory_agent_name = 'routing_agent'
        so that get_strategies() retrieves routing_agent-tagged strategies only.

        Before fix #8, RoutingAgent had no mixin and this method didn't exist."""
        from tests.utils.llm_config import (
            get_llm_base_url,
            get_llm_model,
            get_memory_embedding_model,
        )

        proxy = _MemoryProxy()
        success = proxy.initialize_memory(
            agent_name="routing_agent",
            tenant_id="routing_init_test",
            backend_host="http://localhost",
            backend_port=vespa_instance["http_port"],
            backend_config_port=vespa_instance["config_port"],
            llm_model=get_llm_model(),
            embedding_model=get_memory_embedding_model(),
            llm_base_url=get_llm_base_url(),
            config_manager=config_manager,
            schema_loader=schema_loader,
            auto_create_schema=False,
        )
        assert success, "initialize_memory returned False for routing_agent proxy"
        assert proxy._memory_agent_name == "routing_agent", (
            f"_memory_agent_name is {proxy._memory_agent_name!r}, expected 'routing_agent'. "
            f"get_strategies() will filter on the wrong agent name."
        )
        assert proxy.is_memory_enabled(), (
            "is_memory_enabled() returned False after successful initialize_memory()"
        )


@pytest.mark.integration
class TestCodingAgentStrategyInjection:
    """Fix #9 — CodingAgent.inject_context_into_prompt returns strategies
    from Mem0 when memory is initialized for coding_agent."""

    def test_coding_agent_receives_strategies_via_mixin(self, memory_manager):
        """inject_context_into_prompt must include strategies seeded for coding_agent.

        Before fix #9, CodingAgent lacked MemoryAwareMixin entirely."""
        tenant_id = f"coding_strat_test_{int(time.time() * 1000)}"
        coding_text = (
            "prefer test-driven approach: write failing test before implementation"
        )

        _seed_strategy(
            memory_manager,
            "coding_agent",
            tenant_id,
            coding_text,
            applies_when="code generation tasks",
        )

        # Allow Vespa to index the new strategy before querying.
        time.sleep(_MEM0_INDEX_WAIT_S)

        proxy = _MemoryProxy()
        proxy.memory_manager = memory_manager
        proxy._memory_initialized = True
        proxy._memory_agent_name = "coding_agent"
        proxy._memory_tenant_id = tenant_id

        enriched = proxy.inject_context_into_prompt(
            "implement a binary search function",
            "implement a binary search function",
        )

        assert enriched != "implement a binary search function", (
            f"coding_agent strategy not injected into prompt. "
            f"inject_context_into_prompt returned the raw prompt unchanged. "
            f"Fix #9 (CodingAgent MemoryAwareMixin) may have regressed. "
            f"Enriched prompt: {enriched[:300]}"
        )

    def test_coding_strategy_not_visible_to_search_agent(self, memory_manager):
        """Coding strategies must not bleed into search_agent retrieval.

        This is the same cross-contamination check as fix #6 but exercises
        the coding_agent → search_agent direction."""
        tenant_id = f"coding_isolation_test_{int(time.time() * 1000)}"
        coding_text = "always write docstrings before function body"

        _seed_strategy(memory_manager, "coding_agent", tenant_id, coding_text)

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )
        search_results = learner.get_strategies_for_agent(
            query="search for Python functions",
            agent_name="search_agent",
        )
        texts = [r.get("memory", "") for r in search_results]
        assert not any("docstring" in t.lower() for t in texts), (
            f"coding_agent strategy leaked to search_agent — per-agent filter "
            f"is broken in the coding_agent → search_agent direction. Got: {texts}"
        )
