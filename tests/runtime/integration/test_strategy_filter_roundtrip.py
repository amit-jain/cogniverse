"""Per-agent strategy filter + agent strategy injection round-trip tests.

``StrategyLearner.get_strategies_for_agent()`` filters stored strategies by
``metadata.agent`` at retrieval time so an agent only receives strategies
learned for itself (search_agent strategies never leak into coding_agent
prompts, etc.).

Agents that include ``MemoryAwareMixin`` call ``inject_context_into_prompt``
in ``_process_impl`` to prepend matching strategies to their LLM prompt; the
mixin routes the fetch through ``get_strategies_for_agent(agent_name=...)``.

These tests use real Mem0 backed by real Vespa (via memory_manager fixture).
No mocks at the Mem0/Vespa boundary.
"""

import time
from typing import Any, Callable, List

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.optimizer.strategy_learner import (
    STRATEGY_AGENT_NAME,
    Strategy,
    StrategyLearner,
)

# How long a seeded strategy may take to become visible to Vespa. The feed is
# acknowledged before the document is searchable, and the lag grows with what
# else is writing to the shared agent_memories_test_unit schema.
_STRATEGY_VISIBLE_TIMEOUT_S = 60


def _seed_strategy(
    memory_manager,
    agent_name: str,
    tenant_id: str,
    text: str,
    applies_when: str = "testing strategy filter",
) -> None:
    """Store one strategy entry tagged for agent_name via real StrategyLearner.

    Returns once the entry is listable for its namespace, so a retrieval
    assertion that follows cannot race the write.
    """
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

    namespace = f"{STRATEGY_AGENT_NAME}_{agent_name}"
    stored = _poll(
        lambda: memory_manager.get_all_memories(
            tenant_id=tenant_id,
            agent_name=namespace,
        ),
        lambda rows: any(text in row.get("memory", "") for row in rows),
    )
    assert any(text in row.get("memory", "") for row in stored), (
        f"strategy {text!r} never became listable under {tenant_id}/{namespace} "
        f"within {_STRATEGY_VISIBLE_TIMEOUT_S}s; namespace holds {stored!r}"
    )


def _poll(fetch: Callable[[], Any], done: Callable[[Any], bool]) -> Any:
    """Re-run ``fetch`` until ``done`` accepts its result or the deadline
    passes; returns the last result either way so callers assert on it."""
    deadline = time.monotonic() + _STRATEGY_VISIBLE_TIMEOUT_S
    result = fetch()
    while not done(result) and time.monotonic() < deadline:
        time.sleep(1)
        result = fetch()
    return result


def _retrieve_strategies(
    learner: StrategyLearner,
    *,
    query: str,
    agent_name: str,
    contains: str,
    top_k: int = 5,
) -> List[dict]:
    """Retrieve strategies for an agent, polling until ``contains`` shows up."""
    return _poll(
        lambda: learner.get_strategies_for_agent(
            query=query,
            agent_name=agent_name,
            top_k=top_k,
        ),
        lambda rows: any(contains in row.get("memory", "") for row in rows),
    )


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
        # Retrieve it as search_agent first: without that positive control an
        # empty coding_agent result proves nothing, since a not-yet-searchable
        # write is also empty.
        search_results = _retrieve_strategies(
            learner,
            query="how to search for videos",
            agent_name="search_agent",
            contains=search_text,
        )
        assert [
            r.get("memory", "") for r in search_results if search_text in r["memory"]
        ], (
            f"search_agent cannot retrieve its own strategy, so the coding_agent "
            f"assertion below would be vacuous. Got: {search_results}"
        )

        coding_results = learner.get_strategies_for_agent(
            query="how to search for code snippets",
            agent_name="coding_agent",
        )
        texts = [r.get("memory", "") for r in coding_results]
        assert not any("ColBERT" in t or "colbert" in t.lower() for t in texts), (
            f"search_agent strategy leaked to coding_agent — the per-agent "
            f"filter has regressed. Returned: {texts}"
        )

    def test_search_strategy_visible_to_search_agent(self, memory_manager):
        """Strategy seeded for search_agent MUST appear in search_agent results.

        Fix #6 should not over-filter: agents must still see their own strategies."""
        tenant_id = f"filter_test_pos_{int(time.time() * 1000)}"
        search_text = "use ColBERT reranking when video relevance is ambiguous"

        _seed_strategy(memory_manager, "search_agent", tenant_id, search_text)

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )
        results = _retrieve_strategies(
            learner,
            query="how to search for videos",
            agent_name="search_agent",
            contains=search_text,
        )
        assert [r["memory"] for r in results if search_text in r["memory"]], (
            f"search_agent strategy is missing from search_agent retrieval — "
            f"the per-agent filter is too aggressive, blocking the agent's own "
            f"strategies. Got: {results}"
        )

    def test_cross_agent_isolation_both_directions(self, memory_manager):
        """Two distinct strategies for two agents — each sees only its own.

        Seeding search_agent + gateway_agent strategies into the same tenant,
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
            "gateway_agent",
            tenant_id,
            routing_text,
            applies_when="query routing",
        )

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id=tenant_id,
        )

        # Each agent must first retrieve its OWN strategy — otherwise the
        # no-leak assertions pass on two empty result sets.
        search_results = _retrieve_strategies(
            learner,
            query="search for videos about machine learning",
            agent_name="search_agent",
            contains=search_text,
            top_k=10,
        )
        routing_results = _retrieve_strategies(
            learner,
            query="route this complex multimodal query",
            agent_name="gateway_agent",
            contains=routing_text,
            top_k=10,
        )

        search_texts = [r.get("memory", "") for r in search_results]
        routing_texts = [r.get("memory", "") for r in routing_results]

        assert any(search_text in t for t in search_texts), (
            f"search_agent cannot retrieve its own strategy: {search_texts}"
        )
        assert any(routing_text in t for t in routing_texts), (
            f"gateway_agent cannot retrieve its own strategy: {routing_texts}"
        )
        assert not any("escalate to orchestration" in t for t in search_texts), (
            f"gateway_agent strategy leaked into search_agent results: {search_texts}"
        )
        assert not any("temporal proximity" in t for t in routing_texts), (
            f"search_agent strategy leaked into gateway_agent results: {routing_texts}"
        )


@pytest.mark.integration
class TestGatewayAgentStrategyInjection:
    """``MemoryAwareMixin.inject_context_into_prompt`` returns strategies from
    Mem0 when memory is initialized for gateway_agent."""

    def test_gateway_agent_receives_strategies_via_mixin(self, memory_manager):
        """inject_context_into_prompt must include strategies seeded for
        gateway_agent. This guards the mixin's strategy-injection path."""
        tenant_id = f"gateway_strat_test_{int(time.time() * 1000)}"
        routing_text = "escalate to orchestration when query spans multiple modalities"
        prompt = "route this complex multimodal query"

        _seed_strategy(
            memory_manager,
            "gateway_agent",
            tenant_id,
            routing_text,
            applies_when="multi-modal routing",
        )

        # Wire up the mixin directly — no need to construct a full agent
        # (which requires telemetry, LLM config, etc.). The mixin is the unit
        # under test here.
        proxy = _MemoryProxy()
        proxy.memory_manager = memory_manager
        proxy._memory_initialized = True
        proxy._memory_agent_name = "gateway_agent"
        # Same call production dispatch uses: sets the request-scoped tenant
        # ContextVar (which _current_memory_tenant_id prefers) plus the
        # instance attribute.
        proxy.set_tenant_for_context(tenant_id)

        enriched = _poll(
            lambda: proxy.inject_context_into_prompt(prompt, prompt),
            lambda text: routing_text in text,
        )

        assert routing_text in enriched, (
            f"gateway_agent strategy not injected into prompt — "
            f"inject_context_into_prompt returned {enriched[:300]!r}"
        )
        assert enriched.startswith(f"{prompt}\n\n")
        assert enriched.endswith(f"## Current Query:\n{prompt}")

    def test_gateway_agent_memory_init_sets_agent_name(
        self,
        memory_manager,
        vespa_instance,
        config_manager,
        schema_loader,
        shared_denseon,
    ):
        """initialize_memory() must set _memory_agent_name = 'gateway_agent'
        so that get_strategies() retrieves gateway_agent-tagged strategies only."""
        from tests.utils.llm_config import (
            get_llm_base_url,
            get_llm_model,
        )

        proxy = _MemoryProxy()
        success = proxy.initialize_memory(
            agent_name="gateway_agent",
            tenant_id="routing_init_test",
            backend_host="http://localhost",
            backend_port=vespa_instance["http_port"],
            backend_config_port=vespa_instance["config_port"],
            llm_model=get_llm_model(),
            embedding_model="lightonai/DenseOn",
            llm_base_url=get_llm_base_url(),
            embedder_base_url=shared_denseon,
            config_manager=config_manager,
            schema_loader=schema_loader,
            auto_create_schema=False,
        )
        assert success, "initialize_memory returned False for gateway_agent proxy"
        assert proxy._memory_agent_name == "gateway_agent", (
            f"_memory_agent_name is {proxy._memory_agent_name!r}, expected 'gateway_agent'. "
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
        prompt = "implement a binary search function"

        _seed_strategy(
            memory_manager,
            "coding_agent",
            tenant_id,
            coding_text,
            applies_when="code generation tasks",
        )

        proxy = _MemoryProxy()
        proxy.memory_manager = memory_manager
        proxy._memory_initialized = True
        proxy._memory_agent_name = "coding_agent"
        # Same call production dispatch uses: sets the request-scoped tenant
        # ContextVar (which _current_memory_tenant_id prefers) plus the
        # instance attribute.
        proxy.set_tenant_for_context(tenant_id)

        enriched = _poll(
            lambda: proxy.inject_context_into_prompt(prompt, prompt),
            lambda text: coding_text in text,
        )

        assert coding_text in enriched, (
            f"coding_agent strategy not injected into prompt — "
            f"inject_context_into_prompt returned {enriched[:300]!r}"
        )
        assert enriched.startswith(f"{prompt}\n\n")
        assert enriched.endswith(f"## Current Query:\n{prompt}")

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
        # Positive control first: an empty search_agent result only means
        # something once coding_agent can retrieve the strategy itself.
        coding_results = _retrieve_strategies(
            learner,
            query="write a python function",
            agent_name="coding_agent",
            contains=coding_text,
        )
        assert any(coding_text in r["memory"] for r in coding_results), (
            f"coding_agent cannot retrieve its own strategy, so the search_agent "
            f"assertion below would be vacuous. Got: {coding_results}"
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
