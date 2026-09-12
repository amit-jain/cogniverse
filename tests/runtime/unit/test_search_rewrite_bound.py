"""Every dispatched search bounds its query rewrite.

The rewrite is one LM round trip on the live path. When the serving endpoint is
cold the round trip takes as long as the endpoint takes - 119.7s was measured
against a cold Modal container - so a search that does not bound it waits that
long and the caller has no ceiling to rely on. The answer-grounding path bounds
it; these pin that every other dispatched search does too, at the same budget,
and that a rewrite which overruns degrades to the original query instead of
hanging.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import dspy
import numpy as np
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.search_agent import (
    QUERY_REWRITE_TIMED_OUT,
    SearchAgent,
    SearchAgentDeps,
)
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_SEARCH_RESERVE_S,
    GROUNDING_SEARCH_TIMEOUT_KEY,
    AgentDispatcher,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_SHIPPED_CONFIG = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)
_SHIPPED_ACTIVE_PROFILE = _SHIPPED_CONFIG["active_video_profile"]
_SHIPPED_GROUNDING_BUDGET_S = _SHIPPED_CONFIG[GROUNDING_SEARCH_TIMEOUT_KEY]

_TENANT = "acme:acme"
_QUERY = "people exercising"
_REWRITTEN = "workout routines for people exercising video"
_HIT = {"document_id": "v_001", "score": 0.9, "metadata": {"title": "Morning workout"}}

# A synthetic budget small enough to execute the overrun inside a unit test.
# The reserve is production's, so the bound under test stays derived.
_TEST_BUDGET_S = GROUNDING_SEARCH_RESERVE_S + 0.5
_EXPECTED_BOUND_S = _TEST_BUDGET_S - GROUNDING_SEARCH_RESERVE_S

# Longer than the bound by enough that a passing test cannot be the hang
# finishing early.
_HANG_S = 5.0


class _HangingLM(DummyLM):
    """An LM whose completion takes longer than the rewrite is allowed."""

    def __call__(self, *args, **kwargs):
        time.sleep(_HANG_S)
        return super().__call__(*args, **kwargs)


def _config_get(budget_s):
    def get(key, default=None):
        if key == GROUNDING_SEARCH_TIMEOUT_KEY:
            return budget_s
        if key == "active_video_profile":
            return _SHIPPED_ACTIVE_PROFILE
        return default

    return get


def _dispatcher(budget_s=_TEST_BUDGET_S):
    from cogniverse_foundation.config.manager import ConfigManager

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)

    with (
        patch("cogniverse_agents.search_agent.QueryEncoderFactory"),
        patch("cogniverse_agents.search_agent.get_backend_registry"),
    ):
        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=Mock(),
            config_manager=config_manager,
        )
    agent.query_encoder = SimpleNamespace(
        encode=lambda _query: np.zeros((2, 128), dtype=np.float32)
    )
    searched: list[str] = []

    def _by_text(query, **kwargs):
        searched.append(query)
        return [dict(_HIT)]

    agent._search_by_text = _by_text

    dispatcher = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    dispatcher._get_search_agent = lambda profile: agent

    captured: list[float | None] = []
    real_process = agent.process

    async def _record(search_input):
        captured.append(search_input.query_rewrite_timeout_s)
        return await real_process(search_input)

    agent.process = _record
    return dispatcher, captured, searched, _config_get(budget_s)


class TestEveryDispatchedSearchCarriesTheBound:
    async def test_direct_dispatch_bounds_the_rewrite_at_budget_minus_reserve(self):
        """A direct search names a bound; it does not leave the rewrite open."""
        dispatcher, captured, _, config_get = _dispatcher()

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [_EXPECTED_BOUND_S]

    async def test_the_bound_tracks_the_configured_budget(self):
        """Derived, not hardcoded: a different budget moves the bound with it."""
        budget = GROUNDING_SEARCH_RESERVE_S + 3.25
        dispatcher, captured, _, config_get = _dispatcher(budget)

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [3.25]

    async def test_an_explicit_bound_is_not_overridden(self):
        """The grounded path already computes its own; it stays as passed."""
        dispatcher, captured, _, config_get = _dispatcher()

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(
                _QUERY, _TENANT, top_k=3, query_rewrite_timeout_s=1.5
            )

        assert captured == [1.5]


class TestAnOverrunningRewriteDegradesInsteadOfHanging:
    async def test_a_hung_rewrite_returns_the_original_query_within_the_bound(self):
        """The fault contract: a cold endpoint costs the bound, not its own
        latency, and the search still runs on the query the caller sent."""
        dispatcher, _, searched, config_get = _dispatcher()

        started = time.perf_counter()
        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=_HangingLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            response = await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)
        elapsed = time.perf_counter() - started

        assert response["query_rewrite"] == {
            "enhanced_query": None,
            "degraded": QUERY_REWRITE_TIMED_OUT,
        }
        assert searched == [_QUERY]
        assert response["results"] == [_HIT]
        assert elapsed < _TEST_BUDGET_S, (
            f"the rewrite was not bounded: {elapsed:.2f}s elapsed against a "
            f"{_EXPECTED_BOUND_S}s bound and a {_HANG_S}s hang"
        )

    async def test_concurrent_searches_each_get_their_own_bound(self):
        """N first-touches share the dispatcher; one slow rewrite must not
        extend another's bound or leak its degradation."""
        dispatcher, _, searched, config_get = _dispatcher()

        started = time.perf_counter()
        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=_HangingLM([{"enhanced_query": _REWRITTEN}] * 4)),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            responses = await asyncio.gather(
                *(
                    dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)
                    for _ in range(4)
                )
            )
        elapsed = time.perf_counter() - started

        assert [r["query_rewrite"]["degraded"] for r in responses] == [
            QUERY_REWRITE_TIMED_OUT
        ] * 4
        assert searched == [_QUERY] * 4
        assert elapsed < _HANG_S, (
            f"four bounded rewrites took {elapsed:.2f}s; they serialized on one "
            f"another instead of each bounding at {_EXPECTED_BOUND_S}s"
        )
