"""The envelope a dispatched search hands back, and what the gateway spreads.

The query rewrite is reported under ``query_rewrite``; the response carries no
top-level ``enhanced_query`` on either the direct-dispatch or the gateway path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import dspy
import numpy as np
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.gateway_agent import GatewayAgent as RealGatewayAgent
from cogniverse_agents.gateway_agent import GatewayOutput
from cogniverse_agents.search_agent import (
    QUERY_REWRITE_FAILED,
    SearchAgent,
    SearchAgentDeps,
)
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_SHIPPED_CONFIG = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)
_SHIPPED_ACTIVE_PROFILE = _SHIPPED_CONFIG["active_video_profile"]

_TENANT = "acme:acme"
_ORIGINAL_QUERY = "people exercising"
_REWRITTEN_QUERY = "workout routines for people exercising video"
_REWRITE_FIELDS = {
    "reasoning": "The query names an activity; name what it is filmed as.",
    "search_strategy": "hybrid",
    "enhanced_query": _REWRITTEN_QUERY,
    "confidence": "0.95",
}

_HIT = {
    "document_id": "v_001",
    "score": 0.9123,
    "metadata": {"title": "Morning workout"},
}

# Every top-level key the dispatched search envelope carries. The rewrite is
# reachable only under "query_rewrite": an enhanced query at the top level is
# the inline-routing shape the A2A response does not carry.
_SEARCH_ENVELOPE_KEYS = {
    "status",
    "agent",
    "message",
    "results_count",
    "results",
    "profile",
    "profiles",
    "degraded_profiles",
    "query_rewrite",
    "search_mode",
}

# What the gateway adds on top of the downstream agent's own envelope.
_GATEWAY_ENVELOPE_KEYS = _SEARCH_ENVELOPE_KEYS | {"downstream_result", "gateway"}


def _config_manager():
    from cogniverse_foundation.config.manager import ConfigManager

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def _search_agent(config_manager, hits):
    """A real SearchAgent whose text retrieval answers ``hits``.

    Only the backend round trip is stood in for; the rewrite the envelope
    reports runs for real, through whichever LM is bound.
    """
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
    agent._search_by_text = lambda query, **kwargs: [dict(hit) for hit in hits]
    return agent


def _dispatcher(hits):
    config_manager = _config_manager()
    agent = _search_agent(config_manager, hits)
    dispatcher = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    dispatcher._get_search_agent = lambda profile: agent
    return dispatcher, agent


class TestTheSearchEnvelopeReportsTheRewriteUnderItsOwnKey:
    """``_execute_search_task``'s payload: fixed keys, rewrite nested."""

    async def test_a_rewritten_query_is_reported_under_query_rewrite(self):
        dispatcher, _ = _dispatcher([_HIT])

        with dspy.context(lm=DummyLM([dict(_REWRITE_FIELDS)])):
            response = await dispatcher._execute_search_task(
                _ORIGINAL_QUERY, _TENANT, top_k=3
            )

        assert set(response) == _SEARCH_ENVELOPE_KEYS
        assert response["query_rewrite"] == {
            "enhanced_query": _REWRITTEN_QUERY,
            "degraded": None,
        }
        assert response["status"] == "success"
        assert response["agent"] == "search_agent"
        assert response["message"] == f"Found 1 results for '{_REWRITTEN_QUERY}'"
        assert response["results"] == [_HIT]
        assert response["results_count"] == 1
        assert response["profile"] == _SHIPPED_ACTIVE_PROFILE
        assert response["profiles"] == []
        assert response["degraded_profiles"] == []
        assert response["search_mode"] == "single_profile"

    async def test_a_rewrite_with_no_lm_to_call_names_itself_in_the_same_block(self):
        """The rewrite degrades in place: the search still runs and reports."""
        dispatcher, _ = _dispatcher([_HIT])

        with dspy.context(lm=None):
            response = await dispatcher._execute_search_task(
                _ORIGINAL_QUERY, _TENANT, top_k=3
            )

        assert set(response) == _SEARCH_ENVELOPE_KEYS
        assert response["query_rewrite"] == {
            "enhanced_query": None,
            "degraded": QUERY_REWRITE_FAILED,
        }
        assert response["message"] == f"Found 1 results for '{_ORIGINAL_QUERY}'"
        assert response["results"] == [_HIT]

    async def test_a_rewrite_the_caller_supplied_is_reported_in_the_same_place(self):
        """An orchestrator-made rewrite reaches the same key, not the top."""
        dispatcher, _ = _dispatcher([])

        response = await dispatcher._execute_search_task(
            _ORIGINAL_QUERY,
            _TENANT,
            top_k=3,
            enrichment={"enhanced_query": _REWRITTEN_QUERY},
        )

        assert set(response) == _SEARCH_ENVELOPE_KEYS
        assert response["query_rewrite"] == {
            "enhanced_query": _REWRITTEN_QUERY,
            "degraded": None,
        }
        assert response["message"] == f"No results found for '{_REWRITTEN_QUERY}'"
        assert response["results_count"] == 0


class TestTheGatewayResponseDoesNotSpreadTheRewrite:
    """The gateway surfaces the downstream answer; the rewrite stays nested."""

    @staticmethod
    def _routed_gateway(dispatcher):
        routed = GatewayOutput(
            query=_ORIGINAL_QUERY,
            complexity="simple",
            routed_to="search_agent",
            modality="video",
            generation_type="raw_results",
            confidence=0.9,
            fast_path_confidence_threshold=0.4,
            gliner_threshold=0.3,
            reasoning="keyword route",
        )

        class _GW(RealGatewayAgent):
            def __init__(self):
                self.telemetry_manager = None
                self._input_rails = None
                self._output_rails = None

            async def _process_impl(self, _input):
                return routed

        async def _build(_tenant_id):
            return _GW()

        dispatcher._get_or_build_gateway_agent = _build
        dispatcher._registry.get_agent.return_value = SimpleNamespace(
            capabilities=["search"]
        )
        return routed

    async def test_the_gateway_envelope_carries_the_rewrite_only_when_nested(self):
        dispatcher, _ = _dispatcher([_HIT])
        routed = self._routed_gateway(dispatcher)

        with dspy.context(lm=DummyLM([dict(_REWRITE_FIELDS)])):
            final = await dispatcher._execute_gateway_task(
                _ORIGINAL_QUERY, {"tenant_id": _TENANT}, _TENANT, top_k=3
            )

        assert set(final) == _GATEWAY_ENVELOPE_KEYS
        assert final["query_rewrite"] == {
            "enhanced_query": _REWRITTEN_QUERY,
            "degraded": None,
        }
        assert final["agent"] == "gateway_agent"
        assert final["downstream_result"]["agent"] == "search_agent"
        assert final["downstream_result"]["query_rewrite"] == final["query_rewrite"]
        assert final["gateway"] == {
            "complexity": "simple",
            "modality": "video",
            "generation_type": "raw_results",
            "routed_to": "search_agent",
            "confidence": routed.confidence,
            "fast_path_confidence_threshold": routed.fast_path_confidence_threshold,
            "gliner_threshold": routed.gliner_threshold,
        }
        assert final["results"] == [_HIT]
        assert final["message"] == f"Found 1 results for '{_REWRITTEN_QUERY}'"
