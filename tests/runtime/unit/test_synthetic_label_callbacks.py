"""Runtime synthetic callbacks expose production agent decisions directly."""

from types import SimpleNamespace

import pytest

from cogniverse_runtime.main import (
    _dispatcher_profile_labeler,
    _dispatcher_routing_decider,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_routing_callback_returns_gateway_decision_without_downstream_dispatch():
    decision = SimpleNamespace(
        query="find Marie Curie",
        routed_to="search_agent",
        modality="video",
        complexity="simple",
    )

    class GatewayAgent:
        async def process(self, typed_input):
            assert typed_input.query == "find Marie Curie"
            assert typed_input.tenant_id == "acme:media"
            return decision

    class Dispatcher:
        async def _get_or_build_gateway_agent(self, tenant_id):
            assert tenant_id == "acme:media"
            return GatewayAgent()

        async def dispatch(self, *args, **kwargs):
            raise AssertionError(
                "routing supervision must not execute downstream agents"
            )

    result = await _dispatcher_routing_decider(Dispatcher())(
        "find Marie Curie", "acme:media"
    )

    assert result is decision
    assert result.routed_to == "search_agent"


@pytest.mark.asyncio
async def test_profile_callback_dispatches_all_candidate_profiles():
    calls = []
    decision = {
        "query": "quantum computing",
        "selected_profile": "document_semantic",
        "modality": "document",
        "complexity": "medium",
        "query_intent": "research_lookup",
        "reasoning": "The production selector chose document retrieval.",
    }

    class Dispatcher:
        async def dispatch(self, **kwargs):
            calls.append(kwargs)
            return decision

    result = await _dispatcher_profile_labeler(Dispatcher())(
        "quantum computing",
        ["audio_semantic", "document_semantic"],
        "acme:media",
    )

    assert result == decision
    assert calls == [
        {
            "agent_name": "profile_selection_agent",
            "query": "quantum computing",
            "context": {
                "tenant_id": "acme:media",
                "profiles": ["audio_semantic", "document_semantic"],
            },
        }
    ]
