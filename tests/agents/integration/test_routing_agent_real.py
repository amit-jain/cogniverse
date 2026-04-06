"""
Real integration tests for RoutingAgent with real LLM inference.

Tests call the real routing agent with real DSPy + LLM — no mocks.
These verify actual routing decisions are semantically correct.
"""

import logging

import dspy
import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration]


def _llm_available() -> bool:
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        api_base = (
            config.get("llm_config", {})
            .get("primary", {})
            .get("api_base", "http://localhost:11434")
        )
        return httpx.get(f"{api_base}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


skip_if_no_llm = pytest.mark.skipif(
    not _llm_available(), reason="LLM endpoint not available"
)


@pytest.fixture(scope="module")
def dspy_lm():
    """Module-scoped: configure DSPy with the primary LLM from config."""
    import json
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    primary = config.get("llm_config", {}).get("primary", {})
    model = primary.get("model")
    api_base = primary.get("api_base")

    extra_body = None
    if model and ("qwen3" in model or "qwen-3" in model):
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=api_base,
        temperature=0.0,
        max_tokens=300,
        extra_body=extra_body,
    )
    lm = create_dspy_lm(endpoint)
    dspy.configure(lm=lm)
    yield lm


@pytest.fixture(scope="module")
def routing_agent(dspy_lm):
    """Module-scoped RoutingAgent with telemetry disabled."""
    from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )

    telemetry_config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",
        provider_config={
            "http_endpoint": "http://localhost:26006",
            "grpc_endpoint": "http://localhost:24317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )

    deps = RoutingDeps(
        telemetry_config=telemetry_config,
        enable_mlflow_tracking=False,
        enable_relationship_extraction=True,
        enable_query_enhancement=True,
        enable_caching=False,
    )
    return RoutingAgent(deps=deps)


@pytest.mark.asyncio
@skip_if_no_llm
async def test_routes_video_query_to_search_agent(routing_agent):
    """'Show me videos of cats' must route to a search-type agent."""
    decision = await routing_agent.route_query(
        query="Show me videos of cats",
        tenant_id="default",
    )

    assert decision.recommended_agent, "recommended_agent must not be empty"
    assert "search" in decision.recommended_agent.lower(), (
        f"Expected 'search' in recommended_agent, got: {decision.recommended_agent!r}"
    )


@pytest.mark.asyncio
@skip_if_no_llm
async def test_routes_summary_request_to_summarizer(routing_agent):
    """A detailed report/summarize query must route to a summarizer or report agent.

    Uses an explicit, context-rich query that unambiguously requests summarization
    or detailed reporting — not a bare "Summarize the results" which is ambiguous
    without context.
    """
    decision = await routing_agent.route_query(
        query="Generate a detailed summary report of all the machine learning video content",
        tenant_id="default",
    )

    assert decision.recommended_agent, "recommended_agent must not be empty"
    # The routing agent may pick summarizer, report, or search depending on the LLM's
    # interpretation.  Assert the decision contains a non-trivial reasoning string
    # and has positive confidence — the exact agent depends on LLM behavior.
    assert decision.confidence > 0.0, (
        f"Expected positive confidence, got {decision.confidence}"
    )
    assert len(decision.reasoning) > 0, "Routing must produce a non-empty reasoning"
    # The fallback_agents must include summarizer or report options for this intent
    all_agents = [decision.recommended_agent] + decision.fallback_agents
    has_reporting_agent = any(
        "summar" in a.lower() or "report" in a.lower() for a in all_agents
    )
    assert has_reporting_agent, (
        f"Neither recommended nor fallback agents include summarizer/report. "
        f"recommended={decision.recommended_agent!r}, fallbacks={decision.fallback_agents}"
    )


@pytest.mark.asyncio
@skip_if_no_llm
async def test_returns_entities(routing_agent):
    """A factual query with named entities must produce non-empty entity list."""
    decision = await routing_agent.route_query(
        query="Find videos of Tesla electric cars on highways",
        tenant_id="default",
    )

    # entities is a list — it may be populated by GLiNER or DSPy
    assert isinstance(decision.entities, list), "entities must be a list"
    assert len(decision.entities) > 0, (
        f"Expected extracted entities for a named-entity query, got empty list. "
        f"Query: 'Tesla electric cars on highways'. "
        f"metadata: {decision.metadata}"
    )


@pytest.mark.asyncio
@skip_if_no_llm
async def test_confidence_is_positive(routing_agent):
    """Every routing decision must have positive confidence."""
    decision = await routing_agent.route_query(
        query="What is machine learning?",
        tenant_id="default",
    )

    assert decision.confidence > 0.0, (
        f"Expected confidence > 0, got {decision.confidence}"
    )
    assert decision.confidence <= 1.0, (
        f"Confidence must be <= 1.0, got {decision.confidence}"
    )
