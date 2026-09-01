"""
End-to-end tests for messaging gateway with real services.

Tests the full flow: mock Telegram updates → gateway handler → real runtime API.
Real Mem0 for conversation history. Real invite token flow.

Requires the deployed cluster: runtime at localhost:33000 and the configured
LM endpoint.
"""

import logging
import re

import httpx
import pytest
from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.runtime_client import RuntimeClient
from cogniverse_messaging.telegram_handler import format_agent_response

from tests.e2e.conftest import (
    assert_orchestrated,
    assert_telegram_chunks,
    expected_gateway_routing,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.e2e]

RUNTIME_URL = "http://localhost:33000"

SEARCH_RESPONSE_FIELDS = {
    "status",
    "agent",
    "message",
    "results_count",
    "results",
    "profile",
    "search_mode",
}


def _assert_search_response(response: dict, query: str) -> None:
    """The dispatcher's search_agent payload: fixed keys (plus the multi-turn
    pair when a conversation history was resolved), count == len(results), and
    the message templated on the count (the query in it is LM-enhanced)."""
    multi_turn = {"original_query", "rewritten_query"}
    assert set(response) - multi_turn == SEARCH_RESPONSE_FIELDS, response
    assert set(response) & multi_turn in (set(), multi_turn), response
    if multi_turn <= set(response):
        assert response["original_query"] == query, response
    assert response["status"] == "success", response
    assert response["agent"] == "search_agent", response
    assert response["results_count"] == len(response["results"]), response
    if response["results_count"]:
        assert re.fullmatch(
            rf"Found {response['results_count']} results for '.*'",
            response["message"],
            flags=re.DOTALL,
        ), response["message"]
    else:
        assert re.fullmatch(
            r"No results found for '.*'", response["message"], flags=re.DOTALL
        ), response["message"]


def _assert_gateway_response(response: dict, query: str) -> None:
    """The gateway triage block plus the path-specific payload shape."""
    assert response["status"] == "success", response
    gw = response["gateway"]
    assert set(gw) == {
        "complexity",
        "modality",
        "generation_type",
        "routed_to",
        "confidence",
        "fast_path_confidence_threshold",
        "gliner_threshold",
    }, gw
    assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(query, gw)
    if gw["complexity"] == "simple":
        assert response["agent"] == "gateway_agent", response
        assert response["downstream_result"]["status"] == "success", response
        assert gw["confidence"] >= gw["fast_path_confidence_threshold"], gw
    else:
        assert_orchestrated(response, query, gw)


async def _assert_runtime_ready() -> None:
    # /health/live is cheap; /health does backend + registry lookups and
    # can block under LLM load.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RUNTIME_URL}/health/live")
    except httpx.HTTPError as exc:
        raise AssertionError(
            f"Runtime liveness endpoint must be reachable at {RUNTIME_URL}"
        ) from exc
    assert response.status_code == 200, (
        f"Runtime liveness must return HTTP 200; got {response.status_code}: "
        f"{response.text}"
    )
    assert response.json() == {"status": "alive"}, response.json()


class TestRuntimeClientIntegration:
    """Test RuntimeClient against real runtime API."""

    async def test_health_check(self):
        await _assert_runtime_ready()
        client = RuntimeClient(RUNTIME_URL)
        try:
            result = await client.health()
            assert result is True
        finally:
            await client.close()

    async def test_dispatch_gateway_agent(self):
        """Dispatch a query to gateway_agent via real runtime."""
        await _assert_runtime_ready()
        query = "Show me videos about machine learning"
        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name="gateway_agent",
                query=query,
                tenant_id="flywheel_org:production",
                context_id="test_chat_123",
                top_k=3,
            )
            assert response.get("status") != "error", (
                f"Agent dispatch failed: {response}"
            )
            _assert_gateway_response(response, query)
        finally:
            await client.close()

    async def test_dispatch_search_agent(self):
        """Dispatch a search query via real runtime."""
        await _assert_runtime_ready()
        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name="search_agent",
                query="people exercising",
                tenant_id="flywheel_org:production",
                top_k=3,
            )
            assert response.get("status") != "error", (
                f"Search dispatch failed: {response}"
            )
            _assert_search_response(response, "people exercising")
            assert response["results_count"] <= 3, response
        finally:
            await client.close()


class TestInviteTokenIntegration:
    """Test invite token creation via real admin API."""

    async def test_create_invite_token_via_api(self):
        """Create invite token through the admin endpoint."""
        await _assert_runtime_ready()
        client = RuntimeClient(RUNTIME_URL)
        try:
            token = await client.create_invite_token(
                tenant_id="flywheel_org:production",
                expires_in_hours=1,
            )
            assert re.fullmatch(r"[0-9a-f]{32}", token), token  # uuid4 hex
            assert len(token) == 32  # uuid4 hex
        finally:
            await client.close()


class TestMessageHandlingIntegration:
    """Test message handling flow with real runtime."""

    async def test_full_message_flow(self):
        """Parse command → dispatch to runtime → format response."""
        await _assert_runtime_ready()
        parsed = parse_message(text="/search videos of cats playing")
        assert parsed.agent_name == "search_agent"

        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name=parsed.agent_name,
                query=parsed.query,
                tenant_id="flywheel_org:production",
                context_id="integration_test_chat",
                top_k=3,
            )

            _assert_search_response(response, parsed.query)
            assert response["results_count"] <= 3, response
            chunks = format_agent_response(response)
            assert_telegram_chunks(chunks, response)
        finally:
            await client.close()

    async def test_plain_text_routes_through_gateway_agent(self):
        """Plain text → gateway_agent → response formatted."""
        await _assert_runtime_ready()
        parsed = parse_message(text="What videos do you have about cooking?")
        assert parsed.agent_name == "gateway_agent"

        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name=parsed.agent_name,
                query=parsed.query,
                tenant_id="flywheel_org:production",
                context_id="integration_test_chat_2",
                top_k=3,
            )

            _assert_gateway_response(response, parsed.query)
            chunks = format_agent_response(response)
            assert_telegram_chunks(chunks, response)
        finally:
            await client.close()
