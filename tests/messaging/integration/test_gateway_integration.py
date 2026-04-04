"""
Integration tests for messaging gateway with real services.

Tests the full flow: mock Telegram updates → gateway handler → real runtime API.
Real Mem0 for conversation history. Real invite token flow.

Requires: Runtime at localhost:28000, Ollama at localhost:11434.
"""

import logging

import httpx
import pytest
from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.runtime_client import RuntimeClient
from cogniverse_messaging.telegram_handler import format_agent_response

logger = logging.getLogger(__name__)

RUNTIME_URL = "http://localhost:28000"


def _runtime_available() -> bool:
    try:
        return httpx.get(f"{RUNTIME_URL}/health", timeout=5.0).status_code == 200
    except Exception:
        return False


skip_if_no_runtime = pytest.mark.skipif(
    not _runtime_available(),
    reason="Runtime not available at localhost:28000",
)


@pytest.mark.integration
@skip_if_no_runtime
class TestRuntimeClientIntegration:
    """Test RuntimeClient against real runtime API."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        client = RuntimeClient(RUNTIME_URL)
        try:
            result = await client.health()
            assert result is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_dispatch_routing_agent(self):
        """Dispatch a query to routing_agent via real runtime."""
        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name="routing_agent",
                query="Show me videos about machine learning",
                tenant_id="flywheel_org:production",
                context_id="test_chat_123",
                top_k=3,
            )
            assert response.get("status") != "error", (
                f"Agent dispatch failed: {response}"
            )
            assert "message" in response or "results" in response
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_dispatch_search_agent(self):
        """Dispatch a search query via real runtime."""
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
        finally:
            await client.close()


@pytest.mark.integration
@skip_if_no_runtime
class TestInviteTokenIntegration:
    """Test invite token creation via real admin API."""

    @pytest.mark.asyncio
    async def test_create_invite_token_via_api(self):
        """Create invite token through the admin endpoint."""
        client = RuntimeClient(RUNTIME_URL)
        try:
            token = await client.create_invite_token(
                tenant_id="flywheel_org:production",
                expires_in_hours=1,
            )
            assert token is not None
            assert len(token) == 32  # uuid4 hex
        finally:
            await client.close()


@pytest.mark.integration
@skip_if_no_runtime
class TestMessageHandlingIntegration:
    """Test message handling flow with real runtime."""

    @pytest.mark.asyncio
    async def test_full_message_flow(self):
        """Parse command → dispatch to runtime → format response."""
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

            chunks = format_agent_response(response)
            assert len(chunks) >= 1
            assert all(len(c) <= 4096 for c in chunks)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_plain_text_routes_through_routing_agent(self):
        """Plain text → routing_agent → response formatted."""
        parsed = parse_message(text="What videos do you have about cooking?")
        assert parsed.agent_name == "routing_agent"

        client = RuntimeClient(RUNTIME_URL)
        try:
            response = await client.dispatch_agent(
                agent_name=parsed.agent_name,
                query=parsed.query,
                tenant_id="flywheel_org:production",
                context_id="integration_test_chat_2",
                top_k=3,
            )

            chunks = format_agent_response(response)
            assert len(chunks) >= 1
        finally:
            await client.close()
