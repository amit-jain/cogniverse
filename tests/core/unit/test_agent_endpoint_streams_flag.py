"""An agent's token-streaming declaration survives registration.

``AgentEndpoint.streams_answer_tokens`` tells a streaming consumer whether the
agent emits real per-token output; it is declared in the agent's config entry,
never inferred from capabilities, so it has to travel the registration payload
-> registry -> endpoint path intact.
"""

from __future__ import annotations

import pytest

from cogniverse_core.common.agent_models import (
    DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
    AgentEndpoint,
)
from cogniverse_core.registries.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _registry() -> AgentRegistry:
    return AgentRegistry(tenant_id="acme:unit", config_manager=object())


def test_declared_streaming_reaches_the_endpoint():
    registry = _registry()

    registered = registry.register_agent_from_data(
        {
            "name": "summarizer_agent",
            "url": "http://localhost:8004",
            "capabilities": ["summarization", "text_generation"],
            "streams_answer_tokens": True,
        }
    )

    assert registered is True
    assert registry.get_agent("summarizer_agent").streams_answer_tokens is True


def test_absent_declaration_defaults_to_not_streaming():
    registry = _registry()

    registry.register_agent_from_data(
        {
            "name": "search_agent",
            "url": "http://localhost:8002",
            "capabilities": ["video_search"],
        }
    )

    assert registry.get_agent("search_agent").streams_answer_tokens is False


def test_explicit_false_reaches_the_endpoint():
    registry = _registry()

    registry.register_agent_from_data(
        {
            "name": "search_agent",
            "url": "http://localhost:8002",
            "capabilities": ["video_search"],
            "streams_answer_tokens": False,
        }
    )

    assert registry.get_agent("search_agent").streams_answer_tokens is False


def test_registered_endpoint_equals_the_whole_expected_dataclass():
    """The flag is added without disturbing any other endpoint field."""
    registry = _registry()

    registry.register_agent_from_data(
        {
            "name": "summarizer_agent",
            "url": "http://localhost:8004",
            "capabilities": ["summarization", "text_generation"],
            "streams_answer_tokens": True,
            "health_endpoint": "/health",
            "process_endpoint": "/agents/summarizer_agent/process",
            "timeout": DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
        }
    )

    assert registry.get_agent("summarizer_agent") == AgentEndpoint(
        name="summarizer_agent",
        url="http://localhost:8004",
        capabilities=["summarization", "text_generation"],
        streams_answer_tokens=True,
        health_endpoint="/health",
        process_endpoint="/agents/summarizer_agent/process",
        timeout=DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
        last_health_check=None,
        health_status="unknown",
        health_check_interval=60,
    )


def test_default_endpoint_does_not_claim_token_streaming():
    endpoint = AgentEndpoint(
        name="search_agent", url="http://localhost:8002", capabilities=[]
    )

    assert endpoint.streams_answer_tokens is False
