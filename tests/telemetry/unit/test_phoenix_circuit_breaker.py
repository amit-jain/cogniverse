"""A down Phoenix must trip the telemetry breaker: the dashboard degrades, the
provider surfaces — neither keeps dialing a dead Phoenix each call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_core.common.utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
)


@pytest.fixture(autouse=True)
def _reset():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def _breaker(name):
    return CircuitBreaker.get(
        BreakerConfig(name=name, failure_threshold=2, reset_timeout_s=999)
    )


def test_analytics_degrades_fast_when_phoenix_breaker_open():
    from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.telemetry_url = "http://phoenix:6006"
    analytics._cache = {}
    analytics.client = MagicMock()
    analytics.client.spans.get_spans_dataframe = MagicMock(
        side_effect=ConnectionError("phoenix down")
    )
    analytics._breaker = _breaker("phoenix:analytics")

    # First two calls dial and degrade to [].
    for _ in range(2):
        assert analytics.get_traces() == []
    assert analytics.client.spans.get_spans_dataframe.call_count == 2

    # Third: breaker open -> degrade WITHOUT dialing.
    assert analytics.get_traces() == []
    assert analytics.client.spans.get_spans_dataframe.call_count == 2


@pytest.mark.asyncio
async def test_provider_get_spans_raises_fast_when_breaker_open():
    from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

    store = PhoenixTraceStore.__new__(PhoenixTraceStore)
    store.http_endpoint = "http://phoenix:6006"
    store._breaker = _breaker("phoenix:provider")

    client = MagicMock()
    client.spans.get_spans_dataframe = AsyncMock(
        side_effect=ConnectionError("phoenix down")
    )
    store._get_client = lambda: client

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await store.get_spans(project="p")
    assert client.spans.get_spans_dataframe.call_count == 2

    # Load-bearing callers (checkpoint) see the open circuit as a raise.
    with pytest.raises(CircuitOpenError):
        await store.get_spans(project="p")
    assert client.spans.get_spans_dataframe.call_count == 2
