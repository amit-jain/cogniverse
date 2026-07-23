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


def test_analytics_raises_on_outage_and_fails_fast_when_breaker_open():
    """A Phoenix outage must RAISE from get_traces — returning [] reads as
    "no traces in range" on the dashboard, indistinguishable from genuine
    empty data. The breaker still bounds dialing: once open, the call fails
    fast without touching the client."""
    from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.telemetry_url = "http://phoenix:6006"
    analytics._cache = {}
    analytics.client = MagicMock()
    analytics.client.spans.get_spans_dataframe = MagicMock(
        side_effect=ConnectionError("phoenix down")
    )
    analytics._breaker = _breaker("phoenix:analytics")

    # First two calls dial and raise the transport error.
    for _ in range(2):
        with pytest.raises(ConnectionError, match="phoenix down"):
            analytics.get_traces()
    assert analytics.client.spans.get_spans_dataframe.call_count == 2

    # Third: breaker open -> raises CircuitOpenError WITHOUT dialing.
    with pytest.raises(CircuitOpenError):
        analytics.get_traces()
    assert analytics.client.spans.get_spans_dataframe.call_count == 2


def test_analytics_raises_on_non_transport_error():
    """A code/parse bug inside the fetch must never be flattened to [] —
    that hides real defects as 'no data'."""
    from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.telemetry_url = "http://phoenix:6006"
    analytics._cache = {}
    analytics.client = MagicMock()
    analytics.client.spans.get_spans_dataframe = MagicMock(
        side_effect=KeyError("unexpected frame shape")
    )
    analytics._breaker = _breaker("phoenix:analytics-nontransport")

    with pytest.raises(KeyError, match="unexpected frame shape"):
        analytics.get_traces()


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
