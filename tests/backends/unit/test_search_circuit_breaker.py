"""A down Vespa must trip the search breaker and fail fast, not retry-storm.

Vespa is load-bearing, so once the breaker opens ``search`` raises
CircuitOpenError immediately instead of dialing and burning its retries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
)
from cogniverse_vespa.search_backend import VespaSearchBackend


@pytest.fixture(autouse=True)
def _reset():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def test_search_fast_fails_after_breaker_opens():
    backend = object.__new__(VespaSearchBackend)
    backend._search_breaker = CircuitBreaker.get(
        BreakerConfig(
            name="vespa_search:test", failure_threshold=2, reset_timeout_s=999
        )
    )
    calls = {"n": 0}

    def boom(_qd):
        calls["n"] += 1
        raise ConnectionError("vespa down")

    backend._search_retried = boom

    for _ in range(2):
        with pytest.raises(ConnectionError):
            backend.search({"query": "x"})
    assert calls["n"] == 2

    # Breaker open: the impl is not called again.
    with pytest.raises(CircuitOpenError):
        backend.search({"query": "x"})
    assert calls["n"] == 2


def test_success_keeps_breaker_closed():
    backend = object.__new__(VespaSearchBackend)
    backend._search_breaker = CircuitBreaker.get(
        BreakerConfig(name="vespa_search:test2", failure_threshold=2)
    )
    backend._search_retried = MagicMock(return_value=["result"])
    for _ in range(5):
        assert backend.search({"query": "x"}) == ["result"]
