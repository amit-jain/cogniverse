"""Circuit breaker state machine, driven by a fake clock (no real sleeps)."""

from __future__ import annotations

import pytest

from cogniverse_core.common.utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


@pytest.fixture(autouse=True)
def _isolate_registry():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def _cfg(clock, **kw):
    kw.setdefault("name", "dep")
    kw.setdefault("failure_threshold", 3)
    kw.setdefault("window_s", 60.0)
    kw.setdefault("reset_timeout_s", 30.0)
    return BreakerConfig(clock=clock, **kw)


def _fail():
    raise ConnectionError("down")


def test_stays_closed_below_threshold():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=3))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            br.call(_fail)
    assert br.state is CircuitState.CLOSED


def test_trips_open_at_threshold_then_fails_fast():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=3))
    for _ in range(3):
        with pytest.raises(ConnectionError):
            br.call(_fail)
    assert br.state is CircuitState.OPEN
    # Now it rejects without ever calling fn.
    called = {"n": 0}

    def spy():
        called["n"] += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        br.call(spy)
    assert called["n"] == 0


def test_rolling_window_evicts_old_failures():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=3, window_s=10.0))
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(11.0)  # first failure ages out of the window
    for _ in range(2):
        with pytest.raises(ConnectionError):
            br.call(_fail)
    # Only 2 failures within the window -> still closed.
    assert br.state is CircuitState.CLOSED


def test_open_transitions_to_half_open_after_reset():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, reset_timeout_s=30.0))
    with pytest.raises(ConnectionError):
        br.call(_fail)
    assert br.state is CircuitState.OPEN
    clock.advance(30.0)
    assert br.state is CircuitState.HALF_OPEN


def test_half_open_success_closes():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, reset_timeout_s=30.0))
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(30.0)
    assert br.call(lambda: "recovered") == "recovered"
    assert br.state is CircuitState.CLOSED


def test_half_open_failure_reopens():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, reset_timeout_s=30.0))
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(30.0)
    assert br.state is CircuitState.HALF_OPEN
    with pytest.raises(ConnectionError):
        br.call(_fail)
    assert br.state is CircuitState.OPEN


def test_half_open_admits_only_max_trials():
    clock = _Clock()
    br = CircuitBreaker(
        _cfg(clock, failure_threshold=1, reset_timeout_s=30.0, half_open_max_calls=1)
    )
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(30.0)
    br._before_call()  # consume the single half-open slot
    with pytest.raises(CircuitOpenError):
        br.call(lambda: "second trial rejected")


def test_uncounted_exception_does_not_trip():
    clock = _Clock()
    br = CircuitBreaker(
        _cfg(clock, failure_threshold=1, counted_exceptions=(ConnectionError,))
    )

    def raise_value():
        raise ValueError("client error, not a dependency outage")

    with pytest.raises(ValueError):
        br.call(raise_value)
    assert br.state is CircuitState.CLOSED


def test_threshold_zero_disables_breaker():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=0))
    for _ in range(10):
        with pytest.raises(ConnectionError):
            br.call(_fail)
    assert br.state is CircuitState.CLOSED  # never trips


def test_get_shares_state_by_name():
    clock = _Clock()
    cfg = _cfg(clock, name="shared", failure_threshold=1)
    a = CircuitBreaker.get(cfg)
    b = CircuitBreaker.get(cfg)
    assert a is b
    with pytest.raises(ConnectionError):
        a.call(_fail)
    assert b.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_acall_trips_and_rejects():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1))

    async def afail():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await br.acall(afail)
    assert br.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await br.acall(afail)
