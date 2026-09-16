"""Circuit breaker state machine, driven by a fake clock (no real sleeps)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

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


def _raise(exc: BaseException):
    def call():
        raise exc

    return call


def test_stays_closed_below_threshold():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=3))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            br.call(_fail)
    assert br.state is CircuitState.CLOSED


@pytest.mark.unit
@pytest.mark.ci_fast
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


@pytest.mark.unit
@pytest.mark.ci_fast
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


@pytest.mark.unit
@pytest.mark.ci_fast
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


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_uncounted_half_open_exit_releases_its_slot_and_reopens(error_type):
    clock = _Clock()
    br = CircuitBreaker(
        _cfg(clock, failure_threshold=1, counted_exceptions=(ConnectionError,))
    )
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(30)

    def abort():
        raise error_type("probe interrupted")

    with pytest.raises(error_type, match="probe interrupted"):
        br.call(abort)
    assert br._half_open_calls == 0
    assert br.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError) as rejected:
        br.call(lambda: "second free dial")
    assert rejected.value.name == "dep"
    clock.advance(30)
    assert br.state is CircuitState.HALF_OPEN
    assert br.call(lambda: "recovered") == "recovered"
    assert br.state is CircuitState.CLOSED
    assert br._half_open_calls == 0


def test_uncounted_closed_exit_leaves_the_breaker_closed():
    clock = _Clock()
    br = CircuitBreaker(
        _cfg(clock, failure_threshold=1, counted_exceptions=(ConnectionError,))
    )
    with pytest.raises(ValueError):
        br.call(_raise(ValueError("client error")))
    assert br.state is CircuitState.CLOSED
    assert br.call(lambda: "served") == "served"


@pytest.mark.asyncio
async def test_cancelled_probe_reopens_for_another_reset_window():
    """A chronically slow dependency fails fast again: the probe cancelled at
    its deadline concludes the trial, so the next caller is rejected rather
    than paying a second full deadline."""
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, reset_timeout_s=30.0))
    dials = []

    async def slow():
        dials.append(clock.t)
        await asyncio.Event().wait()

    with pytest.raises(ConnectionError):
        br.call(_fail)
    for round_index in range(3):
        clock.advance(30)
        assert br.state is CircuitState.HALF_OPEN
        probe = asyncio.create_task(br.acall(slow))
        await asyncio.wait_for(asyncio.sleep(0), 2)
        while len(dials) <= round_index:
            await asyncio.sleep(0)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        assert br.state is CircuitState.OPEN
        assert br._half_open_calls == 0
        with pytest.raises(CircuitOpenError):
            await br.acall(slow)
    assert dials == [30.0, 60.0, 90.0]


@pytest.mark.asyncio
async def test_cancelled_probe_ends_the_trial_for_its_concurrent_peer():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, half_open_max_calls=2))
    with pytest.raises(ConnectionError):
        br.call(_fail)
    clock.advance(30)
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    dialled = []

    async def probe(index):
        dialled.append(index)
        entered[index].set()
        await release.wait()
        return index

    tasks = [asyncio.create_task(br.acall(probe, i)) for i in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 2)
        assert dialled == [0, 1]
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert br._half_open_calls == 0
        assert br.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as rejected:
            await br.acall(probe, 2)
        assert rejected.value.name == "dep"
        assert dialled == [0, 1]
        release.set()
        assert await tasks[1] == 1
        assert br.state is CircuitState.OPEN
        assert br._half_open_calls == 0
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


def _stale_call(br, body):
    """Run ``body`` through the breaker on another thread, admitted now and
    completing only when the returned release event is set."""
    entered = threading.Event()
    release = threading.Event()

    def call():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release the in-flight call")
        return body()

    return entered, release, call


def test_stale_sync_success_cannot_close_a_new_recovery():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1))
    entered, release, call = _stale_call(br, lambda: "old success")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(br.call, call)
        try:
            assert entered.wait(2) is True
            with pytest.raises(ConnectionError):
                br.call(_fail)
            clock.advance(30)
            assert br.state is CircuitState.HALF_OPEN
            release.set()
            assert future.result(timeout=2) == "old success"
            assert br.state is CircuitState.HALF_OPEN
            assert br._half_open_calls == 0
            assert br.call(lambda: "new recovery") == "new recovery"
            assert br.state is CircuitState.CLOSED
        finally:
            release.set()


def test_stale_sync_failure_still_reopens_the_breaker():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1))
    entered, release, call = _stale_call(br, _fail)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(br.call, call)
        try:
            assert entered.wait(2) is True
            with pytest.raises(ConnectionError):
                br.call(_fail)
            clock.advance(30)
            assert br.state is CircuitState.HALF_OPEN
            release.set()
            with pytest.raises(ConnectionError, match="down"):
                future.result(timeout=2)
            assert br.state is CircuitState.OPEN
            assert br._half_open_calls == 0
            with pytest.raises(CircuitOpenError):
                br.call(lambda: "not admitted")
            clock.advance(30)
            assert br.call(lambda: "new recovery") == "new recovery"
            assert br.state is CircuitState.CLOSED
        finally:
            release.set()


def test_failures_admitted_before_a_trip_survive_a_lucky_recovery_probe():
    """Calls admitted while CLOSED fail after the breaker has tripped and
    reclosed on one lucky probe. Their failures are the dependency's, not the
    probe's, so they refill the window and the breaker reopens."""
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=3, reset_timeout_s=30.0))
    in_flight = [_stale_call(br, _fail) for _ in range(3)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(br.call, call) for _, _, call in in_flight]
        try:
            for entered, _, _ in in_flight:
                assert entered.wait(2) is True
            for _ in range(3):
                with pytest.raises(ConnectionError):
                    br.call(_fail)
            assert br.state is CircuitState.OPEN
            clock.advance(30)
            assert br.call(lambda: "lucky probe") == "lucky probe"
            assert br.state is CircuitState.CLOSED

            for _, release, _ in in_flight:
                release.set()
            for future in futures:
                with pytest.raises(ConnectionError, match="down"):
                    future.result(timeout=2)
            assert br.state is CircuitState.OPEN
            with pytest.raises(CircuitOpenError) as rejected:
                br.call(lambda: "must not dial a dead dependency")
            assert rejected.value.name == "dep"
        finally:
            for _, release, _ in in_flight:
                release.set()


def test_stale_failures_during_an_outage_do_not_extend_it():
    clock = _Clock()
    br = CircuitBreaker(_cfg(clock, failure_threshold=1, reset_timeout_s=30.0))
    entered, release, call = _stale_call(br, _fail)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(br.call, call)
        try:
            assert entered.wait(2) is True
            with pytest.raises(ConnectionError):
                br.call(_fail)
            assert br.state is CircuitState.OPEN
            clock.advance(29)
            release.set()
            with pytest.raises(ConnectionError, match="down"):
                future.result(timeout=2)
            assert br.state is CircuitState.OPEN
            clock.advance(1)
            assert br.state is CircuitState.HALF_OPEN
        finally:
            release.set()
