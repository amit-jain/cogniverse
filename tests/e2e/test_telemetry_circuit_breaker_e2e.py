"""The telemetry breaker in front of the deployed Phoenix.

Pins that the shipped breaker gives its recovery slot back on every exit of a
trial, so a probe cancelled at its deadline leaves the breaker OPEN with no
reservation held and the next reset window admits a real dial; and that a
counted failure admitted before the trip still counts after a lucky trial
closed the breaker, so a recovery probe cannot erase an outage's in-flight
failures.

The breaker under test is the one the production ``PhoenixTraceStore`` builds
for the cluster's Phoenix endpoint, and the recovery dial is a real read of the
deployed Phoenix.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_core.common.utils.circuit_breaker import CircuitOpenError, CircuitState
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.e2e.conftest import PHOENIX_URL, TENANT_ID, run_async

PHOENIX_GRPC = "localhost:33317"
# The dashboard cancels its recovery probe at this deadline; the value itself is
# the dashboard's, and what matters here is that a cancellation concludes the
# trial whatever the deadline was.
PROBE_ROUNDS = 3


def _trace_store():
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": TENANT_ID,
            "http_endpoint": PHOENIX_URL,
            "grpc_endpoint": PHOENIX_GRPC,
        }
    )
    return provider.traces


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def phoenix_breaker():
    """The shipped Phoenix breaker on a clock the test advances."""
    store = _trace_store()
    breaker = store._breaker
    real_clock = breaker.config.clock
    clock = _FakeClock()
    breaker.config.clock = clock
    yield store, breaker, clock
    breaker.config.clock = real_clock
    # Leave the shared registry entry closed for whatever runs next.
    breaker._state = CircuitState.CLOSED
    breaker._failures.clear()
    breaker._half_open_calls = 0


def _trip(breaker, clock) -> None:
    for _ in range(breaker.config.failure_threshold):
        with pytest.raises(ConnectionError):
            breaker.call(_raise_connection_error)
    assert breaker._state is CircuitState.OPEN
    clock.now += breaker.config.reset_timeout_s


def _raise_connection_error():
    raise ConnectionError("phoenix read failed")


@pytest.mark.e2e
class TestTelemetryBreakerRecoverySlot:
    """A cancelled trial releases its slot and reopens for another window."""

    def test_a_cancelled_probe_leaves_no_reservation_and_reopens(self, phoenix_breaker):
        store, breaker, clock = phoenix_breaker
        dials: list[float] = []

        async def _probe() -> None:
            dials.append(clock.now)
            await asyncio.sleep(3600)

        async def _scenario() -> None:
            for _ in range(PROBE_ROUNDS):
                assert breaker.state is CircuitState.HALF_OPEN
                task = asyncio.create_task(breaker.acall(_probe))
                while not dials or dials[-1] != clock.now:
                    await asyncio.sleep(0.01)
                # The single shipped slot is taken, so a concurrent caller is
                # refused without reaching Phoenix.
                with pytest.raises(CircuitOpenError) as rejected:
                    await breaker.acall(_probe)
                assert rejected.value.name == f"phoenix:{PHOENIX_URL}"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert breaker._state is CircuitState.OPEN
                assert breaker._half_open_calls == 0
                with pytest.raises(CircuitOpenError):
                    await breaker.acall(_probe)
                clock.now += breaker.config.reset_timeout_s

        _trip(breaker, clock)
        run_async(_scenario())

        # Exactly one dial per reset window, none free: a leaked reservation
        # would have refused every window after the first.
        assert dials == [
            breaker.config.reset_timeout_s * (round_index + 1)
            for round_index in range(PROBE_ROUNDS)
        ]
        assert breaker._half_open_calls == 0

        # The window after the last cancellation admits a real read of the
        # deployed Phoenix, and the breaker closes on it.
        assert breaker.state is CircuitState.HALF_OPEN
        now = datetime.now(timezone.utc)
        frame = run_async(
            store.get_spans(
                project=f"cogniverse-{TENANT_ID}",
                start_time=now - timedelta(minutes=1),
                end_time=now,
                limit=1,
            )
        )
        assert list(frame.index.names) == ["context.span_id"]
        assert breaker._state is CircuitState.CLOSED
        assert breaker._half_open_calls == 0

    def test_failures_admitted_before_the_trip_survive_a_lucky_probe(
        self, phoenix_breaker
    ):
        _store, breaker, clock = phoenix_breaker
        threshold = breaker.config.failure_threshold
        entered = [threading.Event() for _ in range(threshold)]
        release = threading.Event()

        def _stale(index: int):
            def _call():
                entered[index].set()
                release.wait(30)
                raise ConnectionError("phoenix read failed")

            return _call

        with ThreadPoolExecutor(max_workers=threshold) as pool:
            inflight = [
                pool.submit(breaker.call, _stale(index)) for index in range(threshold)
            ]
            for event in entered:
                assert event.wait(30) is True
            assert breaker._state is CircuitState.CLOSED

            _trip(breaker, clock)
            assert breaker.call(lambda: "lucky probe") == "lucky probe"
            assert breaker._state is CircuitState.CLOSED
            assert breaker._half_open_calls == 0

            release.set()
            for future in inflight:
                with pytest.raises(ConnectionError):
                    future.result(timeout=30)

        # The dependency's verdict counts whoever admitted the call.
        assert breaker._state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as rejected:
            breaker.call(lambda: "must not dial")
        assert rejected.value.name == f"phoenix:{PHOENIX_URL}"
