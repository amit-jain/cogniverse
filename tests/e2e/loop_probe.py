"""Liveness polling that proves the serving replica kept answering.

A request whose blocking work runs on the API event loop stops the whole
uvicorn worker: ``GET /health/live`` — a constant-returning route with no
dependency — is not answered until the loop is released. Polling it from a
second client while the request under test is in flight turns that into an
observation: a held loop shows as a slow poll.

The per-poll bound is the readiness probe's own timeout-to-period share from
the chart the e2e cluster is deployed with, applied to this probe's cadence,
so the threshold follows the kubelet's budget for this deployment rather than
a number restated here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import httpx
import yaml

from tests.e2e.conftest import K3S_VALUES, RUNTIME

BASE_VALUES = Path(__file__).resolve().parents[2] / "charts/cogniverse/values.yaml"

POLL_INTERVAL_S = 0.1
"""Pause between one poll's answer and the next poll."""


def _probe_field(field: str) -> object:
    """``runtime.readinessProbe.<field>`` as the e2e cluster renders it.

    Helm deep-merges the k3s overlay onto the base chart, so a field the
    overlay does not set is the base chart's.
    """
    base = yaml.safe_load(BASE_VALUES.read_text())["runtime"]["readinessProbe"]
    overlay = (
        yaml.safe_load(K3S_VALUES.read_text())["runtime"].get("readinessProbe") or {}
    )
    merged = {**base, **overlay}
    if field not in merged:
        raise KeyError(
            f"runtime.readinessProbe.{field} is set in neither {BASE_VALUES} "
            f"nor {K3S_VALUES}"
        )
    return merged[field]


def readiness_timeout_s() -> float:
    """Seconds the kubelet gives one readiness probe on this deployment."""
    return float(_probe_field("timeoutSeconds"))


def poll_latency_bound_s() -> float:
    """Longest one liveness poll may take on a loop nothing held.

    The kubelet allows a readiness probe ``timeoutSeconds`` for every
    ``periodSeconds`` it probes at; a poll here is allowed the same share of
    this probe's interval.
    """
    return (
        readiness_timeout_s() * POLL_INTERVAL_S / float(_probe_field("periodSeconds"))
    )


@dataclass(frozen=True)
class LoopProbeResult:
    """Every liveness poll taken while the request under test was in flight.

    Each sample is ``(status, started_s, latency_s)``: the status answered,
    when the poll started relative to the window's start, and how long the
    request took. ``polled_until_stopped`` is whether the polling thread was
    still running when the window closed and finished its last poll.
    """

    samples: Tuple[Tuple[int, float, float], ...]
    window_s: float
    polled_until_stopped: bool

    @property
    def status_codes(self) -> List[int]:
        return [status for status, _started, _latency in self.samples]

    def polls_slower_than(self, bound_s: float) -> List[Tuple[float, float]]:
        """``(started_s, latency_s)`` of every poll at or past ``bound_s``."""
        return [
            (started, latency)
            for _status, started, latency in self.samples
            if latency >= bound_s
        ]


class LoopProbe:
    """Polls ``GET /health/live`` on a second connection until stopped."""

    def __init__(self, interval_s: float = POLL_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._samples: List[Tuple[int, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._polled_until_stopped = False

    def __enter__(self) -> "LoopProbe":
        self._started_at = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def _run(self) -> None:
        # One client for the whole window: a fresh connection per poll would
        # measure TCP setup rather than the loop's readiness to answer.
        with httpx.Client(base_url=RUNTIME, timeout=readiness_timeout_s()) as client:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    status = client.get("/health/live").status_code
                except httpx.HTTPError:
                    # A refused or timed-out poll is the failure this probe
                    # exists to catch; record it as such rather than drop it.
                    status = 0
                self._samples.append(
                    (status, started - self._started_at, time.monotonic() - started)
                )
                self._stop.wait(self._interval_s)

    def stop(self) -> LoopProbeResult:
        if self._stopped_at == 0.0:
            running = self._thread.is_alive()
            self._stop.set()
            self._thread.join(timeout=readiness_timeout_s() + 5.0)
            self._stopped_at = time.monotonic()
            self._polled_until_stopped = running and not self._thread.is_alive()
        return LoopProbeResult(
            samples=tuple(self._samples),
            window_s=self._stopped_at - self._started_at,
            polled_until_stopped=self._polled_until_stopped,
        )


def assert_loop_served(result: LoopProbeResult) -> None:
    """Liveness was polled for the whole window and every poll answered 200
    within :func:`poll_latency_bound_s`.

    A loop held by a synchronous backend call answers the poll in flight only
    once it is released, so the hold is that poll's latency. Time the probe
    itself spends between polls is not the server's and is not measured.
    """
    bound_s = poll_latency_bound_s()
    assert result.polled_until_stopped, (
        f"the liveness probe stopped polling before the window closed after "
        f"{len(result.samples)} polls in {result.window_s:.2f}s"
    )
    assert result.status_codes == [200] * len(result.samples), result.samples
    slow = result.polls_slower_than(bound_s)
    assert slow == [], (
        f"liveness polls at or past {bound_s * 1000:.0f}ms, the chart's "
        f"runtime.readinessProbe timeoutSeconds/periodSeconds share of the "
        f"probe's {POLL_INTERVAL_S:g}s interval: "
        + ", ".join(
            f"started {started:.2f}s took {latency:.2f}s" for started, latency in slow
        )
    )
