"""Liveness polling that proves the serving replica kept answering.

A request whose blocking work runs on the API event loop stops the whole
uvicorn worker: ``GET /health/live`` — a constant-returning route with no
dependency — stops being answered for the duration. Polling it from a
second client while the request under test is in flight turns that into an
observation: the number of answers collected and the latency of each.

The bound a sample is measured against is the readiness probe's own
``timeoutSeconds`` from the chart the e2e cluster is deployed with, so the
threshold is the one the kubelet applies to this deployment rather than a
number restated here.
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
"""Cadence the probe polls at. Ten samples a second is fine-grained enough
that a stall of one backend round trip loses several of them."""


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


@dataclass(frozen=True)
class LoopProbeResult:
    """Every liveness poll taken while the request under test was in flight."""

    samples: Tuple[Tuple[int, float], ...]
    window_s: float

    @property
    def status_codes(self) -> List[int]:
        return [status for status, _latency in self.samples]

    @property
    def max_latency_s(self) -> float:
        return max(latency for _status, latency in self.samples)

    def expected_minimum_polls(self) -> int:
        """Polls a loop that never stalled owes for this window.

        One poll per interval, less one for the partial interval the window
        ends on and one for the request the probe itself was starting when
        the window closed.
        """
        return max(1, int(self.window_s / POLL_INTERVAL_S) - 2)


class LoopProbe:
    """Polls ``GET /health/live`` on a second connection until stopped."""

    def __init__(self, interval_s: float = POLL_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._samples: List[Tuple[int, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at = 0.0
        self._stopped_at = 0.0

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
                self._samples.append((status, time.monotonic() - started))
                self._stop.wait(self._interval_s)

    def stop(self) -> LoopProbeResult:
        if self._stopped_at == 0.0:
            self._stop.set()
            self._thread.join(timeout=readiness_timeout_s() + 5.0)
            self._stopped_at = time.monotonic()
        return LoopProbeResult(
            samples=tuple(self._samples),
            window_s=self._stopped_at - self._started_at,
        )


def assert_loop_served(result: LoopProbeResult) -> None:
    """Every poll answered 200, each within one readiness-probe budget.

    ``expected_minimum_polls`` is the discriminator: a loop held by a
    synchronous backend call answers nothing at all for that span, so the
    sample count collapses even when the samples it did take were fast.
    """
    budget_s = readiness_timeout_s()
    assert result.status_codes == [200] * len(result.samples), result.samples
    assert len(result.samples) >= result.expected_minimum_polls(), (
        f"liveness answered {len(result.samples)} times in {result.window_s:.2f}s "
        f"at a {POLL_INTERVAL_S:g}s cadence; a loop that never stalled owes at "
        f"least {result.expected_minimum_polls()}"
    )
    assert result.max_latency_s < budget_s, (
        f"slowest liveness poll took {result.max_latency_s:.2f}s, at or past the "
        f"chart's runtime.readinessProbe.timeoutSeconds of {budget_s:g}s"
    )
