"""TelemetryManager first-touch init must run its body exactly once.

__new__ is double-checked under the class lock, but __init__ guarded only on an
instance flag with no lock: two threads that both pass __new__ (same singleton)
would both run the init body, and the second's fresh {} caches clobber the
providers the first populated. The init body now runs under the class lock.
"""

from __future__ import annotations

import threading
import time

import pytest

from cogniverse_foundation.telemetry.manager import TelemetryManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _CountingConfig:
    """A minimal telemetry config whose validate() counts init-body runs."""

    otlp_endpoint = "localhost:4317"

    def __init__(self, counter: dict):
        self._counter = counter

    def validate(self) -> None:
        self._counter["n"] += 1
        # Widen the window so a racing thread is inside the body if unguarded.
        time.sleep(0.05)


def test_concurrent_first_init_runs_body_once():
    TelemetryManager.reset()
    counter = {"n": 0}
    config = _CountingConfig(counter)

    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        TelemetryManager(config)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert counter["n"] == 1, (
            f"init body ran {counter['n']} times under concurrent first-touch "
            "(double-init race)"
        )
        # All threads observed the same singleton.
        assert TelemetryManager._instance is not None
        assert TelemetryManager._instance._initialized is True
    finally:
        TelemetryManager.reset()
