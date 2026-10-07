"""A full garbage collection in a serving worker keeps the loop running.

A worker's startup leaves about half a million long-lived objects. A full
(generation 2) collection scans them all and holds the interpreter; the
collection a background thread's allocation triggered stalled liveness polls
for 0.43 to 0.47 s late in a deep-research turn. Startup freezes them
(``freeze_startup_heap``), so later full collections scan only what was made
since. A fresh interpreter holds the worker's import-time heap.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.unit]

# The longest a heartbeat sleeping 5 ms may wait while another thread runs a
# full collection: a third of the 150 ms liveness poll bound the e2e loop
# probe allows.
MAX_HEARTBEAT_GAP_S = 0.05

_PROBE = textwrap.dedent(
    """
    import asyncio, gc, json, threading, time
    from cogniverse_runtime import main

    main.preload_lm_client_modules()
    main.freeze_startup_heap()
    since_startup = [{"turn": [index] * 3} for index in range(100_000)]

    async def run():
        gaps, done = [], threading.Event()

        def collect():
            time.sleep(0.05)
            gc.collect(2)
            time.sleep(0.05)
            done.set()

        threading.Thread(target=collect).start()
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - last)
            last = now
        return max(gaps)

    print(json.dumps({"gap": asyncio.run(run()), "frozen": gc.get_freeze_count()}))
    """
)


def test_a_full_collection_after_startup_keeps_the_loop_running():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])

    assert probe["frozen"] > 100_000, probe
    assert probe["gap"] < MAX_HEARTBEAT_GAP_S, probe
