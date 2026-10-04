"""A process's first span for a tenant keeps the serving loop running.

The first span for a tenant and project builds that project's exporter on the
calling thread, which for a served request is the event loop. Before the
worker serves, it imports the exporter stack (``preload_span_export``); the
first span then holds the loop only to build the exporter from loaded modules.
A fresh interpreter is used because this test process has imported the stack
already.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# The longest a heartbeat sleeping 5 ms may wait while the first span is
# opened on its loop: a third of the 150 ms liveness poll bound the e2e loop
# probe allows.
MAX_HEARTBEAT_GAP_S = 0.05

_PROBE = textwrap.dedent(
    """
    import asyncio, json, sys, time
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    manager = TelemetryManager(TelemetryConfig(
        enabled=True,
        otlp_endpoint="127.0.0.1:1",
        provider_config={
            "http_endpoint": "http://127.0.0.1:1",
            "grpc_endpoint": "http://127.0.0.1:1",
        },
    ))
    manager.preload_span_export()
    loaded = set(sys.modules)

    async def main():
        gaps, done = [], asyncio.Event()

        async def heartbeat():
            last = time.monotonic()
            while not done.is_set():
                await asyncio.sleep(0.005)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        async def first_span():
            await asyncio.sleep(0.05)
            with manager.span("probe", tenant_id="acme:acme"):
                pass
            await asyncio.sleep(0.05)
            done.set()

        await asyncio.gather(heartbeat(), first_span())
        return max(gaps)

    gap = asyncio.run(main())
    imported = sorted(
        name for name in set(sys.modules) - loaded
        if name.split(".")[0] == "phoenix"
        or name.startswith("opentelemetry.exporter")
    )
    print(json.dumps({"gap": gap, "imported": imported}))
    """
)


def test_the_first_span_after_the_startup_preload_keeps_the_loop_running():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])

    assert probe["imported"] == []
    assert probe["gap"] < MAX_HEARTBEAT_GAP_S, probe
