"""A restarted ingestion worker exports its spans to the deployment's Phoenix.

A pod killed mid-job leaves the job in its consumer's pending list. The
replacement worker's reaper re-drives it, and on a restart that re-drive is
the first job the process runs, so it is what builds the telemetry manager.
The re-drive path passed no OTLP endpoint, the manager took the stored
config's ``localhost:4317``, and the pod's spans went nowhere.

Here a fresh worker process starts with the chart's environment
(``TELEMETRY_OTLP_ENDPOINT`` and ``TELEMETRY_HTTP_ENDPOINT`` naming a real
Phoenix) over a Redis that holds an orphaned job, and the job's span must
arrive in that Phoenix.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTAINER_NAME = f"redis-worker-telemetry-{os.getpid()}"

# Runs worker.run() as the pod's entrypoint does, with a processor that
# returns at once and stops the worker a few seconds later so the span
# exporter's batch drains.
_WORKER_SCRIPT = """
import asyncio

from cogniverse_runtime.ingestion_worker import worker


async def main():
    stop = asyncio.Event()

    async def processor(job):
        asyncio.get_running_loop().call_later(3, stop.set)
        return {}

    await worker.run(stop=stop, processor=processor)


asyncio.run(main())

from cogniverse_foundation.telemetry.manager import get_telemetry_manager

manager = get_telemetry_manager()
manager.force_flush(timeout_millis=10000)
print("OTLP_ENDPOINT", manager.config.otlp_endpoint)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def redis_url():
    port = _free_port()
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"127.0.0.1:{port}:6379",
            "redis:7.4-alpine",
        ],
        check=True,
        capture_output=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        ping = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.5)
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


async def _orphan_a_job(redis_url: str, tenant_id: str) -> str:
    """Leave one job claimed by a consumer that is gone, as an OOM kill does."""
    from cogniverse_runtime.ingestion_worker import queue
    from cogniverse_runtime.ingestion_worker.redis_client import (
        close_redis,
        get_redis,
    )

    redis = await get_redis(redis_url)
    try:
        await queue.ensure_consumer_group(redis, "ingestors")
        ingest_id = f"orphan-{uuid4().hex[:8]}"
        await queue.submit(
            redis,
            ingest_id,
            "s3://cogniverse-ingest/clip.mp4",
            "video_colpali_smol500_mv_frame",
            tenant_id,
            f"sha-{ingest_id}",
        )
        claimed = await queue.claim(redis, "ingestors", "killed-pod", block_ms=1000)
        assert [job.ingest_id for job in claimed] == [ingest_id]
        return ingest_id
    finally:
        await close_redis()


async def _job_spans(http_endpoint: str, project: str) -> list:
    from phoenix.client import AsyncClient

    client = AsyncClient(base_url=http_endpoint)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    try:
        frame = await client.spans.get_spans_dataframe(
            project_identifier=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=100,
        )
    except Exception as exc:
        if "not found" in str(exc).lower() or "404" in str(exc):
            return []
        raise
    return sorted(frame["name"]) if not frame.empty else []


def test_a_restarted_worker_exports_the_re_driven_job_to_the_chart_s_endpoint(
    shared_vespa, phoenix_container, redis_url
):
    tenant_id = canonical_tenant_id(f"wtel{uuid4().hex[:8]}")
    ingest_id = asyncio.run(_orphan_a_job(redis_url, tenant_id))

    # The chart's ingestor env (charts/cogniverse/templates/ingestor.yaml),
    # pointed at this test's services; the reaper sweeps once a second.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TELEMETRY_", "INGEST_", "REDIS_"))
    }
    env.update(
        {
            "REDIS_URL": redis_url,
            "INGEST_CONSUMER_GROUP": "ingestors",
            "INGEST_CONSUMER_ID": "replacement-pod",
            "INGEST_REAPER_INTERVAL_SECONDS": "1",
            "INGEST_REAPER_MIN_IDLE_MS": "1",
            "BACKEND_URL": "http://localhost",
            "BACKEND_PORT": str(shared_vespa["http_port"]),
            "TELEMETRY_HTTP_ENDPOINT": phoenix_container["http_endpoint"],
            "TELEMETRY_OTLP_ENDPOINT": phoenix_container["otlp_endpoint"],
            "INFERENCE_SERVICE_URLS": json.dumps({}),
        }
    )
    done = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert f"Reaper re-driving orphaned ingest {ingest_id}" in done.stderr
    assert done.stdout.strip().splitlines()[-1] == (
        f"OTLP_ENDPOINT {phoenix_container['otlp_endpoint']}"
    )

    project = f"cogniverse-{tenant_id}"
    names: list = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not names:
        names = asyncio.run(_job_spans(phoenix_container["http_endpoint"], project))
        if not names:
            time.sleep(2)
    assert names == ["pipeline.worker.process_job"]
