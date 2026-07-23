"""``VLMProcessor.from_config`` must fail loud when no VLM endpoint is set.

With the developer's personal Modal endpoint removed from the shipped config,
a fresh deployment has an empty ``vlm_endpoint``. Description generation must
raise a clear error telling the operator to configure it, not silently POST to
an empty URL or to a stale personal endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor


@pytest.mark.unit
def test_from_config_raises_on_empty_endpoint():
    with pytest.raises(ValueError, match="requires 'vlm_endpoint'"):
        VLMProcessor.from_config({"vlm_endpoint": ""}, logging.getLogger("t"))


@pytest.mark.unit
def test_from_config_raises_on_missing_endpoint():
    with pytest.raises(ValueError, match="requires 'vlm_endpoint'"):
        VLMProcessor.from_config({}, logging.getLogger("t"))


@pytest.mark.unit
def test_from_config_accepts_configured_endpoint():
    proc = VLMProcessor.from_config(
        {"vlm_endpoint": "http://vlm.internal:8000/generate", "batch_size": 4},
        logging.getLogger("t"),
    )
    assert proc.vlm_endpoint == "http://vlm.internal:8000/generate"
    assert proc.batch_size == 4


@pytest.mark.unit
def test_get_descriptor_builds_once_under_concurrency(monkeypatch):
    """Concurrent first-touches must build exactly one VLMDescriptor.

    Description runs per-video on worker threads while videos process
    concurrently, so an unguarded lazy-init would construct one descriptor per
    racing first-touch — in Modal auto_start mode each independently deploys the
    service and the discarded ones are never stopped. The double-checked lock
    must serialize the build to exactly one, shared by every caller.
    """
    n_threads = 8
    construction_count = 0
    count_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    class _FakeDescriptor:
        def __init__(self, **kwargs):
            nonlocal construction_count
            with count_lock:
                construction_count += 1
            time.sleep(0.02)  # widen the window an unguarded init would race in

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.processors.vlm_descriptor.VLMDescriptor",
        _FakeDescriptor,
    )

    proc = VLMProcessor(logging.getLogger("t"), "http://vlm.internal:8000/generate")
    results: list = []

    def touch(_):
        barrier.wait()  # release all threads into the check-then-set together
        results.append(proc._get_descriptor())

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(touch, range(n_threads)))

    assert construction_count == 1
    assert len(results) == n_threads
    assert all(d is results[0] for d in results)
