"""with_concurrency() must reach the pipeline's concurrency control.

The builder set _max_concurrent but never threaded it into the pipeline, and
process_videos_concurrent had its own per-call default — so with_concurrency
was a silent no-op.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline


class _Stop(Exception):
    pass


def _bare_pipeline(max_concurrent: int) -> VideoIngestionPipeline:
    pipe = object.__new__(VideoIngestionPipeline)
    pipe.max_concurrent = max_concurrent
    pipe.tenant_id = "t"
    pipe.logger = logging.getLogger("test")
    pipe.event_queue = None

    async def _noop(_event):
        return None

    pipe._emit_event = _noop
    return pipe


async def _capture_semaphore(pipe, monkeypatch, **kwargs):
    captured = {}

    def fake_semaphore(n):
        captured["n"] = n
        raise _Stop()

    monkeypatch.setattr(asyncio, "Semaphore", fake_semaphore)
    # A single dummy item: an empty batch short-circuits before the semaphore,
    # so we need one file to reach the concurrency-control setup under test.
    with pytest.raises(_Stop):
        await pipe.process_videos_concurrent(["dummy.mp4"], **kwargs)
    return captured["n"]


@pytest.mark.asyncio
async def test_uses_pipeline_configured_concurrency(monkeypatch):
    pipe = _bare_pipeline(max_concurrent=7)
    n = await _capture_semaphore(pipe, monkeypatch, max_concurrent=None)
    assert n == 7


@pytest.mark.asyncio
async def test_per_call_override_wins(monkeypatch):
    pipe = _bare_pipeline(max_concurrent=7)
    n = await _capture_semaphore(pipe, monkeypatch, max_concurrent=2)
    assert n == 2
