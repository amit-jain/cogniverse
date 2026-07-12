"""SingleVectorSegmentationStrategy.segment must offload the sync processor.

process_video() decodes/embeds a whole video synchronously; running it directly
on the event loop inside async segment() serialised concurrent ingests and
stalled every other task. It is now ``await asyncio.to_thread(process_video)``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.ingestion.strategies import SingleVectorSegmentationStrategy

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_segment_offloads_process_video():
    def _slow_process_video(video_path, transcript_data=None):
        time.sleep(0.3)  # stands in for real decode/embed CPU work
        return {"segments": [SimpleNamespace(to_dict=lambda: {"i": 0})], "meta": {}}

    stub = SimpleNamespace(process_video=_slow_process_video)
    ctx = SimpleNamespace(
        processor_manager=SimpleNamespace(get_processor=lambda *a, **k: stub)
    )
    strat = SingleVectorSegmentationStrategy.__new__(SingleVectorSegmentationStrategy)

    async def one():
        return await strat.segment(
            video_path=Path("/v.mp4"), pipeline_context=ctx, transcript_data=None
        )

    t0 = time.monotonic()
    await asyncio.gather(*(one() for _ in range(4)))
    wall = time.monotonic() - t0

    # Offloaded → the 4 blocking calls run concurrently on the thread pool
    # (~0.3s total). On the loop they serialise (~1.2s).
    assert wall < 0.9, f"segment blocked the event loop: {wall:.2f}s for 4x0.3s work"
