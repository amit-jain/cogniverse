"""/ingestion/upload must bound the body so one request can't OOM the pod.

_read_capped reads in chunks and aborts with 413 once the configured limit is
exceeded, rather than buffering an unbounded body via file.read().
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cogniverse_runtime.routers.ingestion import _read_capped


class _FakeUpload:
    """UploadFile-like object that yields fixed-size chunks up to total_bytes."""

    def __init__(self, total_bytes: int, chunk: int = 8 * 1024 * 1024):
        self._remaining = total_bytes
        self._chunk = chunk

    async def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = min(size if size and size > 0 else self._chunk, self._remaining)
        self._remaining -= n
        return b"x" * n


@pytest.mark.asyncio
async def test_under_limit_reads_full_body():
    data = await _read_capped(_FakeUpload(10 * 1024 * 1024), max_bytes=50 * 1024 * 1024)
    assert len(data) == 10 * 1024 * 1024


@pytest.mark.asyncio
async def test_over_limit_raises_413():
    with pytest.raises(HTTPException) as exc:
        await _read_capped(_FakeUpload(60 * 1024 * 1024), max_bytes=50 * 1024 * 1024)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_empty_upload_returns_empty_bytes():
    assert await _read_capped(_FakeUpload(0), max_bytes=1024) == b""
