"""CacheConfig.enable_stats must actually gate statistics collection.

The flag defaulted True and was documented as a toggle, but no code read it:
get/set/delete always incremented the counters and get_stats always returned
them. Setting enable_stats=False now stops both.
"""

from __future__ import annotations

import pytest

from cogniverse_core.common.cache.base import CacheConfig, CacheManager


class _MemBackend:
    """Minimal in-memory CacheBackend stub (get returns a stored value)."""

    def __init__(self):
        self._d = {}

    async def get(self, key):
        return self._d.get(key)

    async def set(self, key, value, ttl=None):
        self._d[key] = value
        return True

    async def delete(self, key):
        return self._d.pop(key, None) is not None

    async def get_stats(self):
        return {"total_files": len(self._d)}


def _manager(enable_stats: bool) -> CacheManager:
    mgr = CacheManager.__new__(CacheManager)
    mgr.config = CacheConfig(backends=[], enable_stats=enable_stats)
    mgr.backends = [_MemBackend()]
    mgr._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    return mgr


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stats_tracked_when_enabled():
    mgr = _manager(enable_stats=True)
    await mgr.set("k", "v")
    assert await mgr.get("k") == "v"
    await mgr.get("missing")
    await mgr.delete("k")

    stats = await mgr.get_stats()
    assert stats["manager"]["sets"] == 1
    assert stats["manager"]["hits"] == 1
    assert stats["manager"]["misses"] == 1
    assert stats["manager"]["deletes"] == 1
    assert stats["manager"]["hit_rate"] == 0.5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stats_not_tracked_when_disabled():
    mgr = _manager(enable_stats=False)
    await mgr.set("k", "v")
    assert await mgr.get("k") == "v"
    await mgr.get("missing")
    await mgr.delete("k")

    assert mgr._stats == {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    stats = await mgr.get_stats()
    assert stats == {"enabled": False}
