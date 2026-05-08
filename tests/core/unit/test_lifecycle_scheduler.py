"""Unit tests for the memory lifecycle scheduler."""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler


class FakeManager:
    """Mem0MemoryManager-shaped stub for scheduler tests."""

    def __init__(self, tenant_id: str, deletes: int, raise_on_call: bool = False):
        self.tenant_id = tenant_id
        self._deletes = deletes
        self._raise = raise_on_call
        self.calls: list[int] = []

    def cleanup_expired_memories(self, max_age_seconds: int) -> int:
        self.calls.append(max_age_seconds)
        if self._raise:
            raise RuntimeError("simulated cleanup failure")
        return self._deletes


class TestTickOnce:
    @pytest.mark.asyncio
    async def test_runs_cleanup_for_every_warm_manager(self):
        managers = [
            FakeManager("tenant-a", deletes=3),
            FakeManager("tenant-b", deletes=0),
            FakeManager("tenant-c", deletes=12),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            interval_seconds=60.0,
            max_age_seconds=86400,
        )

        summary = await scheduler.tick_once()

        assert [m.calls for m in managers] == [[86400], [86400], [86400]]
        assert summary["total_deleted"] == 15
        assert summary["tenants"] == {
            "tenant-a": 3,
            "tenant-b": 0,
            "tenant-c": 12,
        }
        assert summary["max_age_seconds"] == 86400

    @pytest.mark.asyncio
    async def test_per_tenant_failure_does_not_abort_run(self):
        managers = [
            FakeManager("ok-tenant", deletes=2),
            FakeManager("bad-tenant", deletes=0, raise_on_call=True),
            FakeManager("late-tenant", deletes=5),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            interval_seconds=60.0,
            max_age_seconds=120,
        )

        summary = await scheduler.tick_once()

        # Both surviving tenants ran; failed one is recorded.
        assert summary["tenants"]["ok-tenant"] == 2
        assert summary["tenants"]["late-tenant"] == 5
        assert "RuntimeError" in summary["tenants"]["bad-tenant"]
        assert summary["total_deleted"] == 7

    @pytest.mark.asyncio
    async def test_unnamed_manager_recorded_as_unknown(self):
        class Anon:
            tenant_id = ""  # falsy → "unknown"

            def cleanup_expired_memories(self, _):
                return 1

        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [Anon()],
            interval_seconds=60.0,
        )
        summary = await scheduler.tick_once()
        assert summary["tenants"]["unknown"] == 1


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_runs_periodic_ticks_and_stop_cleanly(self):
        managers = [FakeManager("t", deletes=1)]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            interval_seconds=0.05,
            max_age_seconds=10,
        )
        scheduler.start()

        await asyncio.sleep(0.18)
        await scheduler.stop()

        # Each tick increments managers[0].calls by one.
        assert len(managers[0].calls) >= 2

    @pytest.mark.asyncio
    async def test_stop_safe_when_never_started(self):
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [],
        )
        await scheduler.stop()  # must not raise

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [],
            interval_seconds=0.1,
        )
        scheduler.start()
        first = scheduler._task
        scheduler.start()
        assert scheduler._task is first
        await scheduler.stop()


class TestConstructorValidation:
    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            LifecycleScheduler(get_warm_managers=lambda: [], interval_seconds=0)

    def test_rejects_non_positive_max_age(self):
        with pytest.raises(ValueError):
            LifecycleScheduler(get_warm_managers=lambda: [], max_age_seconds=0)
