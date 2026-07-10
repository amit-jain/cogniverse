"""Unit tests for the schema-driven memory lifecycle scheduler."""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.schema import build_default_registry


class FakeManager:
    """Mem0MemoryManager-shaped stub exercising cleanup_with_schema."""

    def __init__(
        self,
        tenant_id: str,
        deletes_by_kind: dict | None = None,
        raise_on_call: bool = False,
    ):
        self.tenant_id = tenant_id
        self._deletes = deletes_by_kind or {}
        self._raise = raise_on_call
        self.calls: list[tuple] = []

    def cleanup_with_schema(self, registry, pinned_ids):
        self.calls.append((registry, pinned_ids))
        if self._raise:
            raise RuntimeError("simulated cleanup failure")
        return dict(self._deletes)


@pytest.fixture
def registry():
    return build_default_registry()


class TestTickOnce:
    @pytest.mark.unit
    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_runs_cleanup_for_every_warm_manager(self, registry):
        managers = [
            FakeManager("tenant-a", {"conversation_turn": 3}),
            FakeManager("tenant-b", {"learned_strategy": 0}),
            FakeManager("tenant-c", {"external_doc": 12}),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            registry=registry,
            interval_seconds=60.0,
        )

        summary = await scheduler.tick_once()

        assert summary["total_deleted"] == 15
        assert summary["tenants"]["tenant-a"] == {"conversation_turn": 3}
        assert summary["tenants"]["tenant-b"] == {"learned_strategy": 0}
        assert summary["tenants"]["tenant-c"] == {"external_doc": 12}
        # Each manager called once with the same registry instance.
        for m in managers:
            assert len(m.calls) == 1
            assert m.calls[0][0] is registry

    @pytest.mark.unit
    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_per_tenant_failure_does_not_abort_run(self, registry):
        managers = [
            FakeManager("ok-tenant", {"conversation_turn": 2}),
            FakeManager("bad-tenant", raise_on_call=True),
            FakeManager("late-tenant", {"external_doc": 5}),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            registry=registry,
            interval_seconds=60.0,
        )

        summary = await scheduler.tick_once()

        assert summary["tenants"]["ok-tenant"] == {"conversation_turn": 2}
        assert summary["tenants"]["late-tenant"] == {"external_doc": 5}
        assert "RuntimeError" in summary["tenants"]["bad-tenant"]
        assert summary["total_deleted"] == 7

    @pytest.mark.asyncio
    async def test_unnamed_manager_recorded_as_unknown(self, registry):
        class Anon:
            tenant_id = ""  # falsy → "unknown"

            def cleanup_with_schema(self, _registry, _pinned):
                return {"conversation_turn": 1}

        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [Anon()],
            registry=registry,
            interval_seconds=60.0,
        )
        summary = await scheduler.tick_once()
        assert summary["tenants"]["unknown"] == {"conversation_turn": 1}

    @pytest.mark.asyncio
    async def test_pin_lookup_threaded_through(self, registry):
        manager = FakeManager("t1", {"conversation_turn": 1})
        captured = {}

        def pin_lookup(mm):
            captured["called_with"] = mm
            return {"m_pinned_1", "m_pinned_2"}

        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [manager],
            registry=registry,
            pin_lookup=pin_lookup,
        )

        await scheduler.tick_once()

        assert manager.calls[0][1] == {"m_pinned_1", "m_pinned_2"}
        assert captured["called_with"] is manager

    @pytest.mark.asyncio
    async def test_pin_lookup_default_is_empty_set(self, registry):
        manager = FakeManager("t1", {"conversation_turn": 1})
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [manager],
            registry=registry,
        )
        await scheduler.tick_once()
        assert manager.calls[0][1] == set()


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_runs_periodic_ticks_and_stop_cleanly(self, registry):
        managers = [FakeManager("t", {"conversation_turn": 1})]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            registry=registry,
            interval_seconds=0.05,
        )
        scheduler.start()

        await asyncio.sleep(0.18)
        await scheduler.stop()

        assert len(managers[0].calls) >= 2

    @pytest.mark.asyncio
    async def test_stop_safe_when_never_started(self, registry):
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [],
            registry=registry,
        )
        await scheduler.stop()  # must not raise

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self, registry):
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [],
            registry=registry,
            interval_seconds=0.1,
        )
        scheduler.start()
        first = scheduler._task
        scheduler.start()
        assert scheduler._task is first
        await scheduler.stop()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestConstructorValidation:
    def test_rejects_non_positive_interval(self, registry):
        with pytest.raises(ValueError):
            LifecycleScheduler(
                get_warm_managers=lambda: [], registry=registry, interval_seconds=0
            )

    def test_rejects_missing_registry(self):
        with pytest.raises(ValueError, match="requires a KnowledgeRegistry"):
            LifecycleScheduler(get_warm_managers=lambda: [], registry=None)
