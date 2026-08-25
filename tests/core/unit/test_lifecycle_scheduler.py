"""Unit tests for the schema-driven memory lifecycle scheduler."""

from __future__ import annotations

import asyncio
import threading

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.schema import build_default_registry


class FakeManager:
    """Mem0MemoryManager-shaped stub exercising schema probes and cleanup."""

    def __init__(
        self,
        tenant_id: str,
        deletes_by_kind: dict | None = None,
        schema_exists: bool = True,
        probe_error: Exception | None = None,
        raise_on_call: bool = False,
        probe_started: threading.Event | None = None,
        probe_release: threading.Event | None = None,
    ):
        self.tenant_id = tenant_id
        self._deletes = deletes_by_kind or {}
        self._schema_exists = schema_exists
        self._probe_error = probe_error
        self._raise = raise_on_call
        self._probe_started = probe_started
        self._probe_release = probe_release
        self.schema_probe_calls = 0
        self.schema_probe_args: list[str] = []
        self.cleanup_calls = 0
        self.calls: list[tuple] = []

    def tenant_partition_schema_exists(self, tenant_id: str) -> bool:
        self.schema_probe_calls += 1
        self.schema_probe_args.append(tenant_id)
        if self._probe_started is not None:
            self._probe_started.set()
        if self._probe_release is not None:
            if not self._probe_release.wait(timeout=2):
                raise TimeoutError("probe release timed out")
        if self._probe_error is not None:
            raise self._probe_error
        return self._schema_exists

    def cleanup_with_schema(self, registry, pinned_ids):
        self.cleanup_calls += 1
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
            FakeManager("tenant-b", {"learned_strategy": 0}, schema_exists=False),
            FakeManager("tenant-c", {"external_doc": 12}),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            registry=registry,
            interval_seconds=60.0,
        )

        summary = await scheduler.tick_once()

        assert summary == {
            "tenants": {
                "tenant-a": {"conversation_turn": 3},
                "tenant-b": "schema absent",
                "tenant-c": {"external_doc": 12},
            },
            "total_deleted": 15,
        }
        assert managers[0].cleanup_calls == 1
        assert managers[1].cleanup_calls == 0
        assert managers[2].cleanup_calls == 1
        assert managers[0].schema_probe_args == [canonical_tenant_id("tenant-a")]
        assert managers[1].schema_probe_args == [canonical_tenant_id("tenant-b")]
        assert managers[2].schema_probe_args == [canonical_tenant_id("tenant-c")]
        for m in (managers[0], managers[2]):
            assert m.calls[0][0] is registry

    @pytest.mark.unit
    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_pin_lookup_failure_skips_cleanup_never_prunes(self, registry):
        """A pin-lookup failure (backend outage) must skip cleanup entirely,
        NOT proceed with an empty pin set — else the scheduler would prune
        genuinely-pinned memories it couldn't confirm."""
        manager = FakeManager("outage-tenant", {"conversation_turn": 9})

        def _raising_pin_lookup(_manager):
            raise ConnectionError("pin store unreachable")

        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [manager],
            registry=registry,
            interval_seconds=60.0,
            pin_lookup=_raising_pin_lookup,
        )

        summary = await scheduler.tick_once()

        # cleanup_with_schema was NEVER called — nothing was pruned.
        assert manager.calls == []
        assert summary["total_deleted"] == 0
        # The tenant is recorded as errored so operators can investigate.
        assert summary["tenants"] == {
            "outage-tenant": "error: ConnectionError",
        }

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

        assert summary == {
            "tenants": {
                "ok-tenant": {"conversation_turn": 2},
                "bad-tenant": "error: RuntimeError",
                "late-tenant": {"external_doc": 5},
            },
            "total_deleted": 7,
        }

    @pytest.mark.asyncio
    async def test_unnamed_manager_recorded_as_unknown(self, registry):
        class Anon:
            tenant_id = ""  # falsy → "unknown"

            def tenant_partition_schema_exists(self, _tenant_id):
                return True

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

    @pytest.mark.asyncio
    async def test_sequential_sweep_keeps_tenants_isolated(self, registry):
        first_probe_started = threading.Event()
        release_first_probe = threading.Event()
        managers = [
            FakeManager(
                "probe-error",
                probe_error=ConnectionError("schema probe failed"),
                probe_started=first_probe_started,
                probe_release=release_first_probe,
            ),
            FakeManager(
                "schema-absent",
                {"learned_strategy": 0},
                schema_exists=False,
            ),
            FakeManager(
                "cleanup-error",
                raise_on_call=True,
            ),
            FakeManager(
                "healthy",
                {"external_doc": 3},
            ),
        ]
        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: managers,
            registry=registry,
            interval_seconds=60.0,
        )

        task = asyncio.create_task(scheduler.tick_once())
        await asyncio.wait_for(asyncio.to_thread(first_probe_started.wait), timeout=2)
        await asyncio.sleep(0.05)

        assert managers[1].schema_probe_calls == 0
        assert managers[2].schema_probe_calls == 0
        assert managers[3].schema_probe_calls == 0

        release_first_probe.set()
        summary = await asyncio.wait_for(task, timeout=3)

        assert summary == {
            "tenants": {
                "probe-error": "error: ConnectionError",
                "schema-absent": "schema absent",
                "cleanup-error": "error: RuntimeError",
                "healthy": {"external_doc": 3},
            },
            "total_deleted": 3,
        }
        assert managers[0].schema_probe_calls == 1
        assert managers[1].schema_probe_calls == 1
        assert managers[2].schema_probe_calls == 1
        assert managers[3].schema_probe_calls == 1
        assert managers[0].schema_probe_args == [canonical_tenant_id("probe-error")]
        assert managers[1].schema_probe_args == [canonical_tenant_id("schema-absent")]
        assert managers[2].schema_probe_args == [canonical_tenant_id("cleanup-error")]
        assert managers[3].schema_probe_args == [canonical_tenant_id("healthy")]
        assert managers[0].cleanup_calls == 0
        assert managers[1].cleanup_calls == 0
        assert managers[2].cleanup_calls == 1
        assert managers[3].cleanup_calls == 1
        assert managers[1].calls == []
        assert managers[3].calls[0][0] is registry


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
