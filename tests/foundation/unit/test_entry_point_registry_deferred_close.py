"""LRU-capacity eviction must not close an instance a request still holds.

The provider cache closed a plugin instance synchronously on capacity eviction.
A concurrent request that had already been handed that instance (and was mid-
await on its HTTP/Vespa session) then hit a use-after-close. The close is now
deferred past a grace window; a cold-start loser and teardown still close now.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.caching import TenantLRUCache
from cogniverse_foundation.registry import entry_point_registry as epr
from cogniverse_foundation.registry.entry_point_registry import EntryPointRegistry

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _ClosableProvider:
    def __init__(self, **kwargs):
        self.closed = False

    def initialize(self, config):
        self._config = config

    def close(self):
        self.closed = True


class _DeferredCloseRegistry(EntryPointRegistry):
    _entry_point_group = "test.deferred.close"
    _label = "deferred close provider"
    _tenant_scoped = True


@pytest.fixture(autouse=True)
def _fresh_registry():
    _DeferredCloseRegistry.reset()
    epr._pending_close.clear()
    # capacity 1 so the second tenant evicts the first.
    _DeferredCloseRegistry._instances = TenantLRUCache(
        capacity=1, on_evict=epr._on_instance_evicted
    )
    _DeferredCloseRegistry.register("p", _ClosableProvider)
    yield
    _DeferredCloseRegistry.reset()
    epr._pending_close.clear()


def test_evicted_instance_is_not_closed_while_held():
    held = _DeferredCloseRegistry.get("p", tenant_id="t1")
    # Inserting a second tenant evicts t1's instance (capacity 1).
    _DeferredCloseRegistry.get("p", tenant_id="t2")

    assert held.closed is False, "evicted instance closed while a holder uses it"
    # It is queued for a deferred close, not dropped.
    queued = [inst for _at, _k, inst in epr._pending_close]
    assert held in queued


def test_grace_window_sweep_closes_queued_instance():
    held = _DeferredCloseRegistry.get("p", tenant_id="t1")
    _DeferredCloseRegistry.get("p", tenant_id="t2")  # evicts + queues t1's

    # Backdate the queued entry past the grace window, then trigger a sweep
    # via another eviction.
    epr._pending_close[:] = [
        (at - epr._EVICTED_CLOSE_GRACE_S - 1, k, inst)
        for (at, k, inst) in epr._pending_close
    ]
    _DeferredCloseRegistry.get("p", tenant_id="t3")  # sweep fires

    assert held.closed is True, "queued instance not closed after the grace window"


def test_reset_flushes_pending_close_immediately():
    held = _DeferredCloseRegistry.get("p", tenant_id="t1")
    _DeferredCloseRegistry.get("p", tenant_id="t2")
    assert held.closed is False

    _DeferredCloseRegistry.reset()

    assert held.closed is True, "teardown must close queued instances immediately"


def test_cold_start_loser_is_closed_immediately(monkeypatch):
    # Force set_if_absent to report a different winner so the built instance is
    # the loser (no other holder) and must be closed now, not deferred.
    winner = _ClosableProvider()

    def _set_if_absent(key, value):
        return winner

    monkeypatch.setattr(
        _DeferredCloseRegistry._instances, "set_if_absent", _set_if_absent
    )
    loser_holder = {}
    real_create = _DeferredCloseRegistry._create_instance.__func__

    def _capture_create(cls, klass, config, tenant_id):
        inst = real_create(cls, klass, config, tenant_id)
        loser_holder["inst"] = inst
        return inst

    monkeypatch.setattr(
        _DeferredCloseRegistry,
        "_create_instance",
        classmethod(_capture_create),
    )

    got = _DeferredCloseRegistry.get("p", tenant_id="t1")
    assert got is winner
    assert loser_holder["inst"].closed is True
    assert loser_holder["inst"] not in [i for _a, _k, i in epr._pending_close]
