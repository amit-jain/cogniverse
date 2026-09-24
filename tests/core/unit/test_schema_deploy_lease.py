"""The deployment lease admits one writer and refuses an expired one.

The lease is what keeps two processes from each replacing the whole Vespa
application package from its own snapshot. These pin the state machine over a
shared store; the two-process behaviour against real Vespa is pinned by
tests/backends/integration/test_schema_deployment_serialization.py.
"""

from __future__ import annotations

import gc
import logging
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_core.registries import schema_deploy_lease
from cogniverse_core.registries.schema_deploy_lease import (
    DeploymentLeaseLost,
    SchemaDeployLease,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _lease(store, **kwargs):
    return SchemaDeployLease(store, **kwargs)


def test_only_one_of_two_concurrent_holders_takes_the_lease():
    store = InMemoryConfigStore()
    first, second = _lease(store, wait_seconds=0), _lease(store, wait_seconds=0)
    ready = threading.Barrier(2)

    def acquire(lease):
        ready.wait(timeout=10)
        try:
            lease.acquire()
            return lease.holder
        except TimeoutError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(acquire, [first, second]))

    winners = [
        outcome for outcome in outcomes if outcome in (first.holder, second.holder)
    ]
    assert len(winners) == 1
    record = store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    assert record.config_value["holder"] == winners[0]
    losers = [outcome for outcome in outcomes if outcome not in winners]
    assert losers == [
        f"Vespa deployment lease still held by {winners[0]!r} after 0s; refusing "
        f"to replace the application package concurrently with another deployer"
    ]


def test_release_hands_the_lease_to_the_next_holder():
    store = InMemoryConfigStore()
    first, second = _lease(store, wait_seconds=0), _lease(store, wait_seconds=0)
    first.acquire()
    with pytest.raises(TimeoutError):
        second.acquire()
    first.release()
    assert second.acquire() is second
    assert second.renew() is None


def test_expired_holder_is_replaced_and_then_refused():
    store = InMemoryConfigStore()
    expiring = _lease(store, lease_seconds=0.0, wait_seconds=0)
    expiring.acquire()

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor

    with pytest.raises(DeploymentLeaseLost) as caught:
        expiring.renew()
    assert str(caught.value) == "Vespa deployment lease expired or was replaced"

    # The refused holder must not take the lease away on its way out.
    expiring.release()
    record = store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    assert record.config_value["holder"] == successor.holder


def test_a_store_outage_fails_the_acquire_instead_of_granting_it():
    class _Down(InMemoryConfigStore):
        def get_config(self, *args, **kwargs):
            raise ConnectionError("config store unreachable")

    with pytest.raises(ConnectionError, match="config store unreachable"):
        _lease(_Down(), wait_seconds=0).acquire()


def test_a_holder_that_stops_renewing_is_taken_over_after_its_hold_time():
    """Takeover is timed by the waiter's own monotonic clock over the hold
    time the record carries, so the holder's clock never enters the decision."""
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=0.5, wait_seconds=0)
    holder.acquire()

    successor = _lease(store, wait_seconds=5)
    started = time.monotonic()
    assert successor.acquire() is successor
    assert 0.5 <= time.monotonic() - started < 5

    with pytest.raises(DeploymentLeaseLost):
        holder.renew()


def test_wall_clock_skew_cannot_take_over_a_live_holder(monkeypatch):
    """Two nodes whose wall clocks differ by a day still exclude each other."""
    store = InMemoryConfigStore()
    holder = _lease(store, wait_seconds=0)
    holder.acquire()

    real_time = time.time
    monkeypatch.setattr(schema_deploy_lease.time, "time", lambda: real_time() + 86400)

    skewed = _lease(store, wait_seconds=0)
    with pytest.raises(TimeoutError):
        skewed.acquire()
    assert holder.renew() is None
    record = store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    assert record.config_value["holder"] == holder.holder


def test_a_store_outage_at_release_is_logged_not_raised(caplog):
    """The package is already activated when release runs: a store failure
    there must not turn a successful deploy into a raised one."""
    store = InMemoryConfigStore()
    lease = _lease(store, wait_seconds=0)
    lease.acquire()

    def _down(*args, **kwargs):
        raise ConnectionError("config store unreachable")

    store.get_config = _down
    with caplog.at_level(logging.WARNING, logger=schema_deploy_lease.__name__):
        assert lease.release() is None

    assert [record.getMessage() for record in caplog.records] == [
        f"Vespa deployment lease held by {lease.holder} could not be released "
        f"(ConnectionError: config store unreachable); peers take it over after "
        f"600s"
    ]


def test_a_transient_store_error_in_renew_is_not_a_lost_lease():
    """A lost lease and an unreachable store demand different handling: the
    first must abandon the activation, the second is the store's failure."""
    store = InMemoryConfigStore()
    lease = _lease(store, wait_seconds=0)
    lease.acquire()

    def _down(*args, **kwargs):
        raise ConnectionError("config store unreachable")

    store.get_config = _down
    with pytest.raises(ConnectionError, match="config store unreachable") as caught:
        lease.renew()
    assert not isinstance(caught.value, DeploymentLeaseLost)


def test_a_replaced_holder_in_renew_is_not_a_transient_error():
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=0.0, wait_seconds=0)
    holder.acquire()
    successor = _lease(store, wait_seconds=0)
    successor.acquire()

    with pytest.raises(DeploymentLeaseLost) as caught:
        holder.renew()
    assert str(caught.value) == "Vespa deployment lease expired or was replaced"


def _lease_record(store):
    return store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )


def test_a_stalled_record_is_taken_over_by_waits_shorter_than_its_hold_time():
    """Each deploy waits less than the hold time; the stall a process has
    watched carries over from one wait to the next, so a holder that died
    without releasing blocks deploys only for its hold time."""
    store = InMemoryConfigStore()
    dead = _lease(store, lease_seconds=0.6, wait_seconds=0)
    dead.acquire()

    started = time.monotonic()
    outcomes = []
    while time.monotonic() - started < 5:
        waiter = _lease(store, wait_seconds=0.1)
        try:
            waiter.acquire()
        except TimeoutError as exc:
            outcomes.append(str(exc))
            continue
        outcomes.append(waiter.holder)
        break
    elapsed = time.monotonic() - started

    assert outcomes[-1] == waiter.holder
    assert set(outcomes[:-1]) == {
        f"Vespa deployment lease still held by {dead.holder!r} after 0.1s; "
        f"refusing to replace the application package concurrently with another "
        f"deployer"
    }
    assert 0.6 <= elapsed < 1.5
    assert _lease_record(store).config_value["holder"] == waiter.holder


def test_a_renewal_between_two_waits_restarts_the_watched_stall():
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=0.6, wait_seconds=0)
    holder.acquire()

    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0).acquire()
    time.sleep(0.4)
    assert holder.renew() is None
    time.sleep(0.4)
    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0).acquire()

    assert holder.renew() is None
    assert _lease_record(store).config_value["holder"] == holder.holder


@pytest.mark.parametrize("store_fault", ["unreachable", "record_not_visible"])
def test_a_holder_this_process_released_uncleared_is_taken_over_at_once(
    store_fault,
):
    """The holder's own process knows nothing activates under it any more,
    so its record does not block the next deploy for the hold time."""
    store = InMemoryConfigStore()
    released = _lease(store, wait_seconds=0)
    released.acquire()

    real_get = store.get_config

    def _fault(*args, **kwargs):
        if store_fault == "unreachable":
            raise ConnectionError("config store unreachable")
        return None

    store.get_config = _fault
    assert released.release() is None
    store.get_config = real_get
    assert _lease_record(store).config_value["holder"] == released.holder

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor
    assert _lease_record(store).config_value["holder"] == successor.holder


def test_a_live_holder_in_the_same_process_is_not_taken_over():
    store = InMemoryConfigStore()
    holder = _lease(store, wait_seconds=0)
    holder.acquire()

    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0.3).acquire()
    assert holder.renew() is None
    assert _lease_record(store).config_value["holder"] == holder.holder


def test_ensure_owned_fences_a_holder_whose_hold_time_has_passed():
    """A stalled holder must stop, not resume against a lease a peer can take.

    Expiry is judged on the holder's own monotonic clock, so it fences
    itself even while the record still names it — exactly the window in
    which a waiter is entitled to take over.
    """
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=0.2, wait_seconds=0)
    holder.acquire()
    holder.ensure_owned()

    time.sleep(0.25)
    with pytest.raises(DeploymentLeaseLost):
        holder.ensure_owned()


def test_ensure_owned_renews_an_ageing_lease_without_a_store_write_when_fresh():
    """Renewal happens once the lease ages, not before every mutation."""
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=1.0, wait_seconds=0)
    holder.acquire()
    writes_after_acquire = 0

    real_compare_and_set = store.compare_and_set_config

    def counting_compare_and_set(*args, **kwargs):
        nonlocal writes_after_acquire
        writes_after_acquire += 1
        return real_compare_and_set(*args, **kwargs)

    store.compare_and_set_config = counting_compare_and_set

    holder.ensure_owned()
    assert writes_after_acquire == 0

    time.sleep(0.6)
    holder.ensure_owned()
    assert writes_after_acquire == 1

    # The renewal moved the hold window, so the holder outlives its
    # original hold time instead of expiring inside a long operation.
    time.sleep(0.6)
    holder.ensure_owned()
    assert writes_after_acquire == 2


def test_an_abandoned_holder_in_this_process_is_taken_over_at_once():
    """A holder whose owner is gone without a release must not outlast it.

    The record cannot express "the thread that took this died"; the holding
    process can, because it knows which of its own holders are still live.
    Without this the record blocks every later deploy for its whole hold
    time, which no waiter waiting less than that can ever wait out.
    """
    store = InMemoryConfigStore()
    abandoned = _lease(store, wait_seconds=0)
    abandoned.acquire()
    abandoned_holder = abandoned.holder
    assert _lease_record(store).config_value["holder"] == abandoned_holder

    del abandoned
    gc.collect()

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor
    assert _lease_record(store).config_value["holder"] == successor.holder


def test_a_holder_whose_process_was_killed_is_taken_over_at_once():
    """A live holder on this host is refused; the same holder, killed, is not."""
    store = InMemoryConfigStore()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    dead = _lease(store, wait_seconds=0)
    dead.holder = ":".join(
        (
            socket.gethostname(),
            schema_deploy_lease._pid_namespace(),
            str(child.pid),
            uuid.uuid4().hex,
        )
    )
    dead.acquire()

    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0).acquire()

    child.kill()
    assert child.wait(timeout=10) is not None

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor
    assert _lease_record(store).config_value["holder"] == successor.holder


def test_a_holder_on_another_host_is_not_taken_over_by_a_pid_probe():
    """This node cannot observe another node's processes, so the pid in a
    foreign holder is meaningless here; only the stall watch may take it."""
    store = InMemoryConfigStore()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)

    foreign = _lease(store, wait_seconds=0)
    foreign.holder = ":".join(
        (
            f"not-{socket.gethostname()}",
            schema_deploy_lease._pid_namespace(),
            str(child.pid),
            uuid.uuid4().hex,
        )
    )
    foreign.acquire()

    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0).acquire()
    assert _lease_record(store).config_value["holder"] == foreign.holder


def _heartbeat_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("deploy-lease-heartbeat:")
    ]


class _SwitchableStore(InMemoryConfigStore):
    """Refuses or fails the lease's own writes while switched on."""

    def __init__(self):
        super().__init__()
        self.refuse = threading.Event()
        self.fail = threading.Event()
        self.refused = threading.Event()
        self.failed = threading.Event()

    def compare_and_set_config(self, *args, **kwargs):
        if self.refuse.is_set():
            self.refused.set()
            return None
        if self.fail.is_set():
            self.failed.set()
            raise ConnectionError("config store unreachable")
        return super().compare_and_set_config(*args, **kwargs)


def test_a_heartbeating_holder_is_never_taken_over_past_its_hold():
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=0.6, wait_seconds=0, heartbeat=True)
    holder.acquire()
    try:
        peer = _lease(store, wait_seconds=2.4)
        with pytest.raises(TimeoutError):
            peer.acquire()
        assert _lease_record(store).config_value["holder"] == holder.holder
        holder.ensure_owned(renew_after=1.0)
        assert holder.renew() is None
    finally:
        holder.release()
    assert _heartbeat_threads() == []


def test_a_refused_heartbeat_renewal_fences_the_holder_before_its_next_step():
    """Once the store refuses a renewal the lease is gone; the next fence
    refuses at once even if the store would accept a write again."""
    store = _SwitchableStore()
    holder = _lease(store, lease_seconds=3.0, wait_seconds=0, heartbeat=True)
    holder.acquire()
    try:
        store.refuse.set()
        assert store.refused.wait(5) is True
        deadline = time.monotonic() + 5
        while _heartbeat_threads() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _heartbeat_threads() == []
        store.refuse.clear()
        with pytest.raises(DeploymentLeaseLost) as caught:
            holder.ensure_owned(renew_after=1.0)
        assert str(caught.value) == "Vespa deployment lease expired or was replaced"
        with pytest.raises(DeploymentLeaseLost):
            holder.renew()
    finally:
        holder.release()


def test_a_transient_store_error_in_the_heartbeat_keeps_the_lease():
    store = _SwitchableStore()
    holder = _lease(store, lease_seconds=3.0, wait_seconds=0, heartbeat=True)
    holder.acquire()
    try:
        before = _lease_record(store).version
        store.fail.set()
        assert store.failed.wait(5) is True
        store.fail.clear()
        holder.ensure_owned(renew_after=1.0)
        deadline = time.monotonic() + 5
        while _lease_record(store).version == before and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _lease_record(store).version > before
        assert _lease_record(store).config_value["holder"] == holder.holder
    finally:
        holder.release()


def test_the_heartbeat_runs_while_held_and_is_gone_after_release():
    store = InMemoryConfigStore()
    holder = _lease(store, lease_seconds=3.0, wait_seconds=0, heartbeat=True)
    holder.acquire()
    assert [thread.name for thread in _heartbeat_threads()] == [
        f"deploy-lease-heartbeat:{holder.holder}"
    ]
    holder.release()
    assert _heartbeat_threads() == []
    assert _lease_record(store).config_value["holder"] is None


def test_a_failed_acquire_starts_no_heartbeat():
    store = InMemoryConfigStore()
    first = _lease(store, wait_seconds=0)
    first.acquire()
    with pytest.raises(TimeoutError):
        _lease(store, wait_seconds=0, heartbeat=True).acquire()
    assert _heartbeat_threads() == []
    first.release()


def test_an_abandoned_heartbeating_holder_is_still_taken_over_at_once():
    """The heartbeat must not keep a holder alive that its owner dropped."""
    store = InMemoryConfigStore()
    abandoned = _lease(store, lease_seconds=3.0, wait_seconds=0, heartbeat=True)
    abandoned.acquire()
    del abandoned
    gc.collect()

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor
    deadline = time.monotonic() + 5
    while _heartbeat_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _heartbeat_threads() == []
    successor.release()
