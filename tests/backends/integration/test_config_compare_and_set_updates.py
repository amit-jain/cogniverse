"""Read-modify-write config updates over a real Vespa config store.

Every runtime worker process and replica holds its own store session. These
tests stand several sessions, threads and OS processes on one Vespa and pin
that a read-modify-write never writes back over another writer's change, that
a writer losing every race raises instead of writing, and that a store failing
mid-update leaves the stored value as it was.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from vespa.exceptions import VespaError

from cogniverse_foundation.config.manager import (
    BackendProfileExistsError,
    ConfigManager,
)
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_sdk.interfaces.config_store import (
    CONFIG_UPDATE_MAX_ATTEMPTS,
    ConfigScope,
    ConfigStoreUnavailableError,
    ConfigWriteConflictError,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.requires_vespa]

# Nothing listens here: the repo-wide dead-port convention.
DEAD_PORT = 29071
SERVICE = "cas_updates"
KEY = "selections"


def _tenant(label: str) -> str:
    name = f"cas{label}{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


def _store(port: int) -> VespaConfigStore:
    return VespaConfigStore(backend_url="http://localhost", backend_port=port)


def _with(key: str, value):
    def update(entry):
        current = dict(entry.config_value) if entry is not None else {}
        current[key] = value
        return current

    return update


def _delete(store: VespaConfigStore, tenant: str, scope=ConfigScope.SYSTEM) -> None:
    service, key = (
        ("backend", "backend_config")
        if scope is ConfigScope.BACKEND
        else (SERVICE, KEY)
    )
    store.delete_config(tenant, scope, service, key)


def _add_competitor_key(tenant: str, value: dict, write: int) -> dict:
    return {**value, f"competitor{write}": write}


def _add_competitor_profile(tenant: str, value: dict, write: int) -> dict:
    profiles = {**value.get("profiles", {}), f"competitor{write}": {"type": "document"}}
    return {**value, "tenant_id": tenant, "profiles": profiles}


class _RacedStore(VespaConfigStore):
    """A real store whose next ``races`` compare-and-sets each lose to a
    competing session's write that lands after this one read the config."""

    def __init__(
        self,
        port: int,
        competitor: VespaConfigStore,
        races: int,
        compete=_add_competitor_key,
    ) -> None:
        super().__init__(backend_url="http://localhost", backend_port=port)
        self._competitor = competitor
        self._races = races
        self._compete = compete
        self.competitor_writes = 0

    def compare_and_set_config(self, tenant_id, scope, service, config_key, *a, **kw):
        if self.competitor_writes < self._races:
            self.competitor_writes += 1
            current = self._competitor.get_config(tenant_id, scope, service, config_key)
            value = self._compete(
                tenant_id,
                dict(current.config_value) if current is not None else {},
                self.competitor_writes,
            )
            self._competitor.set_config(tenant_id, scope, service, config_key, value)
        return super().compare_and_set_config(
            tenant_id, scope, service, config_key, *a, **kw
        )


@pytest.fixture
def store(vespa_instance):
    store = _store(vespa_instance["http_port"])
    yield store
    store.close()


class TestUpdateConfig:
    def test_creates_then_changes_only_what_the_update_returns(self, store):
        tenant = _tenant("create")
        try:
            created = store.update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("a", 1)
            )
            changed = store.update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("b", 2)
            )

            assert (created.version, created.config_value) == (1, {"a": 1})
            assert (changed.version, changed.config_value) == (2, {"a": 1, "b": 2})
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert (stored.version, stored.config_value) == (2, {"a": 1, "b": 2})
        finally:
            _delete(store, tenant)

    def test_a_declined_update_writes_no_version(self, store):
        tenant = _tenant("decline")
        try:
            store.update_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("a", 1))
            seen = []

            def decline(entry):
                seen.append((entry.version, entry.config_value))
                return None

            held = store.update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, decline
            )

            assert seen == [(1, {"a": 1})]
            assert (held.version, held.config_value) == (1, {"a": 1})
            history = store.get_config_history(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert [entry.version for entry in history] == [1]
        finally:
            _delete(store, tenant)

    def test_a_lost_race_reapplies_the_update_onto_the_winners_value(
        self, vespa_instance, store
    ):
        tenant = _tenant("race")
        raced = _RacedStore(vespa_instance["http_port"], store, races=2)
        try:
            store.update_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("a", 1))
            seen = []

            def add_mine(entry):
                seen.append(dict(entry.config_value))
                return {**entry.config_value, "mine": True}

            written = raced.update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, add_mine
            )

            assert raced.competitor_writes == 2
            assert seen == [
                {"a": 1},
                {"a": 1, "competitor1": 1},
                {"a": 1, "competitor1": 1, "competitor2": 2},
            ]
            assert (written.version, written.config_value) == (
                4,
                {"a": 1, "competitor1": 1, "competitor2": 2, "mine": True},
            )
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert stored.config_value == written.config_value
        finally:
            raced.close()
            _delete(store, tenant)

    def test_losing_every_race_raises_and_writes_nothing(self, vespa_instance, store):
        tenant = _tenant("conflict")
        raced = _RacedStore(vespa_instance["http_port"], store, races=3)
        try:
            with pytest.raises(ConfigWriteConflictError) as caught:
                raced.update_config(
                    tenant,
                    ConfigScope.SYSTEM,
                    SERVICE,
                    KEY,
                    _with("mine", True),
                    max_attempts=3,
                )

            assert str(caught.value) == (
                f"config {tenant}:system:{SERVICE}:{KEY} changed under every one "
                "of 3 compare-and-set attempts; nothing was written"
            )
            assert (caught.value.config_id, caught.value.attempts) == (
                f"{tenant}:system:{SERVICE}:{KEY}",
                3,
            )
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert (stored.version, stored.config_value) == (
                3,
                {"competitor1": 1, "competitor2": 2, "competitor3": 3},
            )
        finally:
            raced.close()
            _delete(store, tenant)

    def test_concurrent_sessions_each_keep_their_change(self, vespa_instance, store):
        tenant = _tenant("threads")
        writers = 12
        sessions = [_store(vespa_instance["http_port"]) for _ in range(writers)]
        barrier = threading.Barrier(writers)

        def write(index: int):
            barrier.wait(timeout=60)
            return sessions[index].update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with(f"w{index}", index)
            )

        try:
            with ThreadPoolExecutor(max_workers=writers) as pool:
                written = list(pool.map(write, range(writers)))

            expected = {f"w{index}": index for index in range(writers)}
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert (stored.version, stored.config_value) == (writers, expected)
            assert sorted(entry.version for entry in written) == list(
                range(1, writers + 1)
            )
        finally:
            for session in sessions:
                session.close()
            _delete(store, tenant)

    def test_concurrent_processes_each_keep_their_change(self, vespa_instance, store):
        tenant = _tenant("procs")
        processes = 4
        keys_per_process = 3
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(processes)
        errors = context.Queue()
        workers = [
            context.Process(
                target=_update_in_process,
                args=(
                    vespa_instance["http_port"],
                    tenant,
                    [f"p{index}k{key}" for key in range(keys_per_process)],
                    barrier,
                    errors,
                ),
            )
            for index in range(processes)
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=300)
            reported = [errors.get(timeout=5) for _ in workers]

            assert reported == [None] * processes
            assert [worker.exitcode for worker in workers] == [0] * processes
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert sorted(stored.config_value) == sorted(
                f"p{index}k{key}"
                for index in range(processes)
                for key in range(keys_per_process)
            )
            assert stored.version == processes * keys_per_process
            assert {
                pid for name, pid in stored.config_value.items() if name.endswith("k0")
            } == {worker.pid for worker in workers}
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
            _delete(store, tenant)

    def test_an_unreachable_store_raises_before_the_update_runs(self):
        dead = _store(DEAD_PORT)
        calls = []
        try:
            with pytest.raises(ConfigStoreUnavailableError) as caught:
                dead.update_config(
                    _tenant("dead"),
                    ConfigScope.SYSTEM,
                    SERVICE,
                    KEY,
                    lambda entry: calls.append(entry) or {"a": 1},
                )
        finally:
            dead.close()

        assert calls == []
        assert re.fullmatch(
            r"Failed to read Vespa config visit after 5 attempts over \d+\.\d{3}s: "
            r"ConnectionError: .*",
            str(caught.value),
            re.DOTALL,
        )

    def test_a_refused_write_leaves_the_stored_value(self, vespa_instance, store):
        tenant = _tenant("refused")
        upstream = f"http://localhost:{vespa_instance['http_port']}"
        try:
            store.update_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("a", 1))
            with InterceptFaultProxy(upstream) as proxy:
                proxy.intercept = lambda method, path, body: (
                    (503, {"message": "write refused"}) if method == "POST" else None
                )
                proxied = VespaConfigStore(
                    backend_url="http://127.0.0.1", backend_port=proxy.port
                )
                try:
                    with pytest.raises(VespaError) as caught:
                        proxied.update_config(
                            tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with("b", 2)
                        )
                finally:
                    proxied.close()

            assert [method for method, _, _ in proxy.requests].count("POST") == 1
            assert str(caught.value) == "write refused"
            assert str(caught.value.__cause__).startswith("HTTP 503:")
            stored = store.get_config(tenant, ConfigScope.SYSTEM, SERVICE, KEY)
            assert (stored.version, stored.config_value) == (1, {"a": 1})
        finally:
            _delete(store, tenant)


def _update_in_process(port, tenant, keys, barrier, errors) -> None:
    store = _store(port)
    try:
        barrier.wait(timeout=120)
        for key in keys:
            store.update_config(
                tenant, ConfigScope.SYSTEM, SERVICE, KEY, _with(key, os.getpid())
            )
    except BaseException as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
        raise
    else:
        errors.put(None)
    finally:
        store.close()


def _profile(name: str, model: str = "") -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(
        name,
        {"type": "document", "schema_name": f"{name}_schema", "embedding_model": model},
    )


def _manager(port: int, notifications: list | None = None) -> ConfigManager:
    manager = ConfigManager(store=_store(port))
    if notifications is not None:
        manager.set_profile_change_listener(
            lambda event, name, config: notifications.append((event, name))
        )
    return manager


def _stored_profiles(store: VespaConfigStore, tenant: str) -> dict:
    entry = store.get_config(tenant, ConfigScope.BACKEND, "backend", "backend_config")
    return {
        name: profile.get("embedding_model", "")
        for name, profile in entry.config_value["profiles"].items()
    }


class TestProfileChangesAcrossWriters:
    def test_concurrent_profile_changes_from_separate_managers_all_persist(
        self, vespa_instance, store
    ):
        tenant = _tenant("profiles")
        port = vespa_instance["http_port"]
        seed = _manager(port)
        seed.add_backend_profile(_profile("doomed"), tenant_id=tenant)
        seed.add_backend_profile(_profile("tuned", "model-v1"), tenant_id=tenant)
        managers = [_manager(port) for _ in range(8)]
        barrier = threading.Barrier(len(managers))

        def change(index: int):
            manager = managers[index]
            barrier.wait(timeout=60)
            if index == 0:
                return manager.delete_backend_profile("doomed", tenant_id=tenant)
            if index == 1:
                return manager.update_backend_profile(
                    "tuned",
                    {"embedding_model": "model-v2"},
                    base_tenant_id=tenant,
                    target_tenant_id=tenant,
                ).embedding_model
            return manager.add_backend_profile(
                _profile(f"added{index}"), tenant_id=tenant
            ).profile_name

        try:
            with ThreadPoolExecutor(max_workers=len(managers)) as pool:
                results = list(pool.map(change, range(len(managers))))

            assert results == [True, "model-v2"] + [f"added{i}" for i in range(2, 8)]
            assert _stored_profiles(store, tenant) == {
                "tuned": "model-v2",
                **{f"added{i}": "" for i in range(2, 8)},
            }
        finally:
            for manager in [seed, *managers]:
                manager.store.close()
            _delete(store, tenant, ConfigScope.BACKEND)

    def test_profile_adds_from_separate_processes_all_persist(
        self, vespa_instance, store
    ):
        tenant = _tenant("profprocs")
        processes = 4
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(processes)
        errors = context.Queue()
        workers = [
            context.Process(
                target=_add_profile_in_process,
                args=(vespa_instance["http_port"], tenant, index, barrier, errors),
            )
            for index in range(processes)
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=300)
            reported = [errors.get(timeout=5) for _ in workers]

            assert reported == [None] * processes
            assert [worker.exitcode for worker in workers] == [0] * processes
            assert _stored_profiles(store, tenant) == {
                f"process{index}": "" for index in range(processes)
            }
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
            _delete(store, tenant, ConfigScope.BACKEND)

    def test_create_only_adds_of_one_name_from_separate_managers_store_one(
        self, vespa_instance, store
    ):
        tenant = _tenant("createonly")
        port = vespa_instance["http_port"]
        managers = [_manager(port) for _ in range(4)]
        barrier = threading.Barrier(len(managers))

        def create(index: int):
            barrier.wait(timeout=60)
            try:
                managers[index].add_backend_profile(
                    _profile("shared", f"model{index}"), tenant_id=tenant, replace=False
                )
            except BackendProfileExistsError as exc:
                return str(exc)
            return index

        try:
            with ThreadPoolExecutor(max_workers=len(managers)) as pool:
                outcomes = list(pool.map(create, range(len(managers))))

            winners = [outcome for outcome in outcomes if isinstance(outcome, int)]
            assert len(winners) == 1
            assert sorted(o for o in outcomes if isinstance(o, str)) == [
                f"Profile 'shared' already exists for tenant '{tenant}'"
            ] * (len(managers) - 1)
            assert _stored_profiles(store, tenant) == {"shared": f"model{winners[0]}"}
            history = store.get_config_history(
                tenant, ConfigScope.BACKEND, "backend", "backend_config"
            )
            assert [entry.version for entry in history] == [1]
        finally:
            for manager in managers:
                manager.store.close()
            _delete(store, tenant, ConfigScope.BACKEND)

    def test_reaffirming_an_identical_profile_writes_no_version_but_notifies(
        self, vespa_instance, store
    ):
        tenant = _tenant("reaffirm")
        notifications: list = []
        manager = _manager(vespa_instance["http_port"], notifications)
        try:
            manager.add_backend_profile(_profile("wiki", "m1"), tenant_id=tenant)
            manager.add_backend_profile(_profile("wiki", "m1"), tenant_id=tenant)
            assert manager.delete_backend_profile("absent", tenant_id=tenant) is False

            history = store.get_config_history(
                tenant, ConfigScope.BACKEND, "backend", "backend_config"
            )
            assert [entry.version for entry in history] == [1]
            assert notifications == [("added", "wiki"), ("added", "wiki")]
        finally:
            manager.store.close()
            _delete(store, tenant, ConfigScope.BACKEND)

    def test_a_profile_change_losing_every_race_raises_and_notifies_nothing(
        self, vespa_instance, store
    ):
        tenant = _tenant("profconflict")
        notifications: list = []
        raced = _RacedStore(
            vespa_instance["http_port"],
            store,
            races=CONFIG_UPDATE_MAX_ATTEMPTS,
            compete=_add_competitor_profile,
        )
        manager = ConfigManager(store=raced)
        manager.set_profile_change_listener(
            lambda event, name, config: notifications.append((event, name))
        )
        try:
            with pytest.raises(ConfigWriteConflictError) as caught:
                manager.add_backend_profile(_profile("lost"), tenant_id=tenant)

            assert caught.value.config_id == f"{tenant}:backend:backend:backend_config"
            assert caught.value.attempts == CONFIG_UPDATE_MAX_ATTEMPTS
            assert notifications == []
            assert _stored_profiles(store, tenant) == {
                f"competitor{write}": ""
                for write in range(1, CONFIG_UPDATE_MAX_ATTEMPTS + 1)
            }
        finally:
            raced.close()
            _delete(store, tenant, ConfigScope.BACKEND)

    def test_a_profile_change_against_an_unreachable_store_raises_and_notifies_nothing(
        self,
    ):
        notifications: list = []
        manager = _manager(DEAD_PORT, notifications)
        tenant = _tenant("profdead")
        try:
            with pytest.raises(ConfigStoreUnavailableError):
                manager.add_backend_profile(_profile("unwritten"), tenant_id=tenant)
            with pytest.raises(ConfigStoreUnavailableError):
                manager.delete_backend_profile("unwritten", tenant_id=tenant)
        finally:
            manager.store.close()

        assert notifications == []


def _add_profile_in_process(port, tenant, index, barrier, errors) -> None:
    manager = _manager(port)
    try:
        barrier.wait(timeout=120)
        manager.add_backend_profile(_profile(f"process{index}"), tenant_id=tenant)
    except BaseException as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
        raise
    else:
        errors.put(None)
    finally:
        manager.store.close()
