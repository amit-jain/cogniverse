"""Admin and config writes stay correct across the runtime's worker processes.

Boots the image's own command with two uvicorn workers against the test Vespa
and a test-owned Redis, exactly as ``test_runtime_worker_processes`` does, and
drives the admin routes over connections pinned to each worker — which worker
serves a connection is read from ``/proc``, never from the runtime's answer.
The stored records are read from the config store directly.
"""

from __future__ import annotations

import http.client
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.runtime.integration.test_runtime_worker_processes import (
    WORKERS,
    _runtime,
    _serving,
    _serving_worker,
)

pytestmark = pytest.mark.integration

# Connections opened before every worker holds the share a test asks for; the
# kernel spreads new connections over the workers' listening sockets. uvicorn
# closes a connection idle for 5 s, so each phase of a test pins its own.
MAX_CONNECTIONS = 200


@pytest.fixture(scope="module")
def runtime(tmp_path_factory, workflow_state_redis_url, vespa_instance):
    with _runtime(tmp_path_factory.mktemp("admin_state"), workflow_state_redis_url) as (
        process,
        log,
        port,
    ):
        workers = _serving(process, log)
        assert len(workers) == WORKERS
        yield SimpleNamespace(process=process, log=log, port=port, workers=workers)


@pytest.fixture
def store(vespa_instance):
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    yield store
    store.close()


def _request(
    connection: http.client.HTTPConnection, method: str, path: str, body=None
) -> tuple[int, dict]:
    payload = None if body is None else json.dumps(body)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def _pinned(runtime, per_worker: int) -> dict[int, list[http.client.HTTPConnection]]:
    """``per_worker`` open connections served by each worker, keyed by pid."""
    pinned: dict[int, list[http.client.HTTPConnection]] = {
        pid: [] for pid in runtime.workers
    }
    extra = []
    for _ in range(MAX_CONNECTIONS):
        if all(len(held) == per_worker for held in pinned.values()):
            break
        connection = http.client.HTTPConnection("127.0.0.1", runtime.port, timeout=120)
        # Answered once, so the serving worker has accepted the connection.
        assert _request(connection, "GET", "/health/live") == (200, {"status": "alive"})
        owner = _serving_worker(runtime.port, connection, runtime.workers)
        if len(pinned[owner]) < per_worker:
            pinned[owner].append(connection)
        else:
            extra.append(connection)
    for connection in extra:
        connection.close()
    assert {pid: len(held) for pid, held in pinned.items()} == {
        pid: per_worker for pid in runtime.workers
    }
    return pinned


def _close(pinned) -> None:
    for held in pinned.values():
        for connection in held:
            connection.close()


def _tenant(label: str) -> str:
    name = f"workers{label}{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


def _record(store: VespaConfigStore, tenant: str, key: str):
    return store.get_config(tenant, ConfigScope.SYSTEM, "admin_overrides", key)


def _concurrently(calls) -> list:
    """Run each call on its own thread, released together by one barrier."""
    barrier = threading.Barrier(len(calls))
    results: list = [None] * len(calls)

    def run(index, call):
        barrier.wait(timeout=60)
        results[index] = call()

    threads = [
        threading.Thread(target=run, args=(index, call))
        for index, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    assert [thread.is_alive() for thread in threads] == [False] * len(threads)
    return results


class TestAdminOverridesAcrossWorkers:
    def test_a_pin_quota_put_on_one_worker_is_what_the_other_serves(
        self, runtime, store
    ):
        tenant = _tenant("pinq")
        pinned = _pinned(runtime, 1)
        first, second = (pinned[pid][0] for pid in runtime.workers)
        path = f"/admin/tenants/{tenant}/pin_quotas"
        try:
            put_first = _request(first, "PUT", path, {"user": 7})
            read_second = _request(second, "GET", path)
            put_second = _request(second, "PUT", path, {"tenant_admin": 9})
            read_first = _request(first, "GET", path)
        finally:
            _close(pinned)

        set_user = {"user": 7, "tenant_admin": 500, "org_admin": -1}
        both = {"user": 7, "tenant_admin": 9, "org_admin": -1}
        # What the other worker serves first: the cross-process read is the
        # contract under test.
        assert read_second == (200, {"tenant_id": tenant, "quotas": set_user})
        assert read_first == (200, {"tenant_id": tenant, "quotas": both})
        assert put_first == (200, {"tenant_id": tenant, "quotas": set_user})
        assert put_second == (200, {"tenant_id": tenant, "quotas": both})
        stored = _record(store, tenant, "pin_quotas")
        assert (stored.version, stored.config_value) == (2, both)

    def test_concurrent_variant_puts_on_both_workers_keep_every_selection(
        self, runtime, store
    ):
        tenant = _tenant("variants")
        per_worker = 8
        pinned = _pinned(runtime, per_worker)
        connections = [
            (pid, connection) for pid in runtime.workers for connection in pinned[pid]
        ]
        path = f"/admin/tenants/{tenant}/signature_variants"
        try:
            answers = _concurrently(
                [
                    lambda index=index, connection=connection: _request(
                        connection,
                        "PUT",
                        f"{path}/agent{index}",
                        {"variant_id": f"variant{index}"},
                    )
                    for index, (_, connection) in enumerate(connections)
                ]
            )
        finally:
            _close(pinned)
        readers = _pinned(runtime, 1)
        try:
            reads = {pid: _request(readers[pid][0], "GET", path) for pid in readers}
        finally:
            _close(readers)

        expected = {f"agent{index}": f"variant{index}" for index in range(len(answers))}
        assert [status for status, _ in answers] == [200] * len(connections)
        # The PUT that landed last answered with every selection.
        assert [body["selections"] for _, body in answers].count(expected) == 1
        assert reads == {
            pid: (200, {"tenant_id": tenant, "selections": expected})
            for pid in runtime.workers
        }
        stored = _record(store, tenant, "signature_variants")
        assert (stored.version, stored.config_value) == (len(connections), expected)

    def test_concurrent_pin_quota_puts_on_both_workers_keep_every_field(
        self, runtime, store
    ):
        tenant = _tenant("pinqrace")
        first, second = runtime.workers
        roles = ["user", "tenant_admin", "org_admin"]
        path = f"/admin/tenants/{tenant}/pin_quotas"
        rounds = 4
        observed = []
        for round_index in range(rounds):
            pinned = _pinned(runtime, 2)
            # The three PUTs of a round alternate workers, shifting each round.
            rotation = [
                pinned[first][0],
                pinned[second][0],
                pinned[first][1],
                pinned[second][1],
            ]
            values = {
                role: 10 * (offset + 1) + round_index
                for offset, role in enumerate(roles)
            }
            try:
                answers = _concurrently(
                    [
                        lambda role=role, connection=rotation[(round_index + offset) % len(rotation)]: (
                            _request(connection, "PUT", path, {role: values[role]})
                        )
                        for offset, role in enumerate(roles)
                    ]
                )
            finally:
                _close(pinned)
            readers = _pinned(runtime, 1)
            try:
                observed.append(
                    (
                        [status for status, _ in answers],
                        values,
                        _request(readers[first][0], "GET", path),
                        _request(readers[second][0], "GET", path),
                    )
                )
            finally:
                _close(readers)

        for statuses, values, read_first, read_second in observed:
            assert statuses == [200, 200, 200]
            assert read_first == (200, {"tenant_id": tenant, "quotas": values})
            assert read_second == (200, {"tenant_id": tenant, "quotas": values})
        stored = _record(store, tenant, "pin_quotas")
        assert (stored.version, stored.config_value) == (
            3 * rounds,
            observed[-1][1],
        )


class TestProfilesAcrossWorkers:
    def test_profile_creates_on_both_workers_all_persist(self, runtime, store):
        tenant = _tenant("profiles")
        per_worker = 3
        pinned = _pinned(runtime, per_worker)
        connections = [c for pid in runtime.workers for c in pinned[pid]]
        names = [f"workers_profile_{index}" for index in range(len(connections))]

        def create(connection, name):
            return _request(
                connection,
                "POST",
                "/admin/profiles",
                {
                    "profile_name": name,
                    "tenant_id": tenant,
                    "type": "video",
                    "schema_name": "video_colpali_smol500_mv_frame",
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "multi_vector",
                    "deploy_schema": False,
                },
            )

        try:
            answers = _concurrently(
                [
                    lambda connection=connection, name=name: create(connection, name)
                    for connection, name in zip(connections, names)
                ]
            )
        finally:
            _close(pinned)
        readers = _pinned(runtime, 1)
        try:
            listed = {
                pid: _request(
                    readers[pid][0], "GET", f"/admin/profiles?tenant_id={tenant}"
                )
                for pid in runtime.workers
            }
        finally:
            _close(readers)

        assert [status for status, _ in answers] == [201] * len(names), answers
        for pid, (status, body) in listed.items():
            assert status == 200, (pid, body)
            assert sorted(
                profile["profile_name"] for profile in body["profiles"]
            ) == sorted(names)
        stored = store.get_config(
            tenant, ConfigScope.BACKEND, "backend", "backend_config"
        )
        assert sorted(stored.config_value["profiles"]) == sorted(names)
        assert stored.version == len(names)


class TestInviteAcrossWorkers:
    def test_an_invite_is_claimed_by_exactly_one_user_across_workers(
        self, runtime, store
    ):
        per_worker = 6
        pinned = _pinned(runtime, per_worker)
        connections = [c for pid in runtime.workers for c in pinned[pid]]
        try:
            status, minted = _request(
                connections[0],
                "POST",
                "/admin/messaging/invite",
                {"tenant_id": "acme:workers"},
            )
            assert status == 200
            token = minted["token"]
            answers = _concurrently(
                [
                    lambda index=index, connection=connection: _request(
                        connection,
                        "POST",
                        "/admin/messaging/register",
                        {
                            "platform": "telegram",
                            "external_user_id": f"user{index}",
                            "token": token,
                        },
                    )
                    for index, connection in enumerate(connections)
                ]
            )
        finally:
            _close(pinned)

        refused = [index for index, (status, _) in enumerate(answers) if status == 404]
        holders = [index for index in range(len(answers)) if index not in refused]
        assert len(holders) == 1, answers
        assert [answers[index] for index in refused] == [
            (404, {"detail": "invalid_token"})
        ] * (len(answers) - 1)
        record = store.get_config(
            "_system:_system",
            ConfigScope.SYSTEM,
            "messaging_gateway",
            f"invite_token_{token}",
        ).config_value
        assert record["claimed_by"] == {
            "platform": "telegram",
            "external_user_id": f"user{holders[0]}",
        }
        # The harness runs no embedding service, so the holder's mapping write
        # cannot initialise Mem0: the token stays claimed for that user and
        # unused, and only that user's retry can complete it.
        assert answers[holders[0]] == (
            503,
            {
                "detail": (
                    "registration unavailable: Failed to lazy-init Mem0 for tenant "
                    "__system__: Mem0 lazy-init requires the 'denseon' inference "
                    "service to be present in system_config.inference_service_urls. "
                    "Available: []; token held for this user"
                )
            },
        )
        assert record["used"] is False


def _worker_pids(worker_ids) -> list[int]:
    """The process ids in cluster-event worker ids (``host:pid:suffix``)."""
    return sorted(int(worker_id.split(":")[-2]) for worker_id in worker_ids)


def _tenant_schemas_in_vespa(config_port: int, tenant_id: str) -> list[str]:
    """The tenant's document types the Vespa config server has deployed."""
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    suffix = "_" + tenant_id.replace(":", "_")
    manager = VespaSchemaManager(
        backend_endpoint="http://localhost", backend_port=config_port
    )
    return sorted(
        name
        for name in manager.list_deployed_document_types(raise_on_failure=True)
        if name.endswith(suffix)
    )


class TestTenantDeleteAcrossWorkers:
    def test_a_tenant_deleted_on_one_worker_is_refused_on_the_other(
        self, runtime, vespa_instance, store
    ):
        name = f"workersdel{uuid.uuid4().hex[:8]}"
        tenant = f"{name}:{name}"
        first, second = runtime.workers
        pinned = _pinned(runtime, 1)
        try:
            created = _request(
                pinned[first][0],
                "POST",
                "/admin/tenants",
                {
                    "tenant_id": tenant,
                    "created_by": "worker-test",
                    "base_schemas": ["provenance"],
                },
            )
        finally:
            _close(pinned)
        assert created[0] == 200, created
        assert created[1]["schemas_deployed"] == ["provenance"]
        assert _tenant_schemas_in_vespa(vespa_instance["config_port"], tenant) == [
            f"provenance_{name}_{name}"
        ]

        pinned = _pinned(runtime, 1)
        try:
            deleted = _request(pinned[first][0], "DELETE", f"/admin/tenants/{tenant}")
            refused = _request(
                pinned[second][0],
                "POST",
                "/admin/profiles/video_colpali_smol500_mv_frame/deploy",
                {"tenant_id": tenant, "force": True},
            )
        finally:
            _close(pinned)

        assert deleted[0] == 200, deleted
        assert deleted[1]["status"] == "deleted"
        assert deleted[1]["deleted_schemas"] == [f"provenance_{name}_{name}"]
        # Both worker processes released the tenant before anything dropped.
        assert _worker_pids(deleted[1]["workers_released"]) == sorted(runtime.workers)
        assert refused == (
            410,
            {
                "detail": (
                    f"Tenant '{tenant}' has been deleted; its schemas and "
                    "memories are not written until the tenant is created again"
                )
            },
        )
        assert _tenant_schemas_in_vespa(vespa_instance["config_port"], tenant) == []
        assert store.get_immutable_config(
            "__system__", ConfigScope.SYSTEM, "tenant_deletions", tenant
        ).config_value == {"deleted": True}


class TestSessionCloseAcrossWorkers:
    def test_a_session_close_is_swept_by_every_worker_of_every_replica(
        self, tmp_path, owned_redis, vespa_instance
    ):
        """Two runtimes of their own on one Redis, as two replicas are: no
        earlier request has warmed a memory manager on any of their workers,
        and a close on one replica is answered by all four workers."""
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        redis_url = owned_redis["url"]
        with (
            _runtime(tmp_path, redis_url, "replica_a") as (process_a, log_a, port_a),
            _runtime(tmp_path, redis_url, "replica_b") as (process_b, log_b, _),
        ):
            replica_a = SimpleNamespace(port=port_a, workers=_serving(process_a, log_a))
            replica_b_workers = _serving(process_b, log_b)
            pinned = _pinned(replica_a, 1)
            try:
                status, body = _request(
                    pinned[replica_a.workers[0]][0],
                    "POST",
                    f"/admin/sessions/{session_id}/close",
                )
            finally:
                _close(pinned)

        assert status == 200, body
        assert _worker_pids(body.pop("workers")) == sorted(
            replica_a.workers + replica_b_workers
        )
        assert body == {
            "status": "closed",
            "session_id": session_id,
            "per_tenant": {},
            "total_deleted": 0,
            "skipped_tenants": [],
        }


class TestProfileRecreateAcrossWorkers:
    def test_a_profile_deleted_on_one_worker_can_be_created_again_on_the_other(
        self, runtime
    ):
        """The other worker has just served the profile from its held config;
        the create's uniqueness check reads the store, so it is not refused."""
        tenant = _tenant("recreate")
        first, second = runtime.workers
        body = {
            "profile_name": "recreated_profile",
            "tenant_id": tenant,
            "type": "video",
            "schema_name": "video_colpali_smol500_mv_frame",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "multi_vector",
            "deploy_schema": False,
        }
        pinned = _pinned(runtime, 1)
        try:
            created = _request(pinned[first][0], "POST", "/admin/profiles", body)
            held = _request(
                pinned[second][0],
                "GET",
                f"/admin/profiles/recreated_profile?tenant_id={tenant}",
            )
            deleted = _request(
                pinned[first][0],
                "DELETE",
                f"/admin/profiles/recreated_profile?tenant_id={tenant}",
            )
            recreated = _request(pinned[second][0], "POST", "/admin/profiles", body)
            duplicate = _request(pinned[first][0], "POST", "/admin/profiles", body)
        finally:
            _close(pinned)

        assert (created[0], held[0], deleted[0]) == (201, 200, 200)
        assert held[1]["profile_name"] == "recreated_profile"
        assert recreated[0] == 201, recreated
        assert duplicate == (
            400,
            {
                "detail": {
                    "message": "Profile validation failed",
                    "errors": [
                        f"Profile 'recreated_profile' already exists for tenant "
                        f"'{tenant}'"
                    ],
                }
            },
        )
