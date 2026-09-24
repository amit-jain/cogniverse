"""Application replacement never drops a peer's schema or its documents.

Every Vespa prepare-and-activate posts the WHOLE application package. Two
processes that each enumerate the live schemas and then post their own package
activate packages missing the other's schemas, and with the content-type
removal override that deletes the peer's documents. These tests run the real
``VespaSchemaManager``/``SchemaRegistry`` in two processes against an owned
Vespa and pin the exact surviving schema set and document contents.
"""

import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from vespa.application import Vespa
from vespa.package import ApplicationPackage

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_vespa.metadata_schemas import add_metadata_schemas_to_package
from cogniverse_vespa.vespa_schema_manager import build_services_config

pytestmark = pytest.mark.integration

BASE_SCHEMA = "agent_memories"
LEASE_SECONDS = 2.0


@pytest.fixture(scope="module")
def vespa_instance(second_vespa):
    """An owned container: these tests replace the whole application package."""
    return second_vespa


def _backend(ports):
    BackendRegistry.clear_instances()
    store = VespaConfigStore(backend_port=ports["http_port"])
    return BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "port": ports["http_port"],
                "config_port": ports["config_port"],
            }
        },
        config_manager=ConfigManager(store=store),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )


def _feed(ports, schema, marker):
    app = Vespa(url=f"http://localhost:{ports['http_port']}")
    deadline = time.monotonic() + 60
    while True:
        responses = []
        app.feed_iterable(
            [{"id": "marker", "fields": {"id": "marker", "text": marker}}],
            schema=schema,
            namespace=schema,
            callback=lambda response, doc_id: responses.append(response.status_code),
        )
        if responses == [200]:
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"Schema {schema} did not accept its marker: {responses}")
        time.sleep(0.5)


def _stored(ports, schema):
    response = Vespa(url=f"http://localhost:{ports['http_port']}").get_data(
        schema=schema,
        namespace=schema,
        data_id="marker",
    )
    assert response.status_code == 200, response.json
    return response.json["fields"]


def _peer_process(ports, tenant, started, finished):
    backend = _backend(ports)
    started.set()
    schema = backend.schema_registry.deploy_schema(tenant, BASE_SCHEMA)
    _feed(ports, schema, "peer document")
    finished.set()


def test_peer_activation_waits_for_this_process_deployment(vespa_instance):
    """A peer process cannot activate between our build and our post."""
    backend = _backend(vespa_instance)
    manager = backend.schema_manager
    reached = threading.Event()
    release = threading.Event()
    original = manager._post_package

    def pause(*args, **kwargs):
        reached.set()
        assert release.wait(120) is True
        return original(*args, **kwargs)

    manager._post_package = pause
    ctx = multiprocessing.get_context("spawn")
    started, finished = ctx.Event(), ctx.Event()
    peer = ctx.Process(
        target=_peer_process, args=(vespa_instance, "serialpeer", started, finished)
    )
    peer_name = manager.get_tenant_schema_name("serialpeer", BASE_SCHEMA)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            startup = executor.submit(manager.upload_metadata_schemas)
            assert reached.wait(60) is True
            peer.start()
            assert started.wait(120) is True
            try:
                assert finished.wait(5) is False
            finally:
                release.set()
            startup.result(timeout=300)
        peer.join(timeout=300)
        assert peer.exitcode == 0
        assert finished.is_set() is True
        assert _stored(vespa_instance, peer_name) == {
            "id": "marker",
            "text": "peer document",
        }
    finally:
        release.set()
        manager._post_package = original
        if peer.is_alive():
            peer.terminate()
            peer.join(timeout=30)


class _ActivationProxy:
    """Inject one conflict after a real independent activation on owned Vespa."""

    def __init__(self, ports, on_conflict):
        upstream = f"http://localhost:{ports['config_port']}"
        proxy = self
        self.conflicts = 0

        class Handler(BaseHTTPRequestHandler):
            def forward(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                # Both deploy funnels: the backend's prepareandactivate and
                # the schema manager's fenced session activation.
                activating = self.path.endswith("/prepareandactivate") or (
                    self.command == "PUT" and self.path.endswith("/active")
                )
                if activating and proxy.conflicts == 0:
                    proxy.conflicts += 1
                    on_conflict()
                    self.send_response(409)
                    self.end_headers()
                    self.wfile.write(b'{"error-code":"ACTIVATION_CONFLICT"}')
                    return
                response = requests.request(
                    self.command,
                    upstream + self.path,
                    data=body,
                    headers={
                        "Content-Type": self.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                    timeout=300,
                )
                self.send_response(response.status_code)
                self.end_headers()
                self.wfile.write(response.content)

            do_GET = forward
            do_POST = forward
            do_PUT = forward
            do_DELETE = forward

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.mark.parametrize("operation", ["startup", "schema", "tenant", "orphan"])
def test_conflict_rebuild_preserves_exact_peer_schema_and_document(
    vespa_instance, operation
):
    """A 409 retry rebuilds the package, so a peer activated meanwhile stays."""
    backend = _backend(vespa_instance)
    manager = backend.schema_manager
    registry = backend.schema_registry
    victim = f"conflict{operation}"
    peer = f"survivor{operation}"
    victim_name = registry.deploy_schema(victim, BASE_SCHEMA)
    _feed(vespa_instance, victim_name, "target document")
    if operation == "orphan":
        registry.unregister_schema(victim, BASE_SCHEMA)
    peer_name = manager.get_tenant_schema_name(peer, BASE_SCHEMA)
    definition = FilesystemSchemaLoader(Path("configs/schemas")).load_schema(
        BASE_SCHEMA
    )
    definition["name"] = peer_name

    def outsider_activation():
        """Another deployer activates the peer schema and feeds it."""
        schemas = manager._get_existing_tenant_schemas()
        if operation == "orphan":
            schemas.append(
                JsonSchemaParser().parse_schema(dict(definition, name=victim_name))
            )
        schemas.append(JsonSchemaParser().parse_schema(definition))
        package = ApplicationPackage(name="cogniverse", schema=schemas)
        add_metadata_schemas_to_package(package)
        package.services_config = build_services_config(package)
        response = requests.post(
            f"http://localhost:{vespa_instance['config_port']}"
            f"/application/v2/tenant/default/prepareandactivate",
            data=package.to_zip().getvalue(),
            headers={"Content-Type": "application/zip"},
            timeout=300,
        )
        assert response.status_code == 200, response.text
        registry.register_schema(peer, BASE_SCHEMA, peer_name, json.dumps(definition))
        _feed(vespa_instance, peer_name, "peer document")

    original_port = manager.backend_port
    with _ActivationProxy(vespa_instance, outsider_activation) as proxy:
        manager.backend_port = proxy.server.server_port
        try:
            if operation == "startup":
                manager.upload_metadata_schemas()
            elif operation == "schema":
                assert manager.delete_schema(victim, BASE_SCHEMA) == victim_name
            elif operation == "tenant":
                assert manager.delete_tenant_schemas(victim) == [victim_name]
            else:
                assert manager.delete_orphan_schemas([victim_name]) == [victim_name]
        finally:
            manager.backend_port = original_port
        assert proxy.conflicts == 1

    assert _stored(vespa_instance, peer_name) == {
        "id": "marker",
        "text": "peer document",
    }
    live = set(manager.list_deployed_document_types(raise_on_failure=True))
    registered = {info.full_schema_name for info in registry._get_all_schemas()}
    assert live == registered | manager._PROTECTED_SCHEMAS
    assert (victim_name in live) is (operation == "startup")


def _partition_lease_store(manager):
    """Cut this process off from the deployment lease record.

    A live holder heartbeats, so only a holder that can no longer reach the
    record may lose it; setting the returned event makes every lease read and
    write this process issues fail as a partition would.
    """
    store = manager._schema_registry._config_manager.store
    partitioned = threading.Event()
    real_get, real_cas = store.get_config, store.compare_and_set_config

    def guard(call):
        def wrapper(*args, **kwargs):
            if partitioned.is_set() and kwargs.get("service") == "schema_deploy_lease":
                raise ConnectionError("config store unreachable")
            return call(*args, **kwargs)

        return wrapper

    store.get_config = guard(real_get)
    store.compare_and_set_config = guard(real_cas)
    return partitioned


def _lease_holder_until_killed(ports, entered):
    """Hold the deployment lease with a short expiry until this process dies.

    Nothing here touches a shared synchronisation primitive after ``entered``,
    so the SIGKILL cannot leave one of them locked for the parent.
    """
    from cogniverse_core.registries import schema_deploy_lease

    schema_deploy_lease.DEFAULT_LEASE_SECONDS = LEASE_SECONDS
    manager = _backend(ports).schema_manager
    with manager.deployment_lease():
        entered.set()
        time.sleep(600)


def _lease_holder(ports, entered, release, result):
    """Hold the deployment lease with a short expiry, then try to activate."""
    from cogniverse_core.registries import schema_deploy_lease

    schema_deploy_lease.DEFAULT_LEASE_SECONDS = LEASE_SECONDS
    manager = _backend(ports).schema_manager
    partitioned = _partition_lease_store(manager)
    try:
        with manager.deployment_lease():
            partitioned.set()
            entered.set()
            if release.wait(600):
                manager.upload_metadata_schemas()
        result.put("activated")
    except RuntimeError as exc:
        result.put(str(exc))


def test_killed_lease_holder_recovers_within_its_expiry(vespa_instance):
    """A holder killed mid-deploy blocks peers only until its lease expires."""
    ctx = multiprocessing.get_context("spawn")
    entered = ctx.Event()
    holder = ctx.Process(
        target=_lease_holder_until_killed, args=(vespa_instance, entered)
    )
    holder.start()
    try:
        assert entered.wait(120) is True
        holder.kill()
        holder.join(timeout=30)
        assert holder.exitcode == -9
        manager = _backend(vespa_instance).schema_manager
        before = time.monotonic()
        with manager.deployment_lease() as lease:
            assert lease is not None
        assert time.monotonic() - before < LEASE_SECONDS + 10
    finally:
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=30)


def test_expired_holder_cannot_activate_after_a_successor_deployed(vespa_instance):
    """A holder whose lease was taken over is refused, not allowed to post."""
    ctx = multiprocessing.get_context("spawn")
    entered, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    holder = ctx.Process(
        target=_lease_holder, args=(vespa_instance, entered, release, result)
    )
    holder.start()
    try:
        assert entered.wait(120) is True
        backend = _backend(vespa_instance)
        name = backend.schema_registry.deploy_schema("leasesuccessor", BASE_SCHEMA)
        _feed(vespa_instance, name, "successor document")
        release.set()
        assert result.get(timeout=300) == (
            "Vespa deployment lease expired or was replaced"
        )
        holder.join(timeout=60)
        assert holder.exitcode == 0
        assert _stored(vespa_instance, name) == {
            "id": "marker",
            "text": "successor document",
        }
    finally:
        release.set()
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=30)
        result.close()


def _stalled_holder(ports, prepared, released, result):
    """Prepare a package that omits a peer's schema, stall past the lease,
    then try to activate it with the schema-removal override enabled."""
    from cogniverse_core.registries import schema_deploy_lease

    schema_deploy_lease.DEFAULT_LEASE_SECONDS = LEASE_SECONDS
    manager = _backend(ports).schema_manager
    partitioned = _partition_lease_store(manager)
    real_post = manager._post_package
    statuses = []
    fences = []

    def post(tenant_url, app_zip, fence=None):
        def pause():
            # Stall where the lease cannot help: the session is created and
            # prepared, this holder is cut off from its lease record so a
            # peer takes it over, and nothing re-checks it before the
            # activate. Only the config server can refuse this activation.
            fences.append(True)
            if len(fences) == 2:
                partitioned.set()
                prepared.set()
                released.wait(600)

        response = real_post(tenant_url, app_zip, fence=pause)
        statuses.append(response.status_code)
        return response

    manager._post_package = post
    try:
        manager.upload_metadata_schemas(allow_schema_removal=True)
        result.put(("activated", statuses))
    except Exception as exc:
        result.put((f"{type(exc).__name__}: {exc}", statuses))


def test_stalled_holder_cannot_activate_a_session_a_successor_outran(vespa_instance):
    """A session prepared before a successor's activation is refused by the
    config server, so the successor's schema and document survive."""
    ctx = multiprocessing.get_context("spawn")
    prepared, released, result = ctx.Event(), ctx.Event(), ctx.Queue()
    holder = ctx.Process(
        target=_stalled_holder, args=(vespa_instance, prepared, released, result)
    )
    holder.start()
    try:
        assert prepared.wait(180) is True
        backend = _backend(vespa_instance)
        name = backend.schema_registry.deploy_schema("leasefence", BASE_SCHEMA)
        _feed(vespa_instance, name, "fenced successor document")
        released.set()

        outcome, statuses = result.get(timeout=300)
        assert statuses == [409]
        assert outcome == (
            "DeploymentLeaseLost: Vespa deployment lease expired or was replaced"
        )
        holder.join(timeout=60)
        assert holder.exitcode == 0

        assert _stored(vespa_instance, name) == {
            "id": "marker",
            "text": "fenced successor document",
        }
        live = set(
            backend.schema_manager.list_deployed_document_types(raise_on_failure=True)
        )
        assert name in live
    finally:
        released.set()
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=30)
        result.close()


def _delay_activation(seconds, paused):
    """Hold every session activation in flight for ``seconds``."""
    import requests as requests_module

    real_put = requests_module.put

    def put(url, *args, **kwargs):
        if url.endswith("/active"):
            paused.set()
            time.sleep(seconds)
        return real_put(url, *args, **kwargs)

    requests_module.put = put


def _slow_activation_holder(ports, paused, result):
    """Deploy with a short hold whose activation outlasts that hold."""
    from cogniverse_core.registries import schema_deploy_lease

    schema_deploy_lease.DEFAULT_LEASE_SECONDS = LEASE_SECONDS
    manager = _backend(ports).schema_manager
    _delay_activation(LEASE_SECONDS * 3, paused)
    try:
        manager.upload_metadata_schemas()
        result.put("activated")
    except Exception as exc:
        result.put(f"{type(exc).__name__}: {exc}")


def test_a_heartbeating_holder_is_not_taken_over_during_a_long_activation(
    vespa_instance,
):
    """A live holder whose activation runs longer than its hold keeps the
    lease the whole time; a peer that waits two holds is refused."""
    ctx = multiprocessing.get_context("spawn")
    paused, result = ctx.Event(), ctx.Queue()
    holder = ctx.Process(
        target=_slow_activation_holder, args=(vespa_instance, paused, result)
    )
    holder.start()
    try:
        assert paused.wait(180) is True
        registry = _backend(vespa_instance).schema_registry
        peer = registry.deployment_lease(wait_seconds=LEASE_SECONDS * 2)
        try:
            with pytest.raises(TimeoutError):
                peer.acquire()
        finally:
            peer.release()
        assert result.get(timeout=300) == "activated"
        holder.join(timeout=60)
        assert holder.exitcode == 0
    finally:
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=30)
        result.close()


def _remote_holder_dying_mid_activation(ports, paused):
    """Deploy with the default hold from a node this host cannot probe, and
    stall inside the activation until killed."""
    import socket

    socket.gethostname = lambda: "remote-deploy-node"
    manager = _backend(ports).schema_manager
    _delay_activation(600, paused)
    manager.upload_metadata_schemas()


def test_a_remote_holder_that_dies_mid_activation_is_taken_over_in_one_wait(
    vespa_instance,
):
    """No pid probe reaches another node, so one default wait must outlast the
    dead holder's default hold."""
    from cogniverse_core.registries import schema_deploy_lease

    ctx = multiprocessing.get_context("spawn")
    paused = ctx.Event()
    holder = ctx.Process(
        target=_remote_holder_dying_mid_activation, args=(vespa_instance, paused)
    )
    holder.start()
    try:
        assert paused.wait(180) is True
        holder.kill()
        holder.join(timeout=30)
        assert holder.exitcode == -9
        registry = _backend(vespa_instance).schema_registry
        lease = registry.deployment_lease()
        started = time.monotonic()
        assert lease.acquire() is lease
        waited = time.monotonic() - started
        lease.release()
        assert schema_deploy_lease.DEFAULT_LEASE_SECONDS <= waited
        assert waited < schema_deploy_lease.DEFAULT_WAIT_SECONDS
    finally:
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=30)


def test_a_holder_whose_renewal_is_refused_aborts_before_preparing(
    vespa_instance, monkeypatch
):
    """Once the config store refuses the heartbeat's renewal the lease is
    lost, and the deploy stops before its next mutating step."""
    import requests as requests_module

    from cogniverse_core.registries import schema_deploy_lease
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    monkeypatch.setattr(schema_deploy_lease, "DEFAULT_LEASE_SECONDS", LEASE_SECONDS)
    manager = _backend(vespa_instance).schema_manager
    store = manager._schema_registry._config_manager.store
    refusing = threading.Event()
    refused = threading.Event()
    real_cas = store.compare_and_set_config

    def compare_and_set_config(*args, **kwargs):
        if refusing.is_set() and kwargs.get("service") == "schema_deploy_lease":
            refused.set()
            return None
        return real_cas(*args, **kwargs)

    monkeypatch.setattr(store, "compare_and_set_config", compare_and_set_config)
    real_post, real_put = requests_module.post, requests_module.put
    steps = []

    def post(url, *args, **kwargs):
        response = real_post(url, *args, **kwargs)
        if url.endswith("/session"):
            steps.append("create")
            refusing.set()
            assert refused.wait(LEASE_SECONDS * 3) is True
        return response

    def put(url, *args, **kwargs):
        steps.append(url.rsplit("/", 1)[-1])
        return real_put(url, *args, **kwargs)

    monkeypatch.setattr(requests_module, "post", post)
    monkeypatch.setattr(requests_module, "put", put)

    with pytest.raises(DeploymentLeaseLost) as caught:
        manager.upload_metadata_schemas()

    assert str(caught.value) == "Vespa deployment lease expired or was replaced"
    assert steps == ["create"]
    assert [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("deploy-lease-heartbeat:")
    ] == []


CAPPED_HOLD_SECONDS = 30.0


def _holder_stuck_mid_activation(ports, paused, acquired_at):
    """Heartbeat a short hold under a short cap, stuck inside the activation."""
    from cogniverse_core.registries import schema_deploy_lease

    schema_deploy_lease.DEFAULT_LEASE_SECONDS = LEASE_SECONDS
    schema_deploy_lease.MAX_TOTAL_HOLD_SECONDS = CAPPED_HOLD_SECONDS
    real_acquire = schema_deploy_lease.SchemaDeployLease.acquire

    def acquire(self):
        lease = real_acquire(self)
        acquired_at.put(time.monotonic())
        return lease

    schema_deploy_lease.SchemaDeployLease.acquire = acquire
    manager = _backend(ports).schema_manager
    _delay_activation(600, paused)
    manager.upload_metadata_schemas()


def test_a_live_holder_stuck_past_the_total_hold_cap_is_taken_over(vespa_instance):
    """Under the cap a stuck-but-live holder keeps the lease; past it the
    heartbeat stops and the holder is taken over while its process lives."""
    ctx = multiprocessing.get_context("spawn")
    paused, acquired_at = ctx.Event(), ctx.Queue()
    holder = ctx.Process(
        target=_holder_stuck_mid_activation,
        args=(vespa_instance, paused, acquired_at),
    )
    holder.start()
    try:
        assert paused.wait(180) is True
        held_from = acquired_at.get(timeout=10)
        registry = _backend(vespa_instance).schema_registry

        probe = registry.deployment_lease(wait_seconds=LEASE_SECONDS * 2)
        with pytest.raises(TimeoutError):
            probe.acquire()
        assert time.monotonic() - held_from < CAPPED_HOLD_SECONDS

        successor = registry.deployment_lease(
            wait_seconds=CAPPED_HOLD_SECONDS + LEASE_SECONDS + 30
        )
        assert successor.acquire() is successor
        taken_over_after = time.monotonic() - held_from
        successor.release()
        assert holder.is_alive() is True
        assert CAPPED_HOLD_SECONDS <= taken_over_after
        assert taken_over_after < CAPPED_HOLD_SECONDS + LEASE_SECONDS + 10
    finally:
        holder.kill()
        holder.join(timeout=30)
        acquired_at.close()
