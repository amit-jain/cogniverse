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
                if self.path.endswith("/prepareandactivate") and proxy.conflicts == 0:
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
    try:
        with manager.deployment_lease():
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
