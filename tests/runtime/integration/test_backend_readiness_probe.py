"""Backend-readiness probes and first-install bootstrap.

``_wait_for_backend_startup`` uses async HTTP + ``asyncio.sleep``. These tests prove it distinguishes a deployed
feed from a fresh config server, detects the fresh state without consuming the
retry budget, and keeps the event loop responsive while both planes are down.

``_bootstrap_metadata_schemas`` runs when the startup config read fails. That
read also fails transiently on a POPULATED backend (slow cold start, degraded
query), so the bootstrap must refuse to deploy unless the config server says
no application exists — a registry-less metadata-only deploy over a populated
backend would drop every tenant content schema and let Vespa GC their
documents. These tests pin both guards against a real Vespa: the
application-exists refusal, and Vespa rejecting a partial package when schema
removal is disabled.
"""

import asyncio
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.backend_startup import (
    BackendStartupState,
    _bootstrap_metadata_schemas,
    _wait_for_backend_startup,
    _wait_for_config_server,
)

pytestmark = pytest.mark.integration


@contextmanager
def _http_stub(status: int):
    """Serve ``status`` for every GET on an ephemeral local port."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()


_METADATA_SCHEMA_FILES = {
    "organization_metadata.sd",
    "tenant_metadata.sd",
    "config_metadata.sd",
    "adapter_registry.sd",
}


def _deployed_schema_files(config_port: int) -> set[str]:
    """Names of the .sd files in the ACTIVE application package — the
    authoritative record of which schemas exist. A schema absent here has been
    removed from the content cluster and its documents garbage-collected."""
    import httpx

    resp = httpx.get(
        f"http://localhost:{config_port}"
        "/application/v2/tenant/default/application/default"
        "/environment/prod/region/default/instance/default/content/schemas/",
        timeout=10,
    )
    assert resp.status_code == 200, resp.text
    return {entry.rstrip("/").rsplit("/", 1)[-1] for entry in resp.json()}


def test_wait_for_config_server_true_when_port_accepts():
    """A cold Vespa opens its query port before its config/deploy server, so
    the metadata bootstrap waits for the config server to accept connections
    rather than deploying blind and crash-looping the whole runtime."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        assert _wait_for_config_server("127.0.0.1", port, max_attempts=1) is True
    finally:
        listener.close()


def test_wait_for_config_server_false_when_refused():
    # Bind then close so the port is definitely free (connection refused),
    # and cap attempts so the bounded wait returns quickly.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert (
        _wait_for_config_server("127.0.0.1", port, max_attempts=3, interval=0.05)
        is False
    )


async def test_wait_for_backend_startup_against_real_vespa(vespa_instance):
    vespa_base = f"http://localhost:{vespa_instance['http_port']}"
    state = await _wait_for_backend_startup(
        vespa_base,
        f"http://localhost:{vespa_instance['config_port']}",
        budget_s=24.0,
        retry_interval=2.0,
        timeout=5.0,
    )
    assert state is BackendStartupState.FEED_READY


@pytest.mark.no_shared_vespa
async def test_wait_for_backend_startup_detects_fresh_config_server_immediately():
    started = asyncio.get_running_loop().time()
    with _http_stub(404) as config_port:
        state = await _wait_for_backend_startup(
            "http://127.0.0.1:1",
            f"http://127.0.0.1:{config_port}",
            budget_s=300.0,
            retry_interval=5.0,
            timeout=0.5,
        )

    assert state is BackendStartupState.FRESH_INSTALL
    assert asyncio.get_running_loop().time() - started < 1.0


@pytest.mark.no_shared_vespa
async def test_wait_for_backend_startup_returns_unavailable_when_both_planes_are_down():
    state = await _wait_for_backend_startup(
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
        budget_s=0.15,
        retry_interval=0.05,
        timeout=0.5,
    )
    assert state is BackendStartupState.UNAVAILABLE


@pytest.mark.no_shared_vespa
async def test_wait_for_backend_startup_spends_its_whole_wall_clock_budget(caplog):
    """Both planes refuse connections instantly, so attempt count cannot bound
    the wait: only the wall-clock budget does. The loop must keep polling
    until the budget elapses and report progress against that budget."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level("INFO", logger="cogniverse_runtime.backend_startup"):
        state = await _wait_for_backend_startup(
            "http://127.0.0.1:1",
            "http://127.0.0.1:2",
            budget_s=1.0,
            retry_interval=0.1,
            timeout=0.5,
        )
    elapsed = loop.time() - started

    assert state is BackendStartupState.UNAVAILABLE
    assert 1.0 <= elapsed < 1.5
    retry_lines = [
        r.message for r in caplog.records if r.message.startswith("Backend not ready")
    ]
    assert 8 <= len(retry_lines) <= 11
    assert (
        retry_lines[0] == "Backend not ready, retrying (attempt 1, 0s of 1s budget)..."
    )
    assert retry_lines[-1].startswith("Backend not ready, retrying (attempt ")
    assert retry_lines[-1].endswith("s of 1s budget)...")


@contextmanager
def _hung_tcp_listener():
    """Accept connections and never answer: the shape of a paused backend."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.mark.no_shared_vespa
async def test_wait_for_backend_startup_overruns_budget_by_at_most_two_probe_timeouts():
    """A hung backend makes every probe take its full timeout; the last
    attempt may start just before the deadline, so the wall-clock overrun is
    bounded by the two per-attempt probes (data plane + config server). The
    chart's startupProbe window is sized from exactly this bound."""
    loop = asyncio.get_running_loop()
    with _hung_tcp_listener() as data_port, _hung_tcp_listener() as config_port:
        started = loop.time()
        state = await _wait_for_backend_startup(
            f"http://127.0.0.1:{data_port}",
            f"http://127.0.0.1:{config_port}",
            budget_s=0.5,
            retry_interval=0.05,
            timeout=0.4,
        )
        elapsed = loop.time() - started

    assert state is BackendStartupState.UNAVAILABLE
    assert 0.5 <= elapsed <= 0.5 + 2 * 0.4 + 0.1


@pytest.mark.no_shared_vespa
async def test_wait_for_backend_startup_does_not_block_event_loop():
    stop = asyncio.Event()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    state = await _wait_for_backend_startup(
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
        budget_s=0.15,
        retry_interval=0.05,
        timeout=0.5,
    )
    stop.set()
    await ticker_task

    assert state is BackendStartupState.UNAVAILABLE
    # A blocking time.sleep across the three retries would freeze the loop so
    # the concurrent ticker never advances; async sleep lets it keep ticking.
    assert ticks >= 5


def test_application_exists_false_when_backend_fresh():
    """A fresh config server answers 404 for the application resource — the
    only state in which the metadata bootstrap may deploy."""
    from cogniverse_runtime.backend_startup import _application_exists

    with _http_stub(404) as port:
        assert (
            _application_exists("127.0.0.1", port, max_attempts=2, interval=0.05)
            is False
        )


def test_application_exists_true_against_deployed_backend(vespa_instance):
    """The shared Vespa has the metadata application deployed, so the config
    server reports it and the bootstrap must treat the backend as populated."""
    from cogniverse_runtime.backend_startup import _application_exists

    assert (
        _application_exists(
            "localhost", vespa_instance["config_port"], max_attempts=3, interval=1.0
        )
        is True
    )


def test_application_exists_raises_when_indeterminate():
    """A config server that answers neither 200 nor 404 leaves fresh-vs-populated
    unknown; deploying blind risks dropping live schemas, so it must raise."""
    from cogniverse_runtime.backend_startup import _application_exists

    with _http_stub(503) as port:
        with pytest.raises(RuntimeError, match="refusing to bootstrap"):
            _application_exists("127.0.0.1", port, max_attempts=2, interval=0.05)


def test_bootstrap_never_removes_deployed_content_schemas(vespa_instance, monkeypatch):
    """A populated backend whose config read failed transiently must keep every
    content schema when the first-install bootstrap fires.

    The registry-less bootstrap builds a metadata-only package; deployed over a
    populated cluster it removes the content schemas and Vespa garbage-collects
    their documents. The bootstrap must instead detect the deployed application
    and raise, leaving the active package untouched.
    """
    before = _deployed_schema_files(vespa_instance["config_port"])
    content_schemas = before - _METADATA_SCHEMA_FILES
    assert content_schemas, (
        f"fixture should carry tenant content schemas, got only {before}"
    )

    monkeypatch.setattr(
        "cogniverse_vespa.config_utils.calculate_config_port",
        lambda port: vespa_instance["config_port"],
    )
    bootstrap = SimpleNamespace(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )

    raised: Exception | None = None
    try:
        _bootstrap_metadata_schemas(bootstrap, "cogniverse")
    except RuntimeError as exc:
        raised = exc

    after = _deployed_schema_files(vespa_instance["config_port"])
    assert after == before, (
        f"metadata bootstrap removed deployed schemas: {before - after}"
    )
    assert raised is not None and "already has an application" in str(raised)


def test_upload_metadata_schemas_removal_disabled_refuses_partial_package(
    vespa_instance,
):
    """With schema removal disabled, Vespa itself must refuse a metadata-only
    package that lacks the deployed content schemas — the deploy fails and the
    active package keeps every schema. This is the backstop if fresh-detection
    is ever wrong."""
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    before = _deployed_schema_files(vespa_instance["config_port"])
    assert before - _METADATA_SCHEMA_FILES

    manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=vespa_instance["config_port"],
        schema_registry=None,
    )
    with pytest.raises(Exception, match="schema-removal|validation-override"):
        manager.upload_metadata_schemas(
            app_name="cogniverse", allow_schema_removal=False
        )

    assert _deployed_schema_files(vespa_instance["config_port"]) == before


def test_metadata_schemas_current_compares_the_live_definitions(
    vespa_instance, monkeypatch
):
    """Startup skips the metadata redeploy only when every metadata schema
    the config server serves equals the one this build deploys."""
    from vespa.package import Field

    import cogniverse_vespa.metadata_schemas as metadata_schemas
    from cogniverse_runtime.backend_startup import metadata_schemas_current
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    def manager(config_port: int) -> VespaSchemaManager:
        return VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=config_port,
            schema_registry=None,
        )

    live = manager(vespa_instance["config_port"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    current = metadata_schemas_current(live)
    unreachable = metadata_schemas_current(manager(closed_port))

    shipped = metadata_schemas.create_config_metadata_schema

    def drifted():
        schema = shipped()
        schema.add_fields(
            Field(name="drift_probe", type="string", indexing=["attribute"])
        )
        return schema

    monkeypatch.setattr(metadata_schemas, "create_config_metadata_schema", drifted)
    after_drift = metadata_schemas_current(live)

    assert (current, unreachable, after_drift) == (True, False, False)


@pytest.fixture
def registry_vespa(seeded_config_vespa):
    """``connect(tenant, loader)``: a VespaBackend wired to a real
    SchemaRegistry over the shared Vespa's config store, as startup builds."""
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_vespa.backend import VespaBackend
    from cogniverse_vespa.config.config_store import VespaConfigStore

    ports = seeded_config_vespa
    store = VespaConfigStore(
        backend_url="http://127.0.0.1", backend_port=ports["http_port"]
    )
    backends = []

    def connect(tenant_id, loader=None):
        loader = loader or FilesystemSchemaLoader(Path("configs/schemas"))
        manager = ConfigManager(store=store)
        backend = VespaBackend(
            BackendConfig(
                backend_type="vespa",
                url="http://127.0.0.1",
                port=ports["http_port"],
                tenant_id=tenant_id,
            ),
            schema_loader=loader,
            config_manager=manager,
        )
        backend._initialize_backend({"config_port": ports["config_port"]})
        backend.schema_registry = SchemaRegistry(manager, backend, loader)
        backend.schema_manager._schema_registry = backend.schema_registry
        backends.append(backend)
        return backend

    yield connect, store
    for backend in backends:
        backend.close()
    store.close()


@pytest.mark.asyncio
async def test_the_startup_migration_retries_a_held_lease_and_names_a_refused_tenant(
    registry_vespa, monkeypatch, caplog
):
    """The runtime's startup caller over a real SchemaRegistry on real Vespa:
    a peer's deployment lease makes the migration wait out and retry, and a
    tenant whose redeploy Vespa refuses is logged by tenant and schema while
    the other drifted tenant is redeployed."""
    from uuid import uuid4

    from cogniverse_core.registries import schema_deploy_lease
    from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
    from cogniverse_runtime import main as runtime_main
    from tests.core.integration.test_provenance_schema_lifecycle import (
        _IncompatiblePreDigestLoader,
        _PreDigestLoader,
    )

    connect, store = registry_vespa
    drifted = f"startupmig_{uuid4().hex[:10]}:drifted"
    refused = f"startupmig_{uuid4().hex[:10]}:refused"
    schema_dir = Path("configs/schemas")
    drifted_schema = connect(
        drifted, _PreDigestLoader(schema_dir)
    ).schema_registry.deploy_schema(drifted, "provenance")
    refused_owner = connect(refused, _IncompatiblePreDigestLoader(schema_dir))
    refused_schema = refused_owner.schema_registry.deploy_schema(refused, "provenance")
    registry = connect(drifted).schema_registry

    peer = SchemaDeployLease(store, heartbeat=True)
    assert peer.acquire() is peer
    monkeypatch.setattr(schema_deploy_lease, "DEFAULT_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(runtime_main, "METADATA_DEPLOY_RETRY_SECONDS", 1.0)
    caplog.set_level("INFO", logger=runtime_main.logger.name)

    def ours(record) -> bool:
        return record.name == runtime_main.logger.name

    migration = asyncio.create_task(
        runtime_main._migrate_drifted_schemas(lambda: registry, "provenance")
    )
    try:
        async with asyncio.timeout(60):
            while not any(
                ours(record) and record.levelname == "WARNING"
                for record in caplog.records
            ):
                await asyncio.sleep(0.1)
    finally:
        peer.release()
    try:
        await asyncio.wait_for(migration, timeout=600)
    finally:
        refused_owner.schema_manager.delete_schema(refused, "provenance")

    warnings = [
        record.getMessage()
        for record in caplog.records
        if ours(record) and record.levelname == "WARNING"
    ]
    assert warnings == [
        "Migration of drifted provenance schemas did not get the deployment lease "
        f"(Vespa deployment lease still held by {peer.holder!r} after 2.0s; "
        "refusing to replace the application package concurrently with another "
        "deployer); retrying in 1s"
    ]
    [redeployed] = [
        record.args[1]
        for record in caplog.records
        if ours(record) and record.levelname == "INFO"
    ]
    assert drifted_schema in redeployed
    assert refused_schema not in redeployed
    refusals = [
        record.getMessage()
        for record in caplog.records
        if ours(record)
        and record.levelname == "ERROR"
        and refused in record.getMessage()
    ]
    assert len(refusals) == 1
    assert refusals[0].startswith(
        f"Migration of drifted provenance schemas could not redeploy {refused_schema} "
        f"for tenant {refused}: "
    )
    assert "Vespa refused the application package" in refusals[0]
