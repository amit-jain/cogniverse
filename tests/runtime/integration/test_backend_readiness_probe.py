"""Backend-readiness probe and first-install bootstrap in the runtime lifespan.

``_wait_for_backend_ready`` runs at startup inside the async lifespan, so it
must use async HTTP + ``asyncio.sleep`` — a blocking ``httpx.get`` /
``time.sleep`` would freeze the event loop for up to five minutes while Vespa
converges. These tests prove it returns True against a real Vespa and keeps
the loop responsive while retrying an unreachable backend.

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
from types import SimpleNamespace

import pytest

from cogniverse_runtime.main import (
    _bootstrap_metadata_schemas,
    _wait_for_backend_ready,
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


async def test_wait_for_backend_ready_against_real_vespa(vespa_instance):
    vespa_base = f"http://localhost:{vespa_instance['http_port']}"
    ready = await _wait_for_backend_ready(
        vespa_base, max_attempts=12, retry_interval=2.0, timeout=5.0
    )
    assert ready is True


async def test_wait_for_backend_ready_returns_false_when_unreachable():
    ready = await _wait_for_backend_ready(
        "http://127.0.0.1:1", max_attempts=3, retry_interval=0.05, timeout=0.5
    )
    assert ready is False


async def test_wait_for_backend_ready_does_not_block_event_loop():
    stop = asyncio.Event()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    ready = await _wait_for_backend_ready(
        "http://127.0.0.1:1", max_attempts=3, retry_interval=0.05, timeout=0.5
    )
    stop.set()
    await ticker_task

    assert ready is False
    # A blocking time.sleep across the three retries would freeze the loop so
    # the concurrent ticker never advances; async sleep lets it keep ticking.
    assert ticks >= 5


def test_application_exists_false_when_backend_fresh():
    """A fresh config server answers 404 for the application resource — the
    only state in which the metadata bootstrap may deploy."""
    from cogniverse_runtime.main import _application_exists

    with _http_stub(404) as port:
        assert (
            _application_exists("127.0.0.1", port, max_attempts=2, interval=0.05)
            is False
        )


def test_application_exists_true_against_deployed_backend(vespa_instance):
    """The shared Vespa has the metadata application deployed, so the config
    server reports it and the bootstrap must treat the backend as populated."""
    from cogniverse_runtime.main import _application_exists

    assert (
        _application_exists(
            "localhost", vespa_instance["config_port"], max_attempts=3, interval=1.0
        )
        is True
    )


def test_application_exists_raises_when_indeterminate():
    """A config server that answers neither 200 nor 404 leaves fresh-vs-populated
    unknown; deploying blind risks dropping live schemas, so it must raise."""
    from cogniverse_runtime.main import _application_exists

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
