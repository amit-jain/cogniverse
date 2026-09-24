"""A retried Vespa deploy must resend the whole application package.

``_deploy_package`` builds the zip once with ``app_package.to_zip()``, which
returns a ``BytesIO``. ``requests`` reads that stream to EOF on the first
POST, so an ACTIVATION_CONFLICT retry posted an empty body and Vespa
answered ``400 services.xml does not exist in application package``. The
409 retry could therefore never succeed; it only converted a retriable
conflict into a confusing hard failure.

These tests drive a real HTTP server on a real socket. Patching
``requests.post`` cannot catch this class of bug: a mock never reads the
request body, so the stream is never consumed and every retry looks fine.
"""

from __future__ import annotations

import json
import logging
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest
from vespa.package import ApplicationPackage

from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

DEPLOY_PATH = "/application/v2/tenant/default/prepareandactivate"
SESSION_PATH = "/application/v2/tenant/default/session"

CONFLICT_BODY = {
    "error-code": "ACTIVATION_CONFLICT",
    "message": (
        "This session 2002 was prepared when session 2000 was active, but "
        "session 2001 has since become active: refusing to activate this"
    ),
}

EXPECTED_PACKAGE_ENTRIES = {
    "schemas/conflictprobe.sd",
    "search/query-profiles/default.xml",
    "search/query-profiles/types/root.xml",
    "services.xml",
    "validation-overrides.xml",
}


class _ConfigServer:
    """Real config-server stand-in recording every request body.

    Serves both deploy shapes: the single ``prepareandactivate`` the backend
    posts, and the create/prepare/activate session flow the schema manager
    uses so its activation is fenced by the config server. ``statuses`` is
    consumed one entry per activation, whichever shape delivered it.
    """

    def __init__(self, statuses, *, activated_session_id=4242, include_session_id=True):
        self.bodies: list[bytes] = []
        self.paths: list[str] = []
        self.activations = 0
        statuses = list(statuses)
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, status):
                ok_body = {"message": f"Session {activated_session_id} activated."}
                if include_session_id:
                    ok_body["session-id"] = str(activated_session_id)
                payload = (
                    json.dumps(CONFLICT_BODY).encode()
                    if status == 409
                    else json.dumps(ok_body).encode()
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _next_status(self):
                idx = recorder.activations
                recorder.activations += 1
                return statuses[min(idx, len(statuses) - 1)]

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                recorder.bodies.append(self.rfile.read(length))
                recorder.paths.append(self.path)
                if self.path.endswith("/session"):
                    # Session creation always succeeds; the conflict belongs
                    # to the activation.
                    self._answer(200)
                    return
                self._answer(self._next_status())

            def do_PUT(self):
                recorder.paths.append(self.path)
                if self.path.endswith("/prepared"):
                    self._answer(200)
                    return
                self._answer(self._next_status())

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _make_backend(port: int) -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://127.0.0.1"
    backend._port = 8080
    backend._config_port = port
    backend.schema_manager = _make_schema_manager(port)
    return backend


def _make_schema_manager(port: int) -> VespaSchemaManager:
    manager = object.__new__(VespaSchemaManager)
    manager.backend_endpoint = "http://127.0.0.1"
    manager.backend_port = port
    manager._logger = logging.getLogger("test.vsm")
    # No registry, so no config store to lease through: these tests are the
    # single deployer and drive a real config server on a real socket.
    manager._schema_registry = None
    return manager


def _entries(body: bytes) -> set[str]:
    return set(zipfile.ZipFile(BytesIO(body)).namelist())


def test_activation_conflict_retry_resends_the_complete_package():
    """The retry after a 409 must carry the same complete zip, not an
    empty body left over from the consumed stream."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409, 200]) as server:
        backend = _make_backend(server.port)
        generation = backend._deploy_package(app_package)

    assert generation == 4242
    assert len(server.bodies) == 2, (
        f"Expected one conflict then one retry, got {len(server.bodies)} requests"
    )
    assert server.paths == [DEPLOY_PATH, DEPLOY_PATH]
    assert server.bodies[0] == server.bodies[1], (
        "Retry must resend the identical package; retry body was "
        f"{len(server.bodies[1])} bytes vs {len(server.bodies[0])} on the "
        "first attempt"
    )
    assert _entries(server.bodies[1]) == EXPECTED_PACKAGE_ENTRIES


def test_successful_deploy_sends_the_package_once():
    """Pin the happy path: exactly one POST carrying the full package."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([200], activated_session_id=7) as server:
        backend = _make_backend(server.port)
        generation = backend._deploy_package(app_package)

    assert generation == 7
    assert len(server.bodies) == 1
    assert _entries(server.bodies[0]) == EXPECTED_PACKAGE_ENTRIES


def test_activation_without_session_id_raises():
    """A 200 the config server answers without a session-id is not a usable
    activation: the generation is unknown, so the deploy must raise rather
    than convergence-probe against an unknown generation."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([200], include_session_id=False) as server:
        backend = _make_backend(server.port)
        with pytest.raises(RuntimeError) as exc_info:
            backend._deploy_package(app_package)

    assert str(exc_info.value).startswith(
        "Deployment succeeded but the config server response carries no session-id:"
    )


def test_non_retriable_status_raises_after_one_attempt():
    """A 400 is not a conflict: fail immediately, do not burn retries."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([400]) as server:
        backend = _make_backend(server.port)
        with pytest.raises(RuntimeError) as exc_info:
            backend._deploy_package(app_package)

    assert len(server.bodies) == 1
    assert "400" in str(exc_info.value)


def test_conflict_on_every_attempt_exhausts_retries_and_raises():
    """All five attempts carry the full package, then the deploy raises."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409]) as server:
        backend = _make_backend(server.port)
        with pytest.raises(RuntimeError) as exc_info:
            backend._deploy_package(app_package)

    assert len(server.bodies) == 5, f"Expected 5 attempts, got {len(server.bodies)}"
    for i, body in enumerate(server.bodies):
        assert _entries(body) == EXPECTED_PACKAGE_ENTRIES, (
            f"Attempt {i + 1} sent an incomplete package: {_entries(body)}"
        )
    assert "ACTIVATION_CONFLICT" in str(exc_info.value)


def test_schema_manager_retry_rebuilds_and_resends_the_complete_package():
    """VespaSchemaManager._deploy_package builds the package again for every
    attempt — a 409 means someone else's package is now active, so reposting
    the one built before it would drop whatever they added — and each attempt
    must carry the full zip, never a body left over from a consumed stream."""
    built = []

    def build_package():
        built.append(len(built) + 1)
        return ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409, 200]) as server:
        manager = _make_schema_manager(server.port)
        manager._deploy_package(build_package)

    assert built == [1, 2]
    assert len(server.bodies) == 2
    assert [_entries(body) for body in server.bodies] == [
        EXPECTED_PACKAGE_ENTRIES,
        EXPECTED_PACKAGE_ENTRIES,
    ]
    # Each attempt creates its own session and activates that session, so the
    # config server refuses an activation whose session predates a peer's.
    assert server.paths == [
        SESSION_PATH,
        f"{SESSION_PATH}/4242/prepared",
        f"{SESSION_PATH}/4242/active",
        SESSION_PATH,
        f"{SESSION_PATH}/4242/prepared",
        f"{SESSION_PATH}/4242/active",
    ]


class _LosingLease:
    """Owned for ``owned_checks`` ownership checks, then lost."""

    def __init__(self, owned_checks):
        self.owned_checks = owned_checks
        self.checks = 0

    def acquire(self):
        return self

    def ensure_owned(self):
        self.checks += 1
        if self.checks > self.owned_checks:
            raise RuntimeError("Vespa deployment lease expired or was replaced")

    def release(self):
        return None


def test_a_lease_lost_between_prepare_and_activate_never_activates():
    """A holder stalled past its lease must not activate the package it
    prepared. The fence runs immediately before the config server activates,
    so the session is created and prepared and then abandoned."""
    lease = _LosingLease(owned_checks=2)

    with _ConfigServer([200]) as server:
        manager = _make_schema_manager(server.port)
        manager._schema_registry = SimpleNamespace(
            deployment_lease=lambda **kwargs: lease
        )
        with pytest.raises(RuntimeError, match="lease expired or was replaced"):
            manager._deploy_package(lambda: ApplicationPackage(name="conflictprobe"))

    assert lease.checks == 3
    assert server.paths == [SESSION_PATH, f"{SESSION_PATH}/4242/prepared"]
    assert server.activations == 0
    assert [_entries(body) for body in server.bodies] == [EXPECTED_PACKAGE_ENTRIES]


def test_a_lease_lost_after_the_session_is_created_never_prepares():
    """Ownership is re-checked before every mutating step: a lease lost once
    the session exists abandons it before the config server prepares it."""
    lease = _LosingLease(owned_checks=1)

    with _ConfigServer([200]) as server:
        manager = _make_schema_manager(server.port)
        manager._schema_registry = SimpleNamespace(
            deployment_lease=lambda **kwargs: lease
        )
        with pytest.raises(RuntimeError, match="lease expired or was replaced"):
            manager._deploy_package(lambda: ApplicationPackage(name="conflictprobe"))

    assert lease.checks == 2
    assert server.paths == [SESSION_PATH]
    assert server.activations == 0


def test_a_backend_deploy_whose_lease_is_lost_never_retries_the_activation():
    """The single-request prepare-and-activate is fenced before every attempt,
    so a conflict retry never posts once the lease is gone."""
    lease = _LosingLease(owned_checks=1)

    with _ConfigServer([409, 200]) as server:
        backend = _make_backend(server.port)
        backend.schema_manager._schema_registry = SimpleNamespace(
            deployment_lease=lambda **kwargs: lease
        )
        with pytest.raises(RuntimeError, match="lease expired or was replaced"):
            backend._deploy_package(ApplicationPackage(name="conflictprobe"))

    assert lease.checks == 2
    assert server.paths == [DEPLOY_PATH]
    assert server.activations == 1


def test_deploy_fences_check_a_heartbeating_lease_without_store_writes():
    """With the heartbeat renewing, each fence is a local ownership check: the
    deploy path itself writes the lease record only to take and release it."""
    from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
    from tests.utils.memory_store import InMemoryConfigStore

    class _CountingStore(InMemoryConfigStore):
        def __init__(self):
            super().__init__()
            self.deploy_thread_writes = 0

        def compare_and_set_config(self, *args, **kwargs):
            if not threading.current_thread().name.startswith(
                "deploy-lease-heartbeat:"
            ):
                self.deploy_thread_writes += 1
            return super().compare_and_set_config(*args, **kwargs)

    store = _CountingStore()
    with _ConfigServer([200]) as server:
        manager = _make_schema_manager(server.port)
        manager._schema_registry = SimpleNamespace(
            deployment_lease=lambda **kwargs: SchemaDeployLease(store, **kwargs)
        )
        manager._deploy_package(lambda: ApplicationPackage(name="conflictprobe"))

    assert server.paths == [
        SESSION_PATH,
        f"{SESSION_PATH}/4242/prepared",
        f"{SESSION_PATH}/4242/active",
    ]
    assert store.deploy_thread_writes == 2


def _worst_visit_page_seconds(monkeypatch) -> float:
    """One config-store visit page run to its retry bound, measured from the
    store's own retries: every attempt times out on connect and on read."""
    import requests as requests_module

    from cogniverse_sdk.interfaces.config_store import ConfigScope
    from cogniverse_vespa.config import config_store
    from cogniverse_vespa.config.config_store import (
        ConfigStoreUnavailableError,
        VespaConfigStore,
    )

    timeouts: list[float] = []
    backoffs: list[float] = []

    def unreachable(*_args, timeout, **_kwargs):
        timeouts.append(timeout)
        raise requests_module.ConnectionError("config store unreachable")

    monkeypatch.setattr(config_store.requests, "get", unreachable)
    monkeypatch.setattr(config_store.time, "sleep", backoffs.append)
    store = VespaConfigStore(vespa_app=SimpleNamespace(url="http://127.0.0.1:9"))
    with pytest.raises(ConfigStoreUnavailableError):
        store.list_all_configs(scope=ConfigScope.SCHEMA, service="schema_registry")
    monkeypatch.undo()
    return sum(2 * timeout for timeout in timeouts) + sum(backoffs)


def _worst_schema_listing_seconds(monkeypatch) -> float:
    """One config-server schema listing timing out on connect and on read."""
    import requests as requests_module

    timeouts: list[float] = []

    def unreachable(*_args, timeout, **_kwargs):
        timeouts.append(timeout)
        raise requests_module.ConnectionError("config server unreachable")

    monkeypatch.setattr(requests_module, "get", unreachable)
    manager = _make_schema_manager(9)
    with pytest.raises(requests_module.ConnectionError):
        manager.list_deployed_document_types(raise_on_failure=True)
    monkeypatch.undo()
    assert len(timeouts) == 1
    return 2 * timeouts[0]


class _CountingDeleteRegistry:
    """Counts every registry call that is one config-store visit."""

    def __init__(self):
        self.visits = 0

    def deployment_lease(self, **kwargs):
        return SimpleNamespace(
            acquire=lambda: None, ensure_owned=lambda: None, release=lambda: None
        )

    def get_tenant_schemas(self, tenant_id, strict=False):
        self.visits += 1
        return [SimpleNamespace(base_schema_name="conflictprobe")]

    def _get_all_schemas(self, strict=False):
        self.visits += 1
        return [
            SimpleNamespace(
                full_schema_name="conflictprobe_acme_acme",
                schema_definition='{"name": "conflictprobe_acme_acme"}',
            )
        ]

    def reserved_schemas(self, live_names):
        self.visits += 1
        return {}

    def unregister_schema(self, tenant_id, base_schema_name):
        # A tombstone reads the current row before writing the deleted one.
        self.visits += 1
        return None


def test_the_total_hold_cap_covers_the_longest_legitimate_activation(monkeypatch):
    """The heartbeat's cap must outlast the longest lease body: a tenant
    delete whose every read and every deploy request runs to its bound, whose
    first four attempts conflict and whose last activates and tombstones.
    Visits are one page each."""
    from cogniverse_core.registries.schema_deploy_lease import MAX_TOTAL_HOLD_SECONDS
    from cogniverse_vespa import vespa_schema_manager

    visit_page = _worst_visit_page_seconds(monkeypatch)
    listing = _worst_schema_listing_seconds(monkeypatch)
    assert visit_page == 303.75
    assert listing == 20

    registry = _CountingDeleteRegistry()
    listings = []
    backoffs: list[float] = []
    monkeypatch.setattr(vespa_schema_manager.time, "sleep", backoffs.append)
    with _ConfigServer([409, 409, 409, 409, 200]) as server:
        manager = _make_schema_manager(server.port)
        manager._schema_registry = registry
        manager._PROTECTED_SCHEMAS = frozenset()
        manager.get_tenant_schema_name = lambda tid, base: f"{base}_acme_acme"

        def list_deployed(raise_on_failure=False):
            listings.append(raise_on_failure)
            return ["conflictprobe_acme_acme"]

        manager.list_deployed_document_types = list_deployed
        assert manager.delete_tenant_schemas("acme") == ["conflictprobe_acme_acme"]

    requests_made = len(server.paths)
    assert (requests_made, registry.visits, len(listings)) == (15, 13, 7)
    longest = (
        requests_made * sum(vespa_schema_manager.DEPLOY_REQUEST_TIMEOUT_S)
        + sum(backoffs)
        + registry.visits * visit_page
        + len(listings) * listing
    )
    assert longest == 8746.25
    assert MAX_TOTAL_HOLD_SECONDS >= longest
