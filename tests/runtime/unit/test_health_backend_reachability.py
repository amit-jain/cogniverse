"""Readiness / health must reflect real backend connectivity.

``BackendRegistry.list_backends()`` counts backend *class* registrations, and
the Vespa backend self-registers at import — so it is never empty. Before this
fix both /health and /health/ready reported healthy/ready with Vespa completely
down, so k8s routed traffic to a runtime whose every query failed. The probes
now ping the backend's ``/ApplicationStatus``.
"""

from __future__ import annotations

import http.server
import socket
import threading
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Importing the vespa backend registers the "vespa" class so list_backends()
# is non-empty — exactly the condition that made the old probes false-ready.
import cogniverse_vespa.backend  # noqa: F401
from cogniverse_runtime.routers import health


class _OKHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/ApplicationStatus":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # silence stderr spam
        pass


@contextmanager
def _stub_backend():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _OKHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def _dead_url() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    return f"http://127.0.0.1:{port}"


def _client(base_url) -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    if base_url is not None:
        app.state.backend_base_url = base_url
    # Isolate the reachability behaviour from config-driven registry assembly
    # and from another test's cached probe result / readiness-grace state.
    getattr(health._get_agent_registry, "cache_clear", lambda: None)()
    health._reset_probe_state()
    return TestClient(app)


@pytest.mark.unit
class TestReadinessReflectsBackendReachability:
    def test_ready_when_backend_answers(self, monkeypatch):
        with _stub_backend() as base:
            resp = _client(base).get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_not_ready_when_backend_unreachable(self):
        resp = _client(_dead_url()).get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert "unreachable" in body["reason"].lower()

    def test_not_ready_before_startup_wires_backend(self):
        resp = _client(None).get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


@pytest.mark.unit
class TestHealthReflectsBackendReachability:
    def test_health_unhealthy_when_backend_unreachable(self, monkeypatch):
        # Stub the registry assembly so the reachability probe is the decider,
        # not a config-load failure.
        monkeypatch.setattr(
            health,
            "_get_agent_registry",
            lambda: type("R", (), {"list_agents": lambda self: []})(),
        )
        resp = _client(_dead_url()).get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"

    def test_health_healthy_when_backend_reachable(self, monkeypatch):
        monkeypatch.setattr(
            health,
            "_get_agent_registry",
            lambda: type("R", (), {"list_agents": lambda self: []})(),
        )
        with _stub_backend() as base:
            resp = _client(base).get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.mark.unit
class TestReadinessFlapResistance:
    """A backend probe blip must not eject every replica from the Service.

    When Vespa's tail latency exceeds the probe timeout under load, all pods
    fail the probe simultaneously; without a grace window the Service loses
    all endpoints and a slow backend becomes a total outage. Readiness rides
    out a blip that follows a recent success; /health stays strict so
    monitoring goes red; a genuine outage outlasts the grace and still flips
    the pod not-ready. Cold starts get no grace — a pod that never reached
    the backend must not serve.
    """

    def test_blip_after_success_keeps_ready_and_health_red(self, monkeypatch):
        monkeypatch.setattr(
            health,
            "_get_agent_registry",
            lambda: type("R", (), {"list_agents": lambda self: []})(),
        )
        with _stub_backend() as base:
            client = _client(base)
            assert client.get("/health/ready").status_code == 200
        # Backend gone; expire the probe cache so the next hit re-probes.
        monkeypatch.setattr(health, "_PROBE_TTL_S", 0.0)
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["backend_degraded"] is True
        strict = client.get("/health")
        assert strict.status_code == 503
        assert strict.json()["status"] == "unhealthy"

    def test_outage_beyond_grace_flips_not_ready(self, monkeypatch):
        import time

        with _stub_backend() as base:
            client = _client(base)
            assert client.get("/health/ready").status_code == 200
        monkeypatch.setattr(health, "_PROBE_TTL_S", 0.0)
        monkeypatch.setattr(health, "_READY_GRACE_S", 0.05)
        time.sleep(0.1)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

    def test_cold_start_failure_gets_no_grace(self):
        resp = _client(_dead_url()).get("/health/ready")
        assert resp.status_code == 503

    def test_probe_result_shared_across_probes_within_ttl(self, monkeypatch):
        monkeypatch.setattr(
            health,
            "_get_agent_registry",
            lambda: type("R", (), {"list_agents": lambda self: []})(),
        )
        _CountingHandler.hits = 0
        srv = http.server.HTTPServer(("127.0.0.1", 0), _CountingHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            client = _client(f"http://127.0.0.1:{srv.server_address[1]}")
            for _ in range(4):
                assert client.get("/health/ready").status_code == 200
                assert client.get("/health").status_code == 200
        finally:
            srv.shutdown()
        # Eight endpoint hits inside one TTL window -> one upstream probe.
        assert _CountingHandler.hits == 1
