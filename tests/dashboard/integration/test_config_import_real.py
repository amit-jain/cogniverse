"""Dashboard uploads preserve the selected tenant at the Vespa boundary."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration


def _store(port):
    return VespaConfigStore(backend_url="http://127.0.0.1", backend_port=port)


def _upload_app(port, tenant, payload):
    app = AppTest.from_string(
        """
import streamlit as st
from types import SimpleNamespace
from cogniverse_dashboard.tabs.config_management import render_import_export_ui
render_import_export_ui(SimpleNamespace(store=st.session_state['store']),
                        st.session_state['tenant'])
""",
        default_timeout=60,
    )
    app.session_state["store"] = _store(port)
    app.session_state["tenant"] = tenant
    app.run()
    app.file_uploader[0].set_value(
        ("configs.json", json.dumps(payload).encode(), "application/json")
    ).run()
    next(b for b in app.button if b.label == "📤 Import Configurations").click().run()
    return app


@contextmanager
def _vespa_proxy(upstream, *, barrier=None, fail_key=None):
    state = SimpleNamespace(writes=[], failures=0, lock=threading.Lock())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._forward()

        def do_POST(self):
            self._forward()

        def do_PUT(self):
            self._forward()

        def _forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.command in {"POST", "PUT"} and "/document/v1/" in self.path:
                fields = json.loads(body)["fields"]
                with state.lock:
                    state.writes.append((fields["tenant_id"], fields["config_key"]))
                if barrier is not None:
                    barrier.wait(timeout=30)
                if fields["config_key"] == fail_key:
                    state.failures += 1
                    content = b'{"message":"config write rejected during import"}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
            with httpx.Client(timeout=30) as client:
                response = client.request(
                    self.command,
                    upstream + self.path,
                    content=body,
                    headers={
                        key: value
                        for key, value in self.headers.items()
                        if key.lower()
                        not in {
                            "host",
                            "content-length",
                            "connection",
                            "transfer-encoding",
                        }
                    },
                )
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _payload(source, entries):
    return {
        "tenant_id": source,
        "configs": [
            {
                "tenant_id": source,
                "scope": "agent",
                "service": "search_agent",
                "config_key": key,
                "config_value": value,
            }
            for key, value in entries.items()
        ],
    }


def test_import_upload_uses_selected_tenant_and_preserves_source(shared_vespa):
    store = _store(shared_vespa["http_port"])
    source, target = [f"import{uuid4().hex[:8]}:tenant" for _ in range(2)]
    expected = {"model": "selected-model", "top_k": 7}
    store.set_config(source, ConfigScope.AGENT, "search_agent", "settings", expected)
    payload = store.export_configs(source)
    app = _upload_app(shared_vespa["http_port"], target, payload)
    assert [e.value for e in app.error] == []
    assert [e.message for e in app.exception] == []
    assert (
        store.get_config(
            target, ConfigScope.AGENT, "search_agent", "settings"
        ).config_value
        == expected
    )
    assert (
        store.get_config(
            source, ConfigScope.AGENT, "search_agent", "settings"
        ).config_value
        == expected
    )
    assert len(store.list_configs(target)) == 1
    assert len(store.list_configs(source)) == 1


def test_concurrent_import_uploads_keep_each_destination(shared_vespa):
    """Two operators restoring the same export into different tenants.

    The destination belongs to the call, not to the store, so two writes
    overlapping at the Vespa write barrier must land one document each,
    under their own tenant, and leave the export's source tenant empty.
    """
    source = f"source{uuid4().hex[:8]}:tenant"
    tenants = [f"import{uuid4().hex[:8]}:tenant" for _ in range(2)]
    barrier = threading.Barrier(2)
    with _vespa_proxy(shared_vespa["base_url"], barrier=barrier) as (port, state):
        store = _store(port)

        def restore(tenant):
            return store.import_configs(
                tenant_id=tenant,
                configs=_payload(source, {"settings": {"owner": tenant}}),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(restore, tenants)) == [1, 1]
        assert sorted(state.writes) == sorted(
            (tenant, "settings") for tenant in tenants
        )
    store = _store(shared_vespa["http_port"])
    for tenant in tenants:
        assert store.get_config(
            tenant, ConfigScope.AGENT, "search_agent", "settings"
        ).config_value == {"owner": tenant}
    assert store.list_configs(source) == []


def test_import_write_failure_is_visible_without_success(shared_vespa):
    target = f"import{uuid4().hex[:8]}:tenant"
    payload = _payload("file:tenant", {"first": {"value": 1}, "second": {"value": 2}})
    with _vespa_proxy(shared_vespa["base_url"], fail_key="second") as (port, state):
        app = _upload_app(port, target, payload)
        assert len(app.error) == 1
        assert app.error[0].value.startswith(
            f"Import failed: Failed to import 1 of 2 configurations for tenant {target}: second:"
        )
        assert [s.value for s in app.success] == []
        assert state.writes == [(target, "first"), (target, "second")]
        assert state.failures == 1
    store = _store(shared_vespa["http_port"])
    assert store.get_config(
        target, ConfigScope.AGENT, "search_agent", "first"
    ).config_value == {"value": 1}
    assert store.get_config(target, ConfigScope.AGENT, "search_agent", "second") is None
