"""Dashboard uploads preserve the selected tenant at the Vespa boundary."""

from __future__ import annotations

import json
import multiprocessing
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from cogniverse_core.registries.schema_registry import (
    SCHEMA_REGISTRY_SERVICE,
    SchemaRegistry,
)
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


def _child_import(port, tenant, payload, output):
    """Drive one upload in its own process.

    ``AppTest`` drives the single global Streamlit runtime, so two uploads
    overlap only across processes; the write barrier they meet at lives in
    the proxy in the parent.
    """
    app = _upload_app(port, tenant, payload)
    output.put(
        {
            "tenant": tenant,
            "errors": [e.value for e in app.error],
            "exceptions": [e.message for e in app.exception],
        }
    )


@contextmanager
def _children_start_from_this_module():
    """Keep a spawned child out of the parent's ``__main__``.

    ``AppTest`` replaces ``sys.modules["__main__"]`` with the temp script it
    renders, and a spawned child re-runs that path before its target; the
    script fails outside a Streamlit session. Without the path a child
    starts from ``_child_import`` alone.
    """
    main = sys.modules["__main__"]
    path = getattr(main, "__file__", None)
    if path is not None:
        del main.__file__
    try:
        yield
    finally:
        if path is not None:
            main.__file__ = path


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
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    barrier = threading.Barrier(2)
    with _vespa_proxy(shared_vespa["base_url"], barrier=barrier) as (port, state):
        children = [
            context.Process(
                target=_child_import,
                args=(
                    port,
                    tenant,
                    _payload(source, {"settings": {"owner": tenant}}),
                    output,
                ),
            )
            for tenant in tenants
        ]
        try:
            with _children_start_from_this_module():
                for child in children:
                    child.start()
            results = [output.get(timeout=180) for _ in children]
            for child in children:
                child.join(timeout=10)
            assert [child.exitcode for child in children] == [0, 0]
            assert sorted(results, key=lambda row: row["tenant"]) == sorted(
                (
                    {"tenant": tenant, "errors": [], "exceptions": []}
                    for tenant in tenants
                ),
                key=lambda row: row["tenant"],
            )
            assert sorted(state.writes) == sorted(
                (tenant, "settings") for tenant in tenants
            )
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
    store = _store(shared_vespa["http_port"])
    for tenant in tenants:
        assert store.get_config(
            tenant, ConfigScope.AGENT, "search_agent", "settings"
        ).config_value == {"owner": tenant}
        assert len(store.list_configs(tenant)) == 1
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


def _register_schema(store, tenant, base_schema):
    """File a deployment in the registry the way the schema registry does."""
    registry = SchemaRegistry(
        SimpleNamespace(store=store), backend=object(), schema_loader=object()
    )
    registry.register_schema(
        tenant_id=tenant,
        base_schema_name=base_schema,
        full_schema_name=_full_schema_name(base_schema, tenant),
        schema_definition=json.dumps({"name": _full_schema_name(base_schema, tenant)}),
    )


def _full_schema_name(base_schema, tenant):
    return f"{base_schema}_{tenant.replace(':', '_')}"


def _registry_rows(store, tenants):
    """(row tenant, key, tenant the row names, schema the row names) per
    registry row under ``tenants``, as the schema registry reads them."""
    return sorted(
        (
            row.tenant_id,
            row.config_key,
            row.config_value["tenant_id"],
            row.config_value["full_schema_name"],
        )
        for row in store.list_all_configs(
            scope=ConfigScope.SCHEMA, service=SCHEMA_REGISTRY_SERVICE
        )
        if row.tenant_id in tenants
    )


def test_an_export_restores_configurations_without_the_sources_deployments(
    shared_vespa,
):
    store = _store(shared_vespa["http_port"])
    source, target = [f"import{uuid4().hex[:8]}:tenant" for _ in range(2)]
    base_schema = "document_text"
    _register_schema(store, source, base_schema)
    _register_schema(store, target, base_schema)
    store.set_config(source, ConfigScope.AGENT, "search_agent", "settings", {"k": 3})

    payload = store.export_configs(source)
    assert [
        (entry["scope"], entry["service"], entry["config_key"])
        for entry in payload["configs"]
    ] == [("agent", "search_agent", "settings")]

    app = _upload_app(shared_vespa["http_port"], target, payload)
    assert [e.value for e in app.error] == []
    assert [e.message for e in app.exception] == []
    assert store.get_config(
        target, ConfigScope.AGENT, "search_agent", "settings"
    ).config_value == {"k": 3}
    assert sorted(
        (entry.scope.value, entry.service, entry.config_key)
        for entry in store.list_configs(target)
    ) == [
        ("agent", "search_agent", "settings"),
        ("schema", SCHEMA_REGISTRY_SERVICE, f"schema_{base_schema}"),
    ]
    # Each tenant's registry names only that tenant's own deployment.
    assert _registry_rows(store, {source, target}) == sorted(
        [
            (
                tenant,
                f"schema_{base_schema}",
                tenant,
                _full_schema_name(base_schema, tenant),
            )
            for tenant in (source, target)
        ]
    )


def test_an_upload_carrying_schema_rows_is_refused_before_any_write(shared_vespa):
    source, target = [f"import{uuid4().hex[:8]}:tenant" for _ in range(2)]
    payload = _payload(source, {"settings": {"k": 3}})
    payload["configs"].append(
        {
            "tenant_id": source,
            "scope": "schema",
            "service": SCHEMA_REGISTRY_SERVICE,
            "config_key": "schema_document_text",
            "config_value": {
                "tenant_id": source,
                "base_schema_name": "document_text",
                "full_schema_name": _full_schema_name("document_text", source),
                "schema_definition": "{}",
                "config": {},
                "deployment_time": "2026-09-17T00:00:00+00:00",
            },
        }
    )
    with _vespa_proxy(shared_vespa["base_url"]) as (port, state):
        app = _upload_app(port, target, payload)
        assert [e.value for e in app.error] == [
            f"Import failed: Configuration import for tenant {target} refused: "
            "schema rows record deployments made by the schema registry and are "
            f"not importable: {SCHEMA_REGISTRY_SERVICE}/schema_document_text"
        ]
        assert [s.value for s in app.success] == []
        assert state.writes == []
    store = _store(shared_vespa["http_port"])
    assert store.list_configs(target) == []
    assert _registry_rows(store, {source, target}) == []


def test_concurrent_restores_keep_each_tenants_own_registry(shared_vespa):
    """Two operators restore one source's export into two registered tenants
    at once; each tenant's registry still names only its own deployment."""
    store = _store(shared_vespa["http_port"])
    source = f"source{uuid4().hex[:8]}:tenant"
    tenants = [f"import{uuid4().hex[:8]}:tenant" for _ in range(2)]
    base_schema = "document_text"
    for tenant in [source, *tenants]:
        _register_schema(store, tenant, base_schema)
    store.set_config(source, ConfigScope.AGENT, "search_agent", "settings", {"k": 5})
    payload = store.export_configs(source)

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    barrier = threading.Barrier(2)
    with _vespa_proxy(shared_vespa["base_url"], barrier=barrier) as (port, state):
        children = [
            context.Process(target=_child_import, args=(port, tenant, payload, output))
            for tenant in tenants
        ]
        try:
            with _children_start_from_this_module():
                for child in children:
                    child.start()
            results = [output.get(timeout=180) for _ in children]
            for child in children:
                child.join(timeout=10)
            assert [child.exitcode for child in children] == [0, 0]
            assert sorted(results, key=lambda row: row["tenant"]) == [
                {"tenant": tenant, "errors": [], "exceptions": []}
                for tenant in sorted(tenants)
            ]
            assert sorted(state.writes) == sorted(
                (tenant, "settings") for tenant in tenants
            )
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
    for tenant in tenants:
        assert store.get_config(
            tenant, ConfigScope.AGENT, "search_agent", "settings"
        ).config_value == {"k": 5}
    assert _registry_rows(store, {source, *tenants}) == sorted(
        (
            tenant,
            f"schema_{base_schema}",
            tenant,
            _full_schema_name(base_schema, tenant),
        )
        for tenant in [source, *tenants]
    )
