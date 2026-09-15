"""Read the actual gateway and profile-selection writers through dashboard tabs."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from cogniverse_agents.gateway_agent import GatewayAgent, _RoutingThresholds
from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent
from cogniverse_dashboard.tabs import profile_metrics, routing_evaluation
from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

pytestmark = pytest.mark.integration


def _emit(manager, tenants, monkeypatch, *, concurrent=False):
    captured = {}
    lock = threading.Lock()
    original = manager.span
    barrier = threading.Barrier(len(tenants)) if concurrent else None

    @contextmanager
    def observe(name, *, tenant_id, **kwargs):
        with original(name, tenant_id=tenant_id, **kwargs) as span:
            with lock:
                captured[(tenant_id, name)] = f"{span.get_span_context().span_id:016x}"
            if barrier is not None:
                barrier.wait(timeout=10)
            yield span

    monkeypatch.setattr(manager, "span", observe)

    def write(tenant):
        writer = SimpleNamespace(telemetry_manager=manager)
        GatewayAgent._emit_routing_span(
            writer,
            tenant_id=tenant,
            query=f"video for {tenant}",
            complexity="simple",
            modality="video",
            generation_type="search",
            routed_to="search_agent",
            confidence=0.9,
            reasoning="video requested",
            thresholds=_RoutingThresholds(0.4, 0.5),
        )
        asyncio.run(
            ProfileSelectionAgent._emit_profile_span(
                writer,
                query=f"video for {tenant}",
                tenant_id=tenant,
                available_profiles="video_profile",
                selected_profile="video_profile",
                intent="search",
                modality="video",
                complexity="simple",
                confidence=0.9,
            )
        )

    with ThreadPoolExecutor(max_workers=len(tenants)) as pool:
        list(pool.map(write, tenants))
    manager.force_flush(timeout_millis=10000)
    return captured


@pytest.fixture
def tab_reads(monkeypatch):
    st.cache_data.clear()
    reads = []
    original = PhoenixTraceStore.get_spans

    async def observe(self, *args, **kwargs):
        record = {"project": kwargs["project"], "ids": set()}
        reads.append(record)
        frame = await original(self, *args, **kwargs)
        if frame is not None and not frame.empty:
            record["ids"] = set(frame["context.span_id"])
        return frame

    monkeypatch.setattr(PhoenixTraceStore, "get_spans", observe)

    class WindowEnd(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=31)

    monkeypatch.setattr(routing_evaluation, "datetime", WindowEnd)
    monkeypatch.setattr(profile_metrics, "datetime", WindowEnd)
    yield reads
    st.cache_data.clear()


def _render(tab, tenant):
    app = AppTest.from_string(
        f"""
import streamlit as st
from cogniverse_dashboard.tabs.{tab} import render_{tab}_tab
st.session_state['current_tenant'] = {tenant!r}
render_{tab}_tab()
""",
        default_timeout=60,
    ).run()
    assert [e.message for e in app.exception] == []
    return app


def _wait_for_ids(manager, captured):
    async def query():
        found = set()
        for tenant, _name in captured:
            frame = await manager.get_provider(tenant_id=tenant).traces.get_spans(
                project=manager.config.get_project_name(tenant)
            )
            if frame is not None and not frame.empty:
                found.update(frame["context.span_id"])
        return found

    expected = set(captured.values())
    deadline = time.monotonic() + 30
    found = set()
    while time.monotonic() < deadline:
        found = asyncio.run(query())
        if found == expected:
            break
        time.sleep(0.2)
    assert found == expected


@pytest.mark.parametrize("concurrent", [False, True])
def test_tabs_read_only_selected_tenants_producer_spans(
    telemetry_manager_with_phoenix, monkeypatch, tab_reads, concurrent
):
    manager = telemetry_manager_with_phoenix
    tenants = [f"metrics{uuid4().hex[:8]}:tenant" for _ in range(2)]
    captured = _emit(manager, tenants, monkeypatch, concurrent=concurrent)
    _wait_for_ids(manager, captured)
    for tenant in tenants:
        for tab, span_name, metric in [
            ("routing_evaluation", "cogniverse.routing", "Total Decisions"),
            ("profile_metrics", "cogniverse.profile_selection", "VIDEO"),
        ]:
            tab_reads.clear()
            app = _render(tab, tenant)
            assert {m.label: m.value for m in app.metric}[metric] == "1"
            assert tab_reads == [
                {
                    "project": manager.config.get_project_name(tenant),
                    "ids": {captured[(tenant, span_name)]},
                }
            ]
            assert [e.value for e in app.error] == []


@pytest.mark.parametrize("tab", ["routing_evaluation", "profile_metrics"])
def test_tab_query_failure_after_project_lookup_is_not_no_traffic(
    phoenix_container, telemetry_manager_with_phoenix, monkeypatch, tab_reads, tab
):
    manager = telemetry_manager_with_phoenix
    tenant = f"metrics{uuid4().hex[:8]}:tenant"
    captured = _emit(manager, [tenant], monkeypatch)
    _wait_for_ids(manager, captured)
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            paths.append((self.command, self.path))
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.command == "POST" and self.path == "/v1/spans":
                status, content = 503, b'{"detail":"query interrupted"}'
            else:
                response = httpx.request(
                    self.command,
                    phoenix_container["http_endpoint"] + self.path,
                    content=body,
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
                status, content = response.status_code, response.content
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    get_telemetry_registry().clear_cache()
    manager.config.provider_config["http_endpoint"] = (
        f"http://127.0.0.1:{server.server_port}"
    )
    tab_reads.clear()
    try:
        app = _render(tab, tenant)
        assert tab_reads == [
            {"project": manager.config.get_project_name(tenant), "ids": set()}
        ]
        assert ("POST", "/v1/spans") in paths
        assert [m.value for m in app.metric] == []
        assert [m.value for m in app.info if m.value.startswith("No spans found")] == []
        notices = [e.value for e in app.error] + [e.value for e in app.warning]
        assert len(notices) == 1
        assert "503" in notices[0] or "unavailable" in notices[0].lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
