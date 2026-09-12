"""The local semantic-router stack runs the chart's data plane, and says so.

Two failures this pins, both of which shipped green:

* The stack's Envoy config was a hand-written twin of the chart's and had
  never picked up ``message_timeout``. Envoy's default per-message ext_proc
  deadline is 200ms, the router's classifier decides in ~280ms, so Envoy
  cancelled the routing stream and answered 504 on every chat completion.
* Readiness waited on ``/v1/models``, which Envoy answers without ever
  carrying a body to the router. It passes on exactly the stack that cannot
  serve a completion.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from tests.utils import semantic_router_stack as stack

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _rendered() -> dict[str, Any]:
    return yaml.safe_load(stack.render_envoy_config())


def _listener(document: dict[str, Any]) -> dict[str, Any]:
    return document["static_resources"]["listeners"][0]


def _connection_manager(document: dict[str, Any]) -> dict[str, Any]:
    return _listener(document)["filter_chains"][0]["filters"][0]["typed_config"]


def _chart_values() -> dict[str, Any]:
    return yaml.safe_load(stack.CHART_VALUES.read_text())["semanticRouter"]


class TestRenderedEnvoyIsTheChartDataPlane:
    def test_ext_proc_filter_is_the_charts_whole_filter(self):
        """The per-message deadline is the knob whose absence 504s the stack."""
        filters = _connection_manager(_rendered())["http_filters"]
        assert [entry["name"] for entry in filters] == [
            "envoy.filters.http.ext_proc",
            "envoy.filters.http.router",
        ]
        assert filters[0]["typed_config"] == {
            "@type": (
                "type.googleapis.com/envoy.extensions.filters.http."
                "ext_proc.v3.ExternalProcessor"
            ),
            "failure_mode_allow": False,
            "grpc_service": {
                "envoy_grpc": {"cluster_name": "semantic_router"},
                "timeout": "60s",
            },
            "message_timeout": "30s",
            "max_message_timeout": "60s",
            "processing_mode": {
                "request_header_mode": "SEND",
                "request_body_mode": "BUFFERED",
                "response_header_mode": "SEND",
                "response_body_mode": "BUFFERED",
            },
        }

    def test_every_response_names_its_reason_and_byte_counts(self):
        """A 504 that names no reason is why this defect took two runs to place."""
        logs = _connection_manager(_rendered())["access_log"]
        assert [entry["name"] for entry in logs] == ["envoy.access_loggers.stdout"]
        assert logs[0]["typed_config"]["log_format"]["json_format"] == {
            "ts": "%START_TIME%",
            "method": "%REQ(:METHOD)%",
            "path": "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%",
            "status": "%RESPONSE_CODE%",
            "reason": "%RESPONSE_CODE_DETAILS%",
            "flags": "%RESPONSE_FLAGS%",
            "bytes_received": "%BYTES_RECEIVED%",
            "bytes_sent": "%BYTES_SENT%",
            "duration_ms": "%DURATION%",
            "upstream": "%UPSTREAM_HOST%",
            "request_id": "%REQ(X-REQUEST-ID)%",
        }

    def test_listener_and_admin_come_from_the_chart_values(self):
        document = _rendered()
        values = _chart_values()
        assert _listener(document)["address"]["socket_address"] == {
            "address": "0.0.0.0",
            "port_value": values["envoy"]["service"]["port"],
        }
        assert (
            _listener(document)["per_connection_buffer_limit_bytes"]
            == values["envoy"]["maxRequestBytes"]
        )
        assert document["admin"]["address"]["socket_address"] == {
            "address": "0.0.0.0",
            "port_value": values["envoy"]["adminPort"],
        }
        assert stack.envoy_listener_port() == values["envoy"]["service"]["port"]

    def test_clusters_address_the_local_peers_over_plain_http(self):
        """The stub stands in for the served LLM, so the chart's TLS blocks render out."""
        document = _rendered()
        clusters = {
            entry["name"]: entry for entry in document["static_resources"]["clusters"]
        }
        assert sorted(clusters) == ["llm_upstream", "semantic_router"]
        endpoint = clusters["llm_upstream"]["load_assignment"]["endpoints"][0]
        assert endpoint["lb_endpoints"][0]["endpoint"]["address"]["socket_address"] == {
            "address": stack.UPSTREAM_ALIAS,
            "port_value": stack.UPSTREAM_PORT,
        }
        assert "transport_socket" not in clusters["llm_upstream"]
        router_endpoint = clusters["semantic_router"]["load_assignment"]["endpoints"][0]
        assert router_endpoint["lb_endpoints"][0]["endpoint"]["address"][
            "socket_address"
        ] == {
            "address": stack.ROUTER_ALIAS,
            "port_value": _chart_values()["router"]["grpcPort"],
        }
        route = _connection_manager(document)["route_config"]["virtual_hosts"][0][
            "routes"
        ][0]
        assert route["route"] == {"cluster": "llm_upstream", "timeout": "300s"}


class TestRenderRefusesWhatItCannotSubstitute:
    """A chart expression the stack does not know must fail loudly, not silently."""

    def _template(self, tmp_path: Path, monkeypatch, body: str) -> Path:
        path = tmp_path / "envoy.yaml"
        path.write_text(
            stack.CHART_ENVOY_TEMPLATE.read_text().replace(
                "stat_prefix: ingress_http", f"stat_prefix: {body}"
            )
        )
        monkeypatch.setattr(stack, "CHART_ENVOY_TEMPLATE", path)
        return path

    def test_unknown_helper_names_itself(self, tmp_path, monkeypatch):
        self._template(tmp_path, monkeypatch, '{{ include "cogniverse.newThing" . }}')
        with pytest.raises(stack.ChartRenderError) as raised:
            stack.render_envoy_config()
        assert 'include "cogniverse.newThing" .' in str(raised.value)

    def test_absent_chart_value_names_itself(self, tmp_path, monkeypatch):
        self._template(
            tmp_path, monkeypatch, "{{ int .Values.semanticRouter.envoy.newKnob }}"
        )
        with pytest.raises(stack.ChartRenderError) as raised:
            stack.render_envoy_config()
        assert ".Values.semanticRouter.envoy.newKnob" in str(raised.value)
        assert str(stack.CHART_VALUES) in str(raised.value)

    def test_unevaluatable_condition_names_itself(self, tmp_path, monkeypatch):
        path = tmp_path / "envoy.yaml"
        path.write_text(
            stack.CHART_ENVOY_TEMPLATE.read_text().replace(
                "static_resources:",
                '{{- if hasPrefix "x" (include "cogniverse.fullname" .) }}\n'
                "static_resources:",
            )
        )
        monkeypatch.setattr(stack, "CHART_ENVOY_TEMPLATE", path)
        with pytest.raises(stack.ChartRenderError) as raised:
            stack.render_envoy_config()
        assert 'hasPrefix "x"' in str(raised.value)


class _RoutingStub(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint whose chat behaviour each test sets."""

    chat_status = 200
    served_model = "basic-chat"
    chat_hang_s = 0.0

    def _send(self, status: int, payload: dict[str, Any] | None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        self._send(200, {"object": "list", "data": [{"id": "auto"}]})

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        if self.chat_hang_s:
            time.sleep(self.chat_hang_s)
            return
        if self.chat_status != 200:
            self._send(self.chat_status, None)
            return
        reflection = {"served_model": type(self).served_model, "echo": "probe"}
        self._send(
            200,
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(reflection),
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    def log_message(self, *args):
        return


@pytest.fixture
def routing_stub():
    """A real HTTP endpoint, so the probe exercises the transport it will meet."""

    def serve(
        *,
        chat_status: int = 200,
        served_model: str = "basic-chat",
        chat_hang_s: float = 0.0,
    ) -> str:
        handler = type(
            "_Configured",
            (_RoutingStub,),
            {
                "chat_status": chat_status,
                "served_model": served_model,
                "chat_hang_s": chat_hang_s,
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    servers: list[tuple[ThreadingHTTPServer, threading.Thread]] = []
    yield serve
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestReadinessWaitsForARoutedCompletion:
    def test_returns_the_reflection_the_router_produced(self, routing_stub):
        base_url = routing_stub()
        assert stack.wait_for_routed_chat(
            base_url,
            tenant_id="free-tenant",
            tenant_tier="free",
            expected_model="basic-chat",
            budget_s=5,
            poll_s=0.1,
        ) == {"served_model": "basic-chat", "echo": "probe"}

    def test_refuses_a_stack_whose_chat_endpoint_answers_504(self, routing_stub):
        """The state the old ``/v1/models`` probe handed to the tests as ready."""
        base_url = routing_stub(chat_status=504)
        assert (
            stack.routed_chat(base_url, tenant_id="free-tenant", tenant_tier="free")[0]
            == 504
        )
        with pytest.raises(RuntimeError) as raised:
            stack.wait_for_routed_chat(
                base_url,
                tenant_id="free-tenant",
                tenant_tier="free",
                expected_model="basic-chat",
                budget_s=0.5,
                poll_s=0.1,
            )
        message = str(raised.value)
        assert f"POST {base_url}/chat/completions" in message
        assert "'free-tenant'" in message and "'free'" in message
        assert "Last response: HTTP 504" in message

    def test_refuses_a_stack_that_never_rewrites_the_model(self, routing_stub):
        base_url = routing_stub(served_model="auto")
        with pytest.raises(RuntimeError) as raised:
            stack.wait_for_routed_chat(
                base_url,
                tenant_id="free-tenant",
                tenant_tier="free",
                expected_model="basic-chat",
                budget_s=0.5,
                poll_s=0.1,
            )
        assert "HTTP 200 but served_model='auto', expected 'basic-chat'" in str(
            raised.value
        )

    def test_refuses_a_stack_that_never_accepted_a_connection(self):
        with pytest.raises(RuntimeError) as raised:
            stack.wait_for_routed_chat(
                "http://127.0.0.1:1/v1",
                tenant_id="free-tenant",
                tenant_tier="free",
                expected_model="basic-chat",
                budget_s=0.5,
                poll_s=0.1,
            )
        assert "ConnectionError" in str(raised.value)


def _old_models_probe(base_url: str, *, budget_s: float) -> str:
    """The ``/v1/models`` readiness form: a status below 500 satisfies it.

    The loop carries no failure branch, so a caller gated on it proceeds
    whether or not it was ever satisfied.
    """
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/models", timeout=3).status_code < 500:
                return "satisfied"
        except requests.RequestException:
            pass
        time.sleep(0.1)
    return "budget exhausted, stack yielded anyway"


class TestTheModelsProbeIsSatisfiedByAStackThatCannotChat:
    """``/v1/models`` never carries a body to ext_proc, so it answers while
    every completion fails. These pin the differential the fix turns on."""

    def test_models_probe_is_satisfied_where_chat_answers_504(self, routing_stub):
        base_url = routing_stub(chat_status=504)

        assert _old_models_probe(base_url, budget_s=2) == "satisfied"

        with pytest.raises(RuntimeError) as raised:
            stack.wait_for_routed_chat(
                base_url,
                tenant_id="free-tenant",
                tenant_tier="free",
                expected_model="basic-chat",
                budget_s=0.5,
                poll_s=0.1,
            )
        message = str(raised.value)
        assert f"POST {base_url}/chat/completions" in message
        assert "Last response: HTTP 504" in message

    def test_models_probe_is_satisfied_where_chat_never_answers(self, routing_stub):
        base_url = routing_stub(chat_hang_s=5.0)

        assert _old_models_probe(base_url, budget_s=2) == "satisfied"

        with pytest.raises(RuntimeError) as raised:
            stack.wait_for_routed_chat(
                base_url,
                tenant_id="free-tenant",
                tenant_tier="free",
                expected_model="basic-chat",
                budget_s=1.0,
                poll_s=0.1,
                request_timeout_s=0.3,
            )
        message = str(raised.value)
        assert f"POST {base_url}/chat/completions" in message
        assert "ReadTimeout" in message
