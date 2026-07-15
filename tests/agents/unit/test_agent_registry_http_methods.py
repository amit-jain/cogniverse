"""AgentRegistry's HTTP health/discovery methods against a REAL HTTP server.

These six methods (health checks, load balancing, agent-card discovery,
auto-registration, workflow agent selection) do real async HTTP but had zero
callers in tests — kept as the ops surface for multi-pod A2A deployments, so
their contracts are pinned here against an actual server, not mocks.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def agent_server():
    """Real HTTP server serving /health (toggleable) + an agent card."""
    state = {"healthy": True}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                code = 200 if state["healthy"] else 500
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            elif self.path == "/.well-known/agent-card.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "name": "discovered_agent",
                            "capabilities": ["video_search", "summarization"],
                            "process_endpoint": "/agents/discovered_agent/process",
                        }
                    ).encode()
                )
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture()
def registry():
    return AgentRegistry(tenant_id="test:unit", config_manager=Mock())


def _endpoint(name, url, capabilities):
    return AgentEndpoint(
        name=name, url=url, capabilities=capabilities, health_endpoint="/health"
    )


@pytest.mark.asyncio
async def test_health_check_agent_reflects_real_status(registry, agent_server):
    state, url = agent_server
    registry.register_agent(_endpoint("probe_agent", url, ["video_search"]))

    assert await registry.health_check_agent("probe_agent") is True
    assert registry.agents["probe_agent"].health_status == "healthy"

    state["healthy"] = False
    assert await registry.health_check_agent("probe_agent") is False
    assert registry.agents["probe_agent"].health_status == "unhealthy"

    assert await registry.health_check_agent("never_registered") is False


@pytest.mark.asyncio
async def test_health_check_all_covers_reachable_and_dead(registry, agent_server):
    _, url = agent_server
    registry.register_agent(_endpoint("alive_agent", url, ["video_search"]))
    registry.register_agent(
        _endpoint("dead_agent", "http://127.0.0.1:9", ["summarization"])
    )

    results = await registry.health_check_all()

    assert results["alive_agent"] is True
    assert results["dead_agent"] is False
    assert registry.agents["dead_agent"].health_status in (
        "unhealthy",
        "unreachable",
    )


@pytest.mark.asyncio
async def test_load_balanced_agent_prefers_healthy(registry, agent_server):
    _, url = agent_server
    registry.register_agent(_endpoint("healthy_one", url, ["video_search"]))
    registry.register_agent(
        _endpoint("dead_one", "http://127.0.0.1:9", ["video_search"])
    )
    await registry.health_check_all()

    chosen = registry.get_load_balanced_agent("video_search")
    assert chosen is not None
    assert chosen.name == "healthy_one"

    assert registry.get_load_balanced_agent("no_such_capability") is None


@pytest.mark.asyncio
async def test_discover_agent_by_url_parses_the_real_card(registry, agent_server):
    _, url = agent_server

    endpoint = await registry.discover_agent_by_url(url)

    assert endpoint.name == "discovered_agent"
    assert endpoint.url == url  # card has no url → falls back to the base url
    assert endpoint.capabilities == ["video_search", "summarization"]
    assert endpoint.process_endpoint == "/agents/discovered_agent/process"
    assert endpoint.health_endpoint == "/health"


@pytest.mark.asyncio
async def test_auto_register_from_urls_registers_good_and_reports_bad(
    registry, agent_server
):
    _, url = agent_server

    results = await registry.auto_register_from_urls([url, "http://127.0.0.1:9"])

    assert results[url] is True
    assert results["http://127.0.0.1:9"] is False
    assert "discovered_agent" in registry.agents
    assert "video_search" in registry.capabilities
    assert "discovered_agent" in registry.capabilities["video_search"]


def test_get_agents_for_workflow_selects_by_type(registry):
    registry.register_agent(_endpoint("searcher", "http://a:1", ["video_search"]))
    registry.register_agent(_endpoint("summarizer", "http://b:2", ["summarization"]))
    registry.register_agent(_endpoint("reporter", "http://c:3", ["detailed_analysis"]))

    raw = [a.name for a in registry.get_agents_for_workflow("raw_results")]
    assert raw == ["searcher"]

    summary = [a.name for a in registry.get_agents_for_workflow("summary")]
    assert summary == ["searcher", "summarizer"]

    report = [a.name for a in registry.get_agents_for_workflow("detailed_report")]
    assert report == ["searcher", "reporter"]
