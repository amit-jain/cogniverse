"""Agent availability probes shown in the dashboard sidebar."""

from __future__ import annotations

import httpx
import pytest

from cogniverse_dashboard.agent_status import probe_agents

RUNTIME = "http://runtime:8000"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _runtime_up(agent_responses: dict[str, httpx.Response]):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        name = request.url.path.rsplit("/", 1)[-1]
        return agent_responses.get(name, httpx.Response(404, json={"detail": "no"}))

    return handler, seen


class TestProbePath:
    def test_probes_the_agent_route_the_runtime_actually_serves(self) -> None:
        """The runtime router serves GET /agents/{agent_name}; probing a
        /health suffix it never declared is a guaranteed 404."""
        handler, seen = _runtime_up({"search_agent": httpx.Response(200, json={})})

        with _client(handler) as client:
            probe_agents(RUNTIME, ["search_agent"], client=client)

        assert seen == ["/health", "/agents/search_agent"]

    def test_probed_path_is_a_declared_runtime_route(self) -> None:
        from cogniverse_runtime.routers.agents import router

        paths = {route.path for route in router.routes}

        assert "/{agent_name}" in paths
        assert "/{agent_name}/health" not in paths


class TestStatusReflectsProbe:
    def test_registered_agent_is_online(self) -> None:
        handler, _ = _runtime_up({"search_agent": httpx.Response(200, json={})})

        with _client(handler) as client:
            results = probe_agents(RUNTIME, ["search_agent"], client=client)

        assert results == {
            "Search Agent": {
                "status": "online",
                "url": f"{RUNTIME}/agents/search_agent",
            }
        }

    def test_unregistered_agent_is_offline_not_online(self) -> None:
        """A 404 from the registry means the agent is not served. Reporting it
        online is the defect: the sidebar claimed every agent was up."""
        handler, _ = _runtime_up({})

        with _client(handler) as client:
            results = probe_agents(RUNTIME, ["ghost_agent"], client=client)

        assert results["Ghost Agent"]["status"] == "offline"
        assert results["Ghost Agent"]["message"] == "Not registered (HTTP 404)"

    @pytest.mark.parametrize("code", [400, 401, 500, 503])
    def test_any_non_200_is_offline(self, code: int) -> None:
        handler, _ = _runtime_up({"search_agent": httpx.Response(code, json={})})

        with _client(handler) as client:
            results = probe_agents(RUNTIME, ["search_agent"], client=client)

        assert results["Search Agent"]["status"] == "offline"
        assert results["Search Agent"]["message"] == f"HTTP {code}"

    def test_transport_failure_is_offline(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            raise httpx.ConnectError("refused", request=request)

        with _client(handler) as client:
            results = probe_agents(RUNTIME, ["search_agent"], client=client)

        assert results["Search Agent"]["status"] == "offline"
        assert results["Search Agent"]["message"] == "Connection failed"

    def test_mixed_registry_reports_each_agent_separately(self) -> None:
        handler, _ = _runtime_up({"search_agent": httpx.Response(200, json={})})

        with _client(handler) as client:
            results = probe_agents(
                RUNTIME, ["search_agent", "ghost_agent"], client=client
            )

        assert {name: r["status"] for name, r in results.items()} == {
            "Search Agent": "online",
            "Ghost Agent": "offline",
        }


class TestRuntimeUnreachable:
    def test_runtime_non_200_reports_error_and_probes_no_agents(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(503, json={})

        with _client(handler) as client:
            results = probe_agents(RUNTIME, ["search_agent"], client=client)

        assert results == {"error": "Runtime not reachable (HTTP 503)"}
        assert seen == ["/health"]
