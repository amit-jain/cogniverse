"""The optimization report route forwards the report agent's events as SSE."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.registries.agent_registry import AgentRegistryUnavailableError
from cogniverse_runtime.routers import agents, optimization_report
from cogniverse_runtime.routers.optimization_report import REPORT_AGENT, REPORT_QUERY

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Dispatcher:
    """The dispatcher surface the route calls, with scripted events."""

    def __init__(self, events, *, registered=True, fail_after=None, registry=None):
        self.events = events
        self.registered = registered
        self.fail_after = fail_after
        self.registry_error = registry
        self.calls = []

    async def refresh_agent_registry(self):
        if self.registry_error is not None:
            raise self.registry_error

    def is_registered(self, name):
        return self.registered and name == REPORT_AGENT

    async def dispatch_stream(self, agent_name, query, context):
        self.calls.append((agent_name, query, context["tenant_id"]))
        for event in self.events:
            yield event
            await asyncio.sleep(0)
        if self.fail_after is not None:
            raise self.fail_after


def _post(monkeypatch, dispatcher, tenant="acme:prod"):
    monkeypatch.setattr(agents, "_dispatcher", dispatcher)
    app = FastAPI()
    app.include_router(optimization_report.router, prefix="/admin/tenant")

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            return await client.post(f"/admin/tenant/{tenant}/optimize/report")

    return asyncio.run(call())


def _events(response) -> list[dict]:
    return [
        json.loads(block.removeprefix("data: "))
        for block in response.text.split("\n\n")
        if block
    ]


def test_each_agent_event_is_one_data_frame(monkeypatch):
    events = [
        {"type": "status", "phase": "search", "message": "Searching"},
        {"type": "partial", "phase": "token", "data": {"accumulated": "All"}},
        {"type": "final", "data": {"executive_summary": "All good."}},
    ]
    dispatcher = _Dispatcher(events)
    response = _post(monkeypatch, dispatcher, tenant="acme")
    assert (response.status_code, response.headers["content-type"]) == (
        200,
        "text/event-stream; charset=utf-8",
    )
    assert _events(response) == events
    assert dispatcher.calls == [(REPORT_AGENT, REPORT_QUERY, "acme:acme")]


def test_an_agent_failure_mid_stream_ends_on_an_error_event(monkeypatch):
    status = {"type": "status", "phase": "search", "message": "Searching"}
    response = _post(
        monkeypatch,
        _Dispatcher([status], fail_after=ConnectionError("vespa at 10.0.0.1 down")),
    )
    assert response.status_code == 200
    assert _events(response) == [
        status,
        {
            "type": "error",
            "message": "The report agent failed (ConnectionError) before finishing "
            "the report of tenant acme:prod.",
            "error_type": "ConnectionError",
        },
    ]


def test_an_unregistered_report_agent_is_refused(monkeypatch):
    dispatcher = _Dispatcher([], registered=False)
    response = _post(monkeypatch, dispatcher)
    assert (response.status_code, response.json()) == (
        404,
        {
            "detail": f"Agent '{REPORT_AGENT}' is not registered, so no report can "
            "be generated."
        },
    )
    assert dispatcher.calls == []


def test_an_unreadable_registry_is_an_outage_not_a_missing_agent(monkeypatch):
    response = _post(
        monkeypatch,
        _Dispatcher([], registry=AgentRegistryUnavailableError("redis down")),
    )
    assert (response.status_code, response.json()["detail"]) == (
        503,
        {
            "error": "agent_registry_unavailable",
            "message": "The agent registry could not be read; retry.",
            "failure": "AgentRegistryUnavailableError",
            "tenant_id": "acme:prod",
        },
    )
