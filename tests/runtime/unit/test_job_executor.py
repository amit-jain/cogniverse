"""Coverage for the job_executor delivery-routing logic.

Drives the post-action router (``_execute_action``) against a real ASGI app
via ``httpx.ASGITransport`` and exercises the pure-helper classifiers.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

import cogniverse_runtime.job_executor as je
from cogniverse_runtime.job_executor import (
    _cosine_sim,
    _detect_deliveries,
    _execute_action,
    _is_pure_delivery,
)


@pytest.fixture(autouse=True)
def _clear_delivery_cache():
    je._delivery_embeddings.clear()
    yield
    je._delivery_embeddings.clear()


# ---- _cosine_sim ---------------------------------------------------------


def test_cosine_sim_identical_is_one():
    assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_sim_orthogonal_is_zero():
    assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_sim_zero_vector_guarded():
    assert _cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---- _is_pure_delivery ---------------------------------------------------


@pytest.mark.parametrize(
    "action,expected",
    [
        ("save to wiki", True),
        ("send to telegram", True),
        ("summarize and save to wiki", False),
        ("create a detailed report", False),
        # Documented example: "summary" is a processing intent, not pure
        # delivery — the raw result must be summarized first.
        ("send me a summary on telegram", False),
        ("post the analysis to wiki", False),
    ],
)
def test_is_pure_delivery(action, expected):
    assert _is_pure_delivery(action) is expected


# ---- _detect_deliveries (semantic match over stubbed embeddings) ---------


def _fake_embed(text: str, _url: str) -> list:
    t = text.lower()
    has_wiki = "wiki" in t
    has_tg = "telegram" in t
    if has_wiki and has_tg:
        return [0.7, 0.7]
    if has_wiki:
        return [1.0, 0.0]
    if has_tg:
        return [0.0, 1.0]
    return [0.0, 0.0]


@pytest.mark.parametrize(
    "action,expected",
    [
        ("save this to the wiki", ["wiki"]),
        ("send it on telegram", ["telegram"]),
        ("save to wiki and send on telegram", ["wiki", "telegram"]),
        ("just think about it", []),
    ],
)
def test_detect_deliveries(monkeypatch, action, expected):
    monkeypatch.setattr(je, "_embed_text", _fake_embed)
    matched = _detect_deliveries(action, "http://denseon")
    assert sorted(matched) == sorted(expected)


# ---- _execute_action against a real ASGI app -----------------------------


def _stub_app():
    app = FastAPI()
    calls = {"agent": [], "wiki": [], "telegram": []}

    @app.post("/agents/orchestrator_agent/process")
    async def _process(body: dict):
        calls["agent"].append(body)
        return {"message": f"PROCESSED:{body['query']}"}

    @app.post("/wiki/save")
    async def _wiki(body: dict):
        calls["wiki"].append(body)
        return {"slug": "slug-1"}

    @app.post("/messaging/send")
    async def _msg(body: dict):
        calls["telegram"].append(body)
        return {"status": "sent"}

    return app, calls


async def _run_action(action, query, context, deliveries, monkeypatch):
    monkeypatch.setattr(je, "_detect_deliveries", lambda *a, **kw: deliveries)
    app, calls = _stub_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await _execute_action(
            client, "http://test", "acme:acme", action, query, context, "http://denseon"
        )
    return result, calls


async def test_pure_delivery_skips_agent_and_delivers_context(monkeypatch):
    result, calls = await _run_action(
        "save to wiki", "weekly news", "PRIOR RESULT", ["wiki"], monkeypatch
    )

    assert result == "PRIOR RESULT"
    assert calls["agent"] == [], "pure delivery must not call the agent"
    assert len(calls["wiki"]) == 1
    assert calls["wiki"][0]["response"]["answer"] == "PRIOR RESULT"
    assert calls["telegram"] == []


async def test_processing_action_routes_through_agent_then_delivers(monkeypatch):
    result, calls = await _run_action(
        "summarize and send on telegram",
        "weekly news",
        "PRIOR",
        ["telegram"],
        monkeypatch,
    )

    assert len(calls["agent"]) == 1
    assert calls["agent"][0]["query"].startswith("summarize and send on telegram")
    assert "PRIOR" in calls["agent"][0]["query"]
    assert result.startswith("PROCESSED:")
    assert len(calls["telegram"]) == 1
    assert calls["telegram"][0]["message"] == result
    assert calls["wiki"] == []


async def test_agent_only_action_delivers_nowhere(monkeypatch):
    result, calls = await _run_action(
        "create a detailed report", "weekly news", "PRIOR", [], monkeypatch
    )

    assert len(calls["agent"]) == 1
    assert result.startswith("PROCESSED:")
    assert calls["wiki"] == []
    assert calls["telegram"] == []
