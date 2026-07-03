"""Coverage for the job_executor delivery-routing logic.

Drives the post-action router (``_execute_action``) against a real ASGI app
via ``httpx.ASGITransport`` and exercises the pure-helper classifiers.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI, HTTPException

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


def _fake_embed(text: str, _url: str, *, is_query: bool = False) -> list:
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


# ---- delivery error ladder against a real ASGI app -----------------------


async def _deliver_telegram_against(status_code, seen):
    app = FastAPI()

    @app.post("/messaging/send")
    async def _msg(body: dict):
        seen.append(body)
        if status_code != 200:
            raise HTTPException(status_code=status_code)
        return {"status": "sent"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await je._deliver_to_telegram(client, "http://test", "acme:acme", "hello world")


async def test_deliver_to_telegram_success_posts_payload(caplog):
    seen = []
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        await _deliver_telegram_against(200, seen)

    assert seen == [{"tenant_id": "acme:acme", "message": "hello world"}]
    assert "Delivered to Telegram" in caplog.text


async def test_deliver_to_telegram_404_is_skipped_not_raised(caplog):
    seen = []
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        # Must return cleanly — a missing messaging endpoint is a skip,
        # not a failure.
        await _deliver_telegram_against(404, seen)

    assert seen == [{"tenant_id": "acme:acme", "message": "hello world"}]
    assert "Messaging endpoint not available" in caplog.text
    assert "Telegram delivery failed" not in caplog.text


async def test_deliver_to_telegram_server_error_is_swallowed(caplog):
    seen = []
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        # A 500 is swallowed (logged, not raised) so one failed delivery
        # never aborts the surrounding job.
        await _deliver_telegram_against(500, seen)

    assert seen == [{"tenant_id": "acme:acme", "message": "hello world"}]
    assert "Telegram delivery failed" in caplog.text
    assert "Messaging endpoint not available" not in caplog.text


async def test_deliver_to_wiki_success_posts_answer_payload(caplog):
    app = FastAPI()
    seen = []

    @app.post("/wiki/save")
    async def _wiki(body: dict):
        seen.append(body)
        return {"slug": "weekly-1"}

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await je._deliver_to_wiki(
                client, "http://test", "acme:acme", "weekly news", "FINAL RESULT"
            )

    assert len(seen) == 1
    assert seen[0]["query"] == "weekly news"
    assert seen[0]["response"]["answer"] == "FINAL RESULT"
    assert seen[0]["tenant_id"] == "acme:acme"
    assert "Delivered to wiki: slug=weekly-1" in caplog.text


async def test_deliver_to_wiki_server_error_is_swallowed(caplog):
    app = FastAPI()

    @app.post("/wiki/save")
    async def _wiki(body: dict):
        raise HTTPException(status_code=500)

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Swallowed — no raise.
            await je._deliver_to_wiki(
                client, "http://test", "acme:acme", "weekly news", "FINAL RESULT"
            )

    assert "Wiki delivery failed" in caplog.text


async def _call_agent_against(response_body: dict) -> str:
    app = FastAPI()

    @app.post("/agents/orchestrator_agent/process")
    async def _proc(body: dict):
        return response_body

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await je._call_agent(client, "http://test", "acme:acme", "find papers")


async def test_call_agent_returns_message_field():
    out = await _call_agent_against({"message": "Found 5 papers", "status": "success"})
    assert out == "Found 5 papers"


async def test_call_agent_returns_string_result():
    out = await _call_agent_against({"result": "Found 5 papers"})
    assert out == "Found 5 papers"


async def test_call_agent_unwraps_dict_result_response():
    # Some agents wrap the text one level deep; delivery must never
    # persist str(dict) into the wiki.
    out = await _call_agent_against(
        {"result": {"response": "Found 5 ColPali papers on video retrieval"}}
    )
    assert out == "Found 5 ColPali papers on video retrieval"


async def test_call_agent_unwraps_dict_result_answer():
    out = await _call_agent_against({"result": {"answer": "The answer text"}})
    assert out == "The answer text"
