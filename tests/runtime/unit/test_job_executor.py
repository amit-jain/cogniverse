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


def test_cosine_sim_length_mismatch_raises():
    """zip() used to silently truncate the longer vector, producing a
    plausible-but-wrong similarity; mismatched dims must raise instead."""
    with pytest.raises(
        ValueError,
        match=r"cosine similarity requires equal-length vectors: got 2 and 3",
    ):
        _cosine_sim([1.0, 0.0], [1.0, 0.0, 0.0])


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

    @app.post("/admin/messaging/send")
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
    # tenant_id must travel INSIDE context — the /process route reads
    # task.context["tenant_id"] and the dispatcher 400s on a missing one; a
    # top-level tenant_id is silently dropped by AgentTask.
    assert calls["agent"][0]["context"] == {"tenant_id": "acme:acme"}
    assert "tenant_id" not in {
        k for k in calls["agent"][0] if k not in ("agent_name", "query", "context")
    }
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

    @app.post("/admin/messaging/send")
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


async def test_deliver_to_telegram_accepts_enqueued_envelope(caplog):
    """The caller accepts the real /messaging/send response ({"enqueued": N})
    as success — the route and job_executor agree on the 2xx shape."""
    seen = []
    app = FastAPI()

    @app.post("/admin/messaging/send")
    async def _msg(body: dict):
        seen.append(body)
        return {"enqueued": 2}

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="cogniverse_runtime.job_executor"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await je._deliver_to_telegram(
                client, "http://test", "acme:acme", "hello world"
            )

    assert seen == [{"tenant_id": "acme:acme", "message": "hello world"}]
    assert "Delivered to Telegram" in caplog.text
    assert "Telegram delivery failed" not in caplog.text


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


async def test_call_agent_sends_tenant_id_nested_in_context():
    """_call_agent must nest tenant_id inside context, not top-level — the
    /process route reads task.context["tenant_id"] (the AgentTask model has no
    top-level tenant_id field, and the dispatcher 400s without the nested one),
    so a regression to a top-level tenant_id would silently drop it."""
    captured: dict = {}
    app = FastAPI()

    @app.post("/agents/orchestrator_agent/process")
    async def _proc(body: dict):
        captured.update(body)
        return {"message": "ok"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await je._call_agent(client, "http://test", "acme:acme", "find papers")

    assert captured["context"] == {"tenant_id": "acme:acme"}
    assert "tenant_id" not in captured  # never sent top-level


# ---- main() exit-code contract -------------------------------------------


class TestMainExitCode:
    """Argo reads the container exit code to decide retry/failure: run_job
    raising must exit 1; a clean run must exit 0 (main returns, no
    SystemExit). Argparse is driven through sys.argv exactly as the
    CronWorkflow template invokes the module."""

    _ARGV = [
        "job_executor",
        "--job-id",
        "job-42",
        "--tenant-id",
        "acme:prod",
        "--runtime-url",
        "http://runtime.svc:28000",
    ]

    def test_success_returns_without_system_exit(self, monkeypatch):
        calls = []

        async def _ok(job_id, tenant_id, runtime_url):
            calls.append((job_id, tenant_id, runtime_url))

        monkeypatch.setattr(je, "run_job", _ok)
        monkeypatch.setattr("sys.argv", list(self._ARGV))

        assert je.main() is None  # returns normally → process exit code 0
        assert calls == [("job-42", "acme:prod", "http://runtime.svc:28000")]

    def test_run_job_failure_exits_1(self, monkeypatch):
        async def _boom(job_id, tenant_id, runtime_url):
            raise RuntimeError("config store unreachable")

        monkeypatch.setattr(je, "run_job", _boom)
        monkeypatch.setattr("sys.argv", list(self._ARGV))

        with pytest.raises(SystemExit) as excinfo:
            je.main()
        assert excinfo.value.code == 1

    def test_missing_required_args_exit_2(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["job_executor"])
        with pytest.raises(SystemExit) as excinfo:
            je.main()
        assert excinfo.value.code == 2  # argparse usage error
