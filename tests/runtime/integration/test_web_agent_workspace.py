"""The web client's agent workspace, driven in Chromium against the runtime's
AG-UI surface.

The built client's Node server runs agents through CopilotKit against the
``/ag-ui`` and ``/agents`` routers on a real uvicorn socket. The real
dispatcher runs deterministic agents whose final payloads have the shapes the
runtime's own agents produce; every run's turn is saved through the real
conversation ledger on Redis into Mem0 on real Vespa (behind a fault proxy),
and a conversation the page restores is read back from there.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.conversation import ConversationStore
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import ag_ui, agents, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.memory_store import InMemoryConfigStore
from tests.utils.web_client import (
    browse_as,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)
from tests.utils.web_ops import harness_key_admin, memory_on_vespa

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "webchat:alpha"
KEY = "web-workspace-harness-key"
SPAN_ID = "00000000000000ab"
VIDEO_HIT = {
    "id": "v1_seg_3",
    "document_id": "id:video:video::v1_seg_3",
    "score": 5.29,
    "rrf_score": 0.0333,
    "metadata": {
        "video_id": "v1",
        "video_title": "match.mp4",
        "audio_transcript": "- Yeah.",
        "segment_description": "A man throws a ball on a grassy field.",
    },
    "temporal_info": {"start_time": 65.0, "end_time": 71.5},
}
DOCUMENT_HIT = {
    "document_id": "doc_7",
    "document_url": "s3://docs/report.pdf",
    "title": "Quarterly report",
    "page_number": 3,
    "document_type": "pdf",
    "content_preview": "Revenue grew 12% on the back of video search.",
    "relevance_score": 0.82,
    "strategy_used": "text",
    "metadata": {},
}
CODE = "def reverse(s):\n    return s[::-1]\n\nprint(reverse('hello'))"
CODE_RESULT = {
    "plan": "Write reverse().",
    "code_changes": [
        {
            "file_path": "/workspace/solution.py",
            "content": CODE,
            "change_type": "create",
        }
    ],
    "execution_results": [
        {
            "command": "python /workspace/solution.py",
            "exit_code": 0,
            "stdout": "olleh\n",
            "stderr": "",
            "success": True,
        }
    ],
    "summary": "Completed coding task in 1 iteration(s).",
    "iterations_used": 1,
    "files_modified": ["/workspace/solution.py"],
}
slow_started = threading.Event()
key_points_release = threading.Event()
EXECUTION_SUMMARY = "Ran search_agent and document_agent in parallel."
ENSEMBLE_PROFILES = [
    "video_colpali_smol500_mv_frame",
    "video_videoprism_base_mv_chunk_30s",
]
THEMES = ["field sports", "ball games", "spectators", "grass"]
DRAFT_SUMMARY = "A man throws a ball on a field."
KEY_POINTS = ["One clip shows a throw", "Spectators watch from the side"]


class WorkspaceDeps(AgentDeps):
    pass


class WorkspaceInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class SearchOutput(AgentOutput):
    summary: str = ""
    span_id: str = ""
    results: list = []


class SearchAgent(AgentBase[WorkspaceInput, SearchOutput, WorkspaceDeps]):
    """Streams its summary, then hits of the search agent's and the document
    agent's public shapes."""

    async def _process_impl(self, input: WorkspaceInput) -> SearchOutput:
        summary = (
            f"Found 2 results for {input.query} "
            f"after {len(input.conversation_history)} earlier messages."
        )
        accumulated = ""
        for start in range(0, len(summary), 9):
            chunk = summary[start : start + 9]
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": accumulated, "output_field": "summary"},
            )
            await asyncio.sleep(0)
        return SearchOutput(
            summary=summary, span_id=SPAN_ID, results=[VIDEO_HIT, DOCUMENT_HIT]
        )


class AnswerOutput(AgentOutput):
    answer: str = ""
    result: dict = {}
    orchestration_result: dict = {}
    key_points: list = []


class ChatAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    """Answers with the query and how much history the run sent it."""

    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        return AnswerOutput(
            answer=f"Noted {input.query} after "
            f"{len(input.conversation_history)} earlier messages."
        )


class CodingAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    """Answers with the coding agent's envelope: its output under ``result``."""

    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        return AnswerOutput(answer=CODE_RESULT["summary"], result=CODE_RESULT)


class OrchestratorAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    """Answers with an orchestration whose steps found hits of two shapes."""

    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        return AnswerOutput(
            answer="Searched videos and documents.",
            orchestration_result={
                "agent_results": {
                    "query_enhancement_agent": {"enhanced_query": input.query},
                    "search_agent": {"span_id": SPAN_ID, "results": [VIDEO_HIT]},
                    "document_agent": {"results": [DOCUMENT_HIT]},
                },
                "execution_summary": EXECUTION_SUMMARY,
            },
        )


class EnsembleOutput(AgentOutput):
    answer: str = ""
    results: list = []
    profile: str | None = None
    profiles: list = []
    search_mode: str = ""
    degraded_profiles: list = []


class EnsembleAgent(AgentBase[WorkspaceInput, EnsembleOutput, WorkspaceDeps]):
    """Answers with the search agent's envelope for an ensemble one of whose
    profiles did not run; a query starting "nothing" finds no hits."""

    async def _process_impl(self, input: WorkspaceInput) -> EnsembleOutput:
        found = [] if input.query.startswith("nothing") else [VIDEO_HIT]
        return EnsembleOutput(
            answer=f"Found {len(found)} results",
            results=found,
            profiles=ENSEMBLE_PROFILES,
            search_mode="ensemble",
            degraded_profiles=[
                {"profile": ENSEMBLE_PROFILES[1], "reason": "encoder unavailable"}
            ],
        )


class KeyPointsAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    """Reports the themes it found and a draft, holds until released, then
    answers with key points."""

    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        self.emit_progress(
            "thinking", "Content analysis complete", data={"themes": THEMES}
        )
        self.emit_progress(
            "summarization", "Summary generated", data={"summary": DRAFT_SUMMARY}
        )
        await asyncio.to_thread(key_points_release.wait, 60)
        return AnswerOutput(answer=DRAFT_SUMMARY, key_points=KEY_POINTS)


class FailingAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        raise RuntimeError("secret-backend-detail")


class SlowAgent(AgentBase[WorkspaceInput, AnswerOutput, WorkspaceDeps]):
    """Reports a phase, then holds the run open."""

    async def _process_impl(self, input: WorkspaceInput) -> AnswerOutput:
        self.emit_progress("thinking", "Thinking it over")
        slow_started.set()
        await asyncio.sleep(60)
        return AnswerOutput(answer="done thinking")


_AGENT_CLASSES = {
    "ensemble_agent": f"{__name__}:EnsembleAgent",
    "key_points_agent": f"{__name__}:KeyPointsAgent",
    "search_agent": f"{__name__}:SearchAgent",
    "chat_agent": f"{__name__}:ChatAgent",
    "coding_agent": f"{__name__}:CodingAgent",
    "orchestrator_agent": f"{__name__}:OrchestratorAgent",
    "failing_agent": f"{__name__}:FailingAgent",
    "slow_agent": f"{__name__}:SlowAgent",
    # Registered last, so the default is not merely the first agent listed.
    "gateway_agent": f"{__name__}:ChatAgent",
}
_TOKEN_STREAMING = {"search_agent": True, "slow_agent": True}


@pytest.fixture(scope="module")
def memory(vespa_instance, config_manager):
    with memory_on_vespa(vespa_instance, config_manager, (TENANT,)) as (
        managers,
        proxy,
    ):
        yield managers, proxy


@pytest.fixture(scope="module")
def runtime_url(memory, workflow_state_redis_url):
    managers, _ = memory
    store = InMemoryConfigStore()
    store.initialize()
    registry_config = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT, config_manager=registry_config)
    for name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://localhost:8000",
                capabilities=["web"],
                streams_answer_tokens=_TOKEN_STREAMING.get(name, False),
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=registry_config, schema_loader=None
    )
    dispatcher._conversation_store_factory = lambda tenant_id: ConversationStore(
        managers[tenant_id], tenant_id
    )

    @asynccontextmanager
    async def lifespan(_app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        prefix = f"test:workspace:{uuid.uuid4().hex}"
        openai_compat.set_continuation_store(
            ContinuationStore(redis, key_prefix=prefix)
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                redis,
                save_lease_s=CONVERSATION_SAVE_LEASE_S,
                failure_capacity=CONVERSATION_PERSIST_FAILURE_CAPACITY,
                key_prefix=f"{prefix}:conversation",
            )
        )
        try:
            yield
        finally:
            await dispatcher.drain_conversation_saves()
            dispatcher.set_conversation_ledger(None)
            openai_compat.set_continuation_store(None)
            await redis.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(ag_ui.router, prefix="/ag-ui")
    app.include_router(agents.router, prefix="/agents")
    agents.set_agent_registry(registry)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY: TENANT})
    with harness_key_admin(app, registry_config), serve_app(app) as url:
        yield url
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    for name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(name, None)


@pytest.fixture()
def web_url(built_client, runtime_url, memory):
    _, proxy = memory
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []
    proxy.intercept = None


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context()
    browse_as(context, TENANT)
    page = context.new_page()
    yield page
    context.close()


def _thread_of(page: Page, agent: str) -> str:
    prefix = f"#/agents/{agent}/"
    hash_ = page.evaluate("window.location.hash")
    assert hash_.startswith(prefix), hash_
    return hash_.removeprefix(prefix)


def _open(page: Page, web_url: str, agent: str, label: str) -> str:
    page.goto(f"{web_url}/#/agents/{agent}")
    expect(page.get_by_role("heading", name=label, level=1)).to_be_visible()
    expect(page.get_by_placeholder(f"Ask {label}…")).to_be_visible()
    page.wait_for_function(
        f"window.location.hash.startsWith('#/agents/{agent}/')", timeout=10_000
    )
    return _thread_of(page, agent)


def _send(page: Page, label: str, text: str) -> None:
    box = page.get_by_placeholder(f"Ask {label}…")
    box.fill(text)
    box.press("Enter")


def _user_messages(page: Page):
    return page.get_by_test_id("copilot-user-message")


def _assistant_messages(page: Page):
    return page.get_by_test_id("copilot-assistant-message")


def _saved_turns(runtime_url: str, thread: str):
    response = httpx.get(
        f"{runtime_url}/ag-ui/threads/{thread}",
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestRunNotices:
    def test_a_failed_run_shows_its_error_in_the_conversation(self, page, web_url):
        _open(page, web_url, "failing_agent", "Failing")
        _send(page, "Failing", "break it")
        notice = page.locator(".run-notice")
        expect(notice).to_have_text(
            "The run failed: failing_agent failed with RuntimeError. See server "
            "logs for detail.",
            timeout=60_000,
        )
        expect(notice).to_have_attribute("role", "alert")
        expect(_user_messages(page)).to_have_count(1)
        expect(_user_messages(page)).to_contain_text("break it")
        expect(_assistant_messages(page)).to_have_count(0)

    def test_a_cancelled_run_says_so(self, page, web_url):
        slow_started.clear()
        _open(page, web_url, "slow_agent", "Slow")
        _send(page, "Slow", "think hard")
        expect(page.get_by_role("status").first).to_have_text(
            "Thinking it over", timeout=60_000
        )
        assert slow_started.wait(30)
        page.get_by_test_id("copilot-send-button").click()
        notice = page.locator(".run-notice")
        expect(notice).to_have_text("You cancelled this run.", timeout=30_000)
        expect(notice).to_have_attribute("role", "status")
        expect(page.locator(".workspace-header .status")).to_have_count(0)


class TestConversations:
    def test_a_conversation_survives_a_reload_and_switching_agents(
        self, page, web_url, runtime_url
    ):
        thread = _open(page, web_url, "chat_agent", "Chat")
        _send(page, "Chat", "cats")
        first = "Noted cats after 0 earlier messages."
        expect(_assistant_messages(page)).to_have_text([first], timeout=60_000)
        _send(page, "Chat", "dogs")
        second = "Noted dogs after 2 earlier messages."
        expect(_assistant_messages(page)).to_have_text([first, second], timeout=60_000)
        assert _saved_turns(runtime_url, thread) == {
            "thread_id": thread,
            "state": "loaded",
            "reason": None,
            "turns": [
                {"role": "user", "content": "cats"},
                {"role": "assistant", "content": first},
                {"role": "user", "content": "dogs"},
                {"role": "assistant", "content": second},
            ],
        }

        page.reload()
        expect(_user_messages(page)).to_have_text(["cats", "dogs"], timeout=30_000)
        expect(_assistant_messages(page)).to_have_text([first, second])
        assert _thread_of(page, "chat_agent") == thread

        page.get_by_role("navigation").get_by_role("link", name="Coding").click()
        expect(page.get_by_role("heading", name="Coding", level=1)).to_be_visible()
        coding_thread = _thread_of(page, "coding_agent")
        assert coding_thread != thread
        _send(page, "Coding", "reverse a string")
        expect(_assistant_messages(page)).to_have_text(
            [CODE_RESULT["summary"]], timeout=60_000
        )

        page.get_by_role("navigation").get_by_role("link", name="Chat").click()
        expect(_user_messages(page)).to_have_text(["cats", "dogs"], timeout=30_000)
        expect(_assistant_messages(page)).to_have_text([first, second])
        assert _thread_of(page, "chat_agent") == thread

        # The next message continues the restored conversation.
        _send(page, "Chat", "birds")
        third = "Noted birds after 4 earlier messages."
        expect(_assistant_messages(page)).to_have_text(
            [first, second, third], timeout=60_000
        )

        page.get_by_role("button", name="New conversation").click()
        expect(_user_messages(page)).to_have_count(0)
        new_thread = _thread_of(page, "chat_agent")
        assert new_thread != thread
        page.go_back()
        expect(_user_messages(page)).to_have_text(
            ["cats", "dogs", "birds"], timeout=30_000
        )

    def test_conversations_held_at_once_each_restore_their_own_turns(
        self, browser, web_url
    ):
        contexts = [browser.new_context() for _ in range(3)]
        for context in contexts:
            browse_as(context, TENANT)
        pages = [context.new_page() for context in contexts]
        try:
            threads = [_open(page, web_url, "search_agent", "Search") for page in pages]
            assert len(set(threads)) == 3
            for index, page in enumerate(pages):
                _send(page, "Search", f"topic {index}")
            for index, page in enumerate(pages):
                expect(_assistant_messages(page)).to_have_text(
                    [f"Found 2 results for topic {index} after 0 earlier messages."],
                    timeout=60_000,
                )
            for page in pages:
                page.reload()
            for index, page in enumerate(pages):
                expect(_user_messages(page)).to_have_text(
                    [f"topic {index}"], timeout=30_000
                )
        finally:
            for context in contexts:
                context.close()

    def test_a_store_outage_on_restore_is_an_error_not_an_empty_conversation(
        self, page, web_url, memory
    ):
        _, proxy = memory
        thread = _open(page, web_url, "search_agent", "Search")
        _send(page, "Search", "kept")
        expect(_assistant_messages(page)).to_have_count(1, timeout=60_000)
        proxy.intercept = lambda method, path, body: (503, {"error": "down"})
        page.reload()
        notice = page.locator(".run-notice")
        expect(notice).to_have_text(
            "The run failed: This conversation could not be restored: The "
            "conversation store is unavailable (HTTPError). See server logs for "
            "detail.",
            timeout=30_000,
        )
        expect(notice).to_have_attribute("role", "alert")
        assert _thread_of(page, "search_agent") == thread
        proxy.intercept = None


class TestResults:
    def test_search_hits_show_their_titles_snippets_and_scores(self, page, web_url):
        _open(page, web_url, "search_agent", "Search")
        _send(page, "Search", "fields")
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-title")).to_have_text(
            ["match.mp4", "Quarterly report"], timeout=60_000
        )
        expect(results.locator(".result-snippet")).to_have_text(
            [
                "A man throws a ball on a grassy field.",
                "Revenue grew 12% on the back of video search.",
            ]
        )
        expect(results.locator(".result-score")).to_have_text(["0.033", "0.820"])
        expect(results.locator(".result-time")).to_have_text([" 1:05–1:11"])
        expect(results.get_by_role("group")).to_have_count(2)
        expect(results.get_by_role("group", name="Relevance of doc_7")).to_be_visible()

    def test_the_coding_agent_shows_its_code_and_run_output(self, page, web_url):
        _open(page, web_url, "coding_agent", "Coding")
        _send(page, "Coding", "reverse a string")
        code = page.get_by_role("region", name="Code from Coding")
        expect(code.locator(".code-file figcaption")).to_have_text(
            ["/workspace/solution.py (create)"], timeout=60_000
        )
        expect(code.locator(".code-file code")).to_have_text([CODE])
        expect(code.locator(".code-run figcaption")).to_have_text(
            ["python /workspace/solution.py — exit code 0"]
        )
        expect(code.get_by_label("Output")).to_have_text("olleh")

    def test_an_orchestration_shows_each_agents_hits(self, page, web_url):
        _open(page, web_url, "orchestrator_agent", "Orchestrator")
        _send(page, "Orchestrator", "videos and documents")
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-group")).to_have_text(
            ["Search", "Document"], timeout=60_000
        )
        expect(results.locator(".result-title")).to_have_text(
            ["match.mp4", "Quarterly report"]
        )
        # Only the search recorded a span, so only its hit can be rated.
        expect(results.get_by_role("group")).to_have_count(1)
        expect(
            results.get_by_role("group", name="Relevance of id:video:video::v1_seg_3")
        ).to_be_visible()
        expect(
            results.get_by_role("region", name="Orchestration summary").locator("p")
        ).to_have_text(EXECUTION_SUMMARY)

    def test_a_partial_ensemble_and_an_empty_search_say_so(self, page, web_url):
        _open(page, web_url, "ensemble_agent", "Ensemble")
        _send(page, "Ensemble", "fields at dusk")
        search = page.get_by_role("complementary", name="Results").get_by_role(
            "region", name="Search by Ensemble"
        )
        expect(search.locator(".result-found")).to_have_text(
            "Found 1 result for 'fields at dusk'.", timeout=60_000
        )
        expect(search.locator(".result-metrics dt")).to_have_text(
            ["Results", "Latency", "Profile", "Search mode"]
        )
        expect(search.locator(".result-metrics dd").nth(2)).to_have_text(
            ", ".join(ENSEMBLE_PROFILES)
        )
        expect(search.locator(".result-metrics dd").nth(3)).to_have_text("ensemble")
        expect(search.get_by_role("alert")).to_have_text(
            f"Partial results: {ENSEMBLE_PROFILES[1]} did not run "
            "(encoder unavailable)."
        )

        _send(page, "Ensemble", "nothing at all")
        expect(search.locator(".result-found")).to_have_text(
            "No results for 'nothing at all'.", timeout=60_000
        )
        expect(search.locator(".result-card")).to_have_count(0)


class TestProgress:
    def test_a_dispatched_run_shows_its_status_themes_and_draft_then_key_points(
        self, page, web_url
    ):
        key_points_release.clear()
        _open(page, web_url, "key_points_agent", "Key points")
        _send(page, "Key points", "summarize the match")

        header = page.locator(".workspace-header")
        expect(header.get_by_role("status")).to_have_text(
            "Summary generated", timeout=60_000
        )
        progress = page.locator(".progress-detail")
        expect(progress.locator(".status")).to_have_text(
            "Themes: field sports, ball games, spectators"
        )
        expect(progress.locator(".draft-summary")).to_have_text(DRAFT_SUMMARY)
        key_points_release.set()

        points = page.get_by_role("complementary", name="Results").get_by_role(
            "region", name="Key points"
        )
        expect(points.get_by_role("listitem")).to_have_text(KEY_POINTS, timeout=60_000)
        expect(header.get_by_role("status")).to_have_count(0)
        expect(page.locator(".progress-detail")).to_have_count(0)


class TestNavigation:
    def test_the_gateway_opens_by_default(self, page, web_url):
        page.goto(f"{web_url}/#/")
        expect(page.get_by_role("heading", name="Gateway", level=1)).to_be_visible()
        page.wait_for_function(
            "window.location.hash.startsWith('#/agents/gateway_agent/')",
            timeout=10_000,
        )

    def test_an_agent_the_runtime_does_not_serve_is_named(self, page, web_url):
        page.goto(f"{web_url}/#/agents/retired_agent")
        expect(page.get_by_role("heading", name="Gateway", level=1)).to_be_visible()
        expect(page.locator(".main > .alert")).to_have_text(
            "Agent 'retired_agent' is not registered with the runtime, so this is "
            "gateway_agent instead."
        )
        page.get_by_role("navigation").get_by_role("link", name="Chat").click()
        expect(page.get_by_role("heading", name="Chat", level=1)).to_be_visible()
        expect(page.locator(".main > .alert")).to_have_count(0)

    def test_an_empty_message_cannot_be_sent(self, page, web_url):
        _open(page, web_url, "chat_agent", "Chat")
        send = page.get_by_test_id("copilot-send-button")
        expect(send).to_be_disabled()
        box = page.get_by_placeholder("Ask Chat…")
        box.fill("   ")
        expect(send).to_be_disabled()
        box.press("Enter")
        page.wait_for_timeout(1000)
        expect(_user_messages(page)).to_have_count(0)
        box.fill("hello")
        expect(send).to_be_enabled()

    def test_the_shell_loads_and_navigates_without_console_errors(self, page, web_url):
        # This runtime's tenant registry answers 503 by design (see
        # ``harness_key_admin``), which the browser logs for each tenant check.
        probe = f"/ui-api/runtime/admin/tenants/{quote(TENANT, safe='')}"
        errors = []
        page.on(
            "console",
            lambda message: (
                errors.append(message.text)
                if message.type == "error"
                and not (
                    message.text.startswith("Failed to load resource")
                    and message.location["url"].endswith(probe)
                )
                else None
            ),
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{web_url}/#/")
        expect(page.get_by_placeholder("Ask Gateway…")).to_be_visible()
        nav = page.get_by_role("navigation")
        nav.get_by_role("link", name="Chat", exact=True).click()
        expect(page.get_by_placeholder("Ask Chat…")).to_be_visible()
        first = page.url
        page.get_by_role("button", name="New conversation").click()
        expect(page).not_to_have_url(first)
        expect(page.get_by_placeholder("Ask Chat…")).to_be_visible()
        nav.get_by_role("link", name="Search", exact=True).click()
        expect(page.get_by_placeholder("Ask Search…")).to_be_visible()
        assert errors == []
