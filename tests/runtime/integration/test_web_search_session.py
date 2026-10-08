"""A search conversation in the web client's agent workspace, against the
dispatcher's own ``SearchAgent`` and real Phoenix.

The runtime serves the AG-UI and agents routers with the production search
path; only the query encoder and the search backend are replaced, and the
backend answers the first ``top_k`` of a fixed hit list as Vespa answers a
query's ``hits``. Search spans, their relevance ratings and a conversation's
evaluation are written to and read back from Phoenix, writes going through a
forwarding proxy so a test can hold or fail them.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from fastapi import FastAPI
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.search_agent import SearchAgent, SearchInput
from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_foundation.telemetry.span_contract import SESSION_EVALUATION
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import ag_ui, agents, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.stub_search import (
    REPO_ROOT,
    StubBackend,
    StubEncoder,
    build_stub_search_agent,
    memory_config_manager,
    stub_encoder_factory,
)
from tests.utils.web_client import (
    browse_as,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)
from tests.utils.web_ops import harness_key_admin

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

KEY = "web-session-harness-key"
OTHER_KEY = "web-session-other-harness-key"
QUERY = "tower lit at night"
HITS = [
    (
        "tower_night",
        0.91,
        {
            "documentid": "id:video:video::tower_night",
            "video_id": "tower_night",
            "video_title": "Tower at night",
            "text_content": "The tower lit against a dark sky.",
        },
    ),
    (
        "tower_dusk",
        0.62,
        {
            "documentid": "id:video:video::tower_dusk",
            "video_id": "tower_dusk",
            "video_title": "Tower at dusk",
            "text_content": "The tower as the lights come on.",
        },
    ),
    (
        "tower_day",
        0.21,
        {
            "documentid": "id:video:video::tower_day",
            "video_id": "tower_day",
            "video_title": "Tower by day",
            "text_content": "The tower under a clear noon sky.",
        },
    ),
]
SPAN_NAME = "SearchAgent.process"
THEMES = ["night skyline", "tower lights", "dusk"]
# The top_k each search asked the backend for, in order.
backend_top_ks: List[int] = []


class TopKBackend(StubBackend):
    """Answers the first ``top_k`` hits of a query, as Vespa does."""

    def search(self, query_dict):
        backend_top_ks.append(query_dict["top_k"])
        return super().search(query_dict)[: query_dict["top_k"]]


class SummaryDeps(AgentDeps):
    pass


class SummaryInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    search_results: list = []


class SummaryOutput(AgentOutput):
    summary: str = ""
    key_points: list = []


class GroundedSummarizer(AgentBase[SummaryInput, SummaryOutput, SummaryDeps]):
    """Summarizes the hits its run was grounded in, by title."""

    async def _process_impl(self, input: SummaryInput) -> SummaryOutput:
        self.emit_progress(
            "thinking", "Content analysis complete", data={"themes": THEMES}
        )
        titles = [hit["metadata"]["video_title"] for hit in input.search_results]
        return SummaryOutput(
            summary=f"{len(titles)} clips: {'; '.join(titles)}.",
            key_points=[f"{title} matches" for title in titles],
        )


def _search_agent(agent: SearchAgent) -> SearchAgent:
    agent.query_encoder = StubEncoder()
    backend = TopKBackend(HITS)
    agent._get_backend = lambda: backend
    agent.is_memory_enabled = lambda: False
    return agent


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_proxy.url,
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(scope="module")
def runtime_url(telemetry, workflow_state_redis_url):
    config_manager = memory_config_manager()
    registry = AgentRegistry(tenant_id="acme:web", config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent", url="http://localhost:8000", capabilities=["search"]
        )
    )
    registry.register_agent(
        AgentEndpoint(
            name="summarizer_agent",
            url="http://localhost:8000",
            capabilities=["web_summary"],
        )
    )
    ConfigLoader.AGENT_CLASSES["summarizer_agent"] = f"{__name__}:GroundedSummarizer"
    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(
            base_path=REPO_ROOT / "configs" / "schemas"
        ),
    )
    build_search_agent = dispatcher._get_search_agent

    def search_agent(profile, tenant_id):
        with stub_encoder_factory():
            agent = build_search_agent(profile, tenant_id)
        return _search_agent(agent)

    dispatcher._get_search_agent = search_agent
    dispatcher._conversation_store_factory = lambda tenant_id: None

    @asynccontextmanager
    async def lifespan(_app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        prefix = f"test:session:{uuid.uuid4().hex}"
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
            dispatcher.set_conversation_ledger(None)
            openai_compat.set_continuation_store(None)
            await redis.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(ag_ui.router, prefix="/ag-ui")
    app.include_router(agents.router, prefix="/agents")
    agents.set_agent_registry(registry)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    with harness_key_admin(app, config_manager), serve_app(app) as url:
        yield url
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    ConfigLoader.AGENT_CLASSES.pop("summarizer_agent", None)


@pytest.fixture()
def tenants(runtime_url, phoenix_proxy):
    """Two fresh tenants: ``KEY`` is the first's, ``OTHER_KEY`` the second's."""
    tenant = canonical_tenant_id(f"session{uuid.uuid4().hex[:8]}")
    other = canonical_tenant_id(f"sessionother{uuid.uuid4().hex[:8]}")
    openai_compat.set_api_keys({KEY: tenant, OTHER_KEY: other})
    backend_top_ks.clear()
    yield tenant, other
    phoenix_proxy.intercept = None


@pytest.fixture()
def web_url(built_client, runtime_url, tenants):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def page(browser, tenants):
    context = browser.new_context(accept_downloads=True)
    browse_as(context, tenants[0])
    page = context.new_page()
    yield page
    context.close()


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _run_events(runtime_url: str, key: str, agent: str, **forwarded: Any) -> list:
    response = httpx.post(
        f"{runtime_url}/ag-ui/{agent}",
        headers=_auth(key),
        json={
            "threadId": f"thread-{uuid.uuid4().hex}",
            "runId": "run-1",
            "state": {},
            "messages": [{"id": "u1", "role": "user", "content": QUERY}],
            "tools": [],
            "context": [],
            "forwardedProps": forwarded,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _snapshot(events: list) -> dict:
    (snapshot,) = [event for event in events if event["type"] == "STATE_SNAPSHOT"]
    return snapshot["snapshot"]["result"]


def _search(telemetry, tenant_id) -> str:
    """Run a search for ``tenant_id``; the id of the span it recorded."""
    agent = build_stub_search_agent(tenant_id, HITS)
    agent.set_telemetry_manager(telemetry)
    output = run_in_own_loop(
        agent.process(
            SearchInput(query=QUERY, tenant_id=tenant_id, enhanced_query=QUERY, top_k=3)
        )
    )
    telemetry.force_flush(timeout_millis=10000)
    return output.span_id


def _provider(telemetry, tenant_id):
    project = telemetry.config.get_project_name(tenant_id)
    return telemetry.get_provider(tenant_id=tenant_id, project_name=project), project


def _search_spans(telemetry, tenant_id, count, timeout=90.0):
    provider, project = _provider(telemetry, tenant_id)
    deadline = time.monotonic() + timeout
    while True:
        end = datetime.now(timezone.utc)
        spans = run_in_own_loop(
            provider.traces.get_spans(
                project=project,
                start_time=end - timedelta(hours=1),
                end_time=end,
                filters={"name": SPAN_NAME},
            )
        )
        if len(spans) >= count or time.monotonic() > deadline:
            return spans
        time.sleep(2)


def _evaluations(telemetry, tenant_id, span_ids, expected, timeout=90.0):
    """{span id: (label, score, session id)} of the spans' session
    evaluations once they equal ``expected``, else at the timeout."""
    provider, project = _provider(telemetry, tenant_id)
    spans = _search_spans(telemetry, tenant_id, len(span_ids))
    spans = spans[spans["context.span_id"].isin(span_ids)]
    deadline = time.monotonic() + timeout
    while True:
        annotations = run_in_own_loop(
            provider.annotations.get_annotations(
                spans_df=spans, project=project, annotation_names=[SESSION_EVALUATION]
            )
        )
        found = {
            span_id: (
                row["result.label"],
                row["result.score"],
                row["metadata"]["session_id"],
            )
            for span_id, row in annotations.iterrows()
        }
        if found == expected or time.monotonic() > deadline:
            return found
        time.sleep(2)


def _evaluate(runtime_url, key, thread_id, body):
    return httpx.post(
        f"{runtime_url}/ag-ui/threads/{thread_id}/evaluation",
        headers=_auth(key),
        json=body,
        timeout=120,
    )


def _annotation_writes(proxy, since):
    return [
        path
        for method, path, _ in proxy.requests[since:]
        if method == "POST" and "annotations" in path
    ]


class TestRunParameters:
    def test_a_runs_top_k_is_how_many_hits_the_search_asks_for(
        self, runtime_url, tenants
    ):
        limited = _snapshot(
            _run_events(runtime_url, KEY, "search_agent", cogniverse={"top_k": 2})
        )
        unlimited = _snapshot(_run_events(runtime_url, KEY, "search_agent"))

        assert [hit["id"] for hit in limited["results"]] == [
            "tower_night",
            "tower_dusk",
        ]
        assert [hit["id"] for hit in unlimited["results"]] == [
            "tower_night",
            "tower_dusk",
            "tower_day",
        ]
        assert backend_top_ks == [2, 10]

    def test_an_answer_agent_is_grounded_in_the_hits_the_run_sends(
        self, runtime_url, tenants
    ):
        hits = _snapshot(
            _run_events(runtime_url, KEY, "search_agent", cogniverse={"top_k": 2})
        )["results"]

        events = _run_events(
            runtime_url, KEY, "summarizer_agent", cogniverse={"search_results": hits}
        )

        assert _snapshot(events)["key_points"] == [
            "Tower at night matches",
            "Tower at dusk matches",
        ]
        assert [event["value"] for event in events if event["type"] == "CUSTOM"][1] == {
            "phase": "thinking",
            "message": "Content analysis complete",
            "themes": THEMES,
        }


class TestSessionEvaluation:
    def test_a_verdict_is_stored_on_each_search_and_replaced_when_given_again(
        self, runtime_url, tenants, telemetry
    ):
        tenant, _ = tenants
        spans = [_search(telemetry, tenant), _search(telemetry, tenant)]
        _search_spans(telemetry, tenant, 2)
        thread = f"thread-{uuid.uuid4().hex}"

        first = _evaluate(
            runtime_url,
            KEY,
            thread,
            {"outcome": "partial", "score": 0.4, "span_ids": spans},
        )
        assert (first.status_code, first.json()) == (
            200,
            {
                "thread_id": thread,
                "outcome": "partial",
                "score": 0.4,
                "span_ids": sorted(spans),
            },
        )
        expected = {span: ("partial", 0.4, thread) for span in spans}
        assert _evaluations(telemetry, tenant, spans, expected) == expected

        again = _evaluate(
            runtime_url,
            KEY,
            thread,
            {"outcome": "success", "score": 0.9, "span_ids": spans},
        )
        assert again.status_code == 200, again.text
        replaced = {span: ("success", 0.9, thread) for span in spans}
        assert _evaluations(telemetry, tenant, spans, replaced) == replaced

    def test_a_span_of_another_tenant_is_refused_and_nothing_is_written(
        self, runtime_url, tenants, telemetry, phoenix_proxy
    ):
        tenant, other = tenants
        own = _search(telemetry, tenant)
        foreign = _search(telemetry, other)
        _search_spans(telemetry, tenant, 1)
        _search_spans(telemetry, other, 1)
        before = len(phoenix_proxy.requests)

        response = _evaluate(
            runtime_url,
            KEY,
            "thread-x",
            {"outcome": "failure", "score": 0.0, "span_ids": [own, foreign]},
        )

        project = telemetry.config.get_project_name(tenant)
        assert (response.status_code, response.json()) == (
            404,
            {
                "error": {
                    "message": "Conversation thread-x names search spans this "
                    f"tenant does not have: spans {foreign} are not in project "
                    f"{project}.",
                    "type": "invalid_request_error",
                    "code": "span_not_found",
                }
            },
        )
        assert _annotation_writes(phoenix_proxy, before) == []
        assert _evaluations(telemetry, tenant, [own], {}, timeout=6) == {}

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (
                {"outcome": "great", "score": 0.5, "span_ids": ["0" * 16]},
                "Invalid session evaluation: outcome: Input should be 'success', "
                "'partial' or 'failure'",
            ),
            (
                {"outcome": "success", "score": 1.5, "span_ids": ["0" * 16]},
                "Invalid session evaluation: score: Input should be less than or "
                "equal to 1",
            ),
            (
                {"outcome": "success", "score": 0.5, "span_ids": []},
                "Invalid session evaluation: span_ids: List should have at least 1 "
                "item after validation, not 0",
            ),
            (
                {"outcome": "success", "score": 0.5, "span_ids": ["not-a-span"]},
                "Invalid session evaluation: span_ids ['not-a-span'] are not span ids",
            ),
        ],
    )
    def test_a_malformed_verdict_is_refused(
        self, runtime_url, tenants, phoenix_proxy, body, message
    ):
        before = len(phoenix_proxy.requests)

        response = _evaluate(runtime_url, KEY, "thread-x", body)

        assert (response.status_code, response.json()) == (
            400,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )
        assert phoenix_proxy.requests[before:] == []

    def test_a_request_without_a_key_is_refused(self, runtime_url, tenants):
        response = httpx.post(
            f"{runtime_url}/ag-ui/threads/thread-x/evaluation",
            json={"outcome": "success", "score": 0.5, "span_ids": ["0" * 16]},
        )

        assert (response.status_code, response.json()["error"]["code"]) == (
            401,
            openai_compat.UNAUTHORIZED["code"],
        )

    def test_a_failed_write_is_reported_not_stored(
        self, runtime_url, tenants, telemetry, phoenix_proxy
    ):
        tenant, _ = tenants
        span = _search(telemetry, tenant)
        _search_spans(telemetry, tenant, 1)
        phoenix_proxy.intercept = lambda method, path, body: (
            (503, {"detail": "unavailable"})
            if method == "POST" and "annotations" in path
            else None
        )

        response = _evaluate(
            runtime_url,
            KEY,
            "thread-y",
            {"outcome": "success", "score": 0.7, "span_ids": [span]},
        )
        phoenix_proxy.intercept = None

        assert (response.status_code, response.json()) == (
            502,
            {
                "error": {
                    "message": "The evaluation of conversation thread-y was not "
                    "stored (HTTPStatusError). See server logs for detail.",
                    "type": "server_error",
                    "code": "annotation_not_stored",
                    "error_type": "HTTPStatusError",
                }
            },
        )
        assert _evaluations(telemetry, tenant, [span], {}, timeout=6) == {}

    def test_conversations_evaluated_at_once_each_keep_their_own_verdict(
        self, runtime_url, tenants, telemetry, phoenix_proxy
    ):
        tenant, _ = tenants
        count = 4
        spans = [_search(telemetry, tenant) for _ in range(count)]
        _search_spans(telemetry, tenant, count)
        outcomes = ["success", "partial", "failure", "success"]
        barrier = threading.Barrier(count, timeout=60)

        def hold_annotation_writes(method, path, body):
            # Every conversation's write reaches Phoenix together.
            if method == "POST" and "annotations" in path:
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_annotation_writes

        async def evaluate_all():
            async with httpx.AsyncClient(timeout=120) as client:
                return await asyncio.gather(
                    *(
                        client.post(
                            f"{runtime_url}/ag-ui/threads/thread-{index}/evaluation",
                            headers=_auth(KEY),
                            json={
                                "outcome": outcomes[index],
                                "score": index / 10,
                                "span_ids": [spans[index]],
                            },
                        )
                        for index in range(count)
                    )
                )

        responses = run_in_own_loop(evaluate_all())
        phoenix_proxy.intercept = None

        assert [response.status_code for response in responses] == [200] * count
        expected = {
            spans[index]: (outcomes[index], index / 10, f"thread-{index}")
            for index in range(count)
        }
        assert _evaluations(telemetry, tenant, spans, expected) == expected


def _workspace(page: Page, web_url: str) -> str:
    page.goto(f"{web_url}/#/agents/search_agent")
    expect(page.get_by_placeholder("Ask Search…")).to_be_visible()
    page.wait_for_function(
        "window.location.hash.startsWith('#/agents/search_agent/')", timeout=10_000
    )
    return page.evaluate("window.location.hash").removeprefix("#/agents/search_agent/")


def _ask(page: Page, text: str) -> None:
    box = page.get_by_placeholder("Ask Search…")
    box.fill(text)
    box.press("Enter")


class TestSearchWorkspace:
    def test_a_search_conversation_from_settings_to_evaluation(
        self, page, web_url, tenants, telemetry, runtime_url
    ):
        tenant, _ = tenants
        thread = _workspace(page, web_url)
        session = page.get_by_role("group", name="Session")
        expect(session).to_contain_text(f"Session {thread[:8]}0 turns")
        session.get_by_label("Results per search").fill("2")
        session.get_by_label("Minimum score").fill("0.7")

        _ask(page, QUERY)

        results = page.get_by_role("complementary", name="Results")
        search = results.get_by_role("region", name="Search by Search")
        expect(search.locator(".result-found")).to_have_text(
            f"Found 2 results for '{QUERY}'."
        )
        payload = _snapshot(
            _run_events(runtime_url, KEY, "search_agent", cogniverse={"top_k": 2})
        )
        metrics = search.locator(".result-metrics > div")
        expect(metrics.locator("dt")).to_have_text(
            ["Results", "Latency", "Profile", "Search mode"]
        )
        expect(metrics.nth(0).locator("dd")).to_have_text("2")
        expect(metrics.nth(1).locator("dd")).to_have_text(re.compile(r"^\d+ ms$"))
        expect(metrics.nth(2).locator("dd")).to_have_text(payload["profile"])
        expect(metrics.nth(3).locator("dd")).to_have_text(payload["search_mode"])
        expect(search.locator(".result-title")).to_have_text(["Tower at night"])
        expect(
            search.get_by_text("Showing 1 of 2; the rest score below 0.7.")
        ).to_be_visible()
        expect(search.locator(".result-id").nth(1)).to_have_text(
            "Video tower_night · Document id:video:video::tower_night"
        )
        expect(session).to_contain_text(f"Session {thread[:8]}1 turn")
        assert backend_top_ks == [2, 2]

        rating = search.get_by_role(
            "group", name="Relevance of id:video:video::tower_night"
        )
        rating.get_by_role("button", name="Highly Relevant").click()
        annotations = results.get_by_role("region", name="Annotations")
        expect(annotations).to_contain_text("1 rating saved in this conversation.")
        with page.expect_download() as download:
            annotations.get_by_role("button", name="Export annotations").click()
        exported = json.loads(Path(download.value.path()).read_text())
        (span_id,) = [entry["span_id"] for entry in exported["annotations"]]
        assert download.value.suggested_filename.startswith(
            f"search_annotations_{thread[:8]}_"
        )
        assert {
            "search_session": {
                key: value
                for key, value in exported["search_session"].items()
                if key != "exported_at"
            },
            "annotations": [
                {key: value for key, value in entry.items() if key != "rated_at"}
                for entry in exported["annotations"]
            ],
        } == {
            "search_session": {
                "tenant_id": tenant,
                "thread_id": thread,
                "agent": "search_agent",
                "query": QUERY,
                "profile": payload["profile"],
            },
            "annotations": [
                {
                    "query": QUERY,
                    "span_id": span_id,
                    "result_id": "id:video:video::tower_night",
                    "relevance": "Highly Relevant",
                    "score": 1.0,
                }
            ],
        }

        evaluation = results.get_by_role("form", name="Evaluate this conversation")
        save = evaluation.get_by_role("button", name="Save evaluation")
        expect(save).to_be_disabled()
        evaluation.get_by_label("Outcome").select_option("success")
        evaluation.get_by_label("Quality").fill("0.8")
        save.click()
        expect(evaluation.get_by_role("status")).to_have_text(
            "Saved: Success (0.8) on 1 search."
        )
        expected = {span_id: ("success", 0.8, thread)}
        assert _evaluations(telemetry, tenant, [span_id], expected) == expected

        history = results.get_by_role("region", name="History")
        history.get_by_role("button", name="History (1 run)").click()
        expect(history.get_by_role("listitem")).to_have_text(
            [re.compile(rf"^{re.escape(QUERY)} — 2 results at \d\d:\d\d:\d\d$")]
        )

        page.get_by_role("button", name="New conversation").click()
        expect(session).to_contain_text("0 turns")
        new_thread = page.evaluate("window.location.hash").removeprefix(
            "#/agents/search_agent/"
        )
        assert new_thread != thread
        expect(session).to_contain_text(f"Session {new_thread[:8]}")
        expect(results.get_by_role("region", name="Annotations")).to_have_count(0)
        expect(results.get_by_role("region", name="History")).to_have_count(0)

    def test_the_results_on_screen_are_summarized_with_their_key_points(
        self, page, web_url, tenants
    ):
        _workspace(page, web_url)
        page.get_by_role("group", name="Session").get_by_label(
            "Results per search"
        ).fill("2")
        _ask(page, QUERY)
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-title")).to_have_text(
            ["Tower at night", "Tower at dusk"]
        )

        summary = results.get_by_role("region", name="Summary")
        summary.get_by_role("button", name="Summarize results with Summarizer").click()

        expect(summary.locator(".summary-text")).to_have_text(
            "2 clips: Tower at night; Tower at dusk."
        )
        expect(
            summary.get_by_role("list", name="Summary key points").get_by_role(
                "listitem"
            )
        ).to_have_text(["Tower at night matches", "Tower at dusk matches"])

    def test_a_summary_that_fails_says_why(self, page, web_url, tenants):
        _workspace(page, web_url)
        _ask(page, QUERY)
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-title")).to_have_count(3)
        page.route(
            "**/ui-api/runtime/ag-ui/summarizer_agent",
            lambda route: route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps(
                    {
                        "error": {
                            "message": "agent registry unavailable",
                            "type": "server_error",
                            "code": "service_unavailable",
                        }
                    }
                ),
            ),
        )
        summary = results.get_by_role("region", name="Summary")
        summary.get_by_role("button", name="Summarize results with Summarizer").click()

        expect(summary.get_by_role("alert")).to_have_text(
            "The summary failed: agent registry unavailable"
        )
