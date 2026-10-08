"""Rating search results from the web client's agent workspace, against
search spans in real Phoenix.

The runtime serves the AG-UI and agents routers with the dispatcher's own
``SearchAgent``; only its query encoder and search backend are replaced, so
the search span, the results it records and the span id a run hands the
browser are the production ones. Ratings reach Phoenix through a forwarding
proxy, so a test can hold or fail the writes. Every rating is read back from
Phoenix, and the triplet miner reads them as its training signal.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.search_agent import SearchInput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_foundation.telemetry.span_contract import RESULT_RELEVANCE
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.routers import ag_ui, agents, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.stub_search import (
    REPO_ROOT,
    answer_with,
    build_stub_search_agent,
    memory_config_manager,
    stub_encoder_factory,
)
from tests.utils.web_client import (
    build_web_client,
    install_web_client,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

KEY = "web-relevance-harness-key"
OTHER_KEY = "web-relevance-other-harness-key"
QUERY = "cat playing fetch in a park"
POS_CONTENT = "a tabby cat chasing a red ball across the grass"
NEG_CONTENT = "a dog sleeping on a leather couch"
POS_ID = "id:video:video::vid_pos"
NEG_ID = "id:video:video::vid_neg"
HITS = [
    (
        "vid_pos",
        0.91,
        {
            "documentid": POS_ID,
            "video_title": "Fetch in the park",
            "text_content": POS_CONTENT,
        },
    ),
    (
        "vid_neg",
        0.82,
        {
            "documentid": NEG_ID,
            "video_title": "Nap time",
            "text_content": NEG_CONTENT,
        },
    ),
]
SPAN_NAME = "SearchAgent.process"


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    """The global telemetry manager: spans export to Phoenix, reads and
    annotation writes go through ``phoenix_proxy``."""
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
    """The /ag-ui and /agents routers on a real socket, dispatching to the
    dispatcher's own SearchAgent over ``HITS``."""
    config_manager = memory_config_manager()
    registry = AgentRegistry(tenant_id="acme:web", config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://localhost:8000",
            capabilities=["search"],
        )
    )
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
        return answer_with(agent, HITS)

    dispatcher._get_search_agent = search_agent

    # No conversation memory is configured: each run's turn takes its place in
    # the ledger and is not stored.
    dispatcher._conversation_store_factory = lambda tenant_id: None

    @asynccontextmanager
    async def lifespan(_app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        prefix = f"test:relevance:{uuid.uuid4().hex}"
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
    openai_compat.set_key_resolver(None)
    with serve_app(app) as url:
        yield url
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})


@pytest.fixture()
def tenants(runtime_url, phoenix_proxy):
    """Two fresh tenants: ``KEY`` is the first's, ``OTHER_KEY`` the second's."""
    tenant = canonical_tenant_id(f"rate{uuid.uuid4().hex[:8]}")
    other = canonical_tenant_id(f"rateother{uuid.uuid4().hex[:8]}")
    openai_compat.set_api_keys({KEY: tenant, OTHER_KEY: other})
    yield tenant, other
    phoenix_proxy.intercept = None


@pytest.fixture(scope="module")
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("relevance")))


@pytest.fixture()
def web_url(built_client, runtime_url, tenants):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
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
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


def _search(telemetry, tenant_id, hits=HITS):
    """Run a search for ``tenant_id`` through a SearchAgent; returns the id of
    the span it recorded."""
    agent = build_stub_search_agent(tenant_id, hits)
    agent.set_telemetry_manager(telemetry)
    output = run_in_own_loop(
        agent.process(
            SearchInput(
                query=QUERY, tenant_id=tenant_id, enhanced_query=QUERY, top_k=10
            )
        )
    )
    telemetry.force_flush(timeout_millis=10000)
    return output.span_id


def _provider(telemetry, tenant_id):
    project = telemetry.config.get_project_name(tenant_id)
    return telemetry.get_provider(tenant_id=tenant_id, project_name=project), project


def _search_spans(telemetry, tenant_id, count, timeout=90.0):
    """The tenant's search spans once ``count`` of them are readable."""
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


def _ratings(telemetry, tenant_id, span_id, count, timeout=90.0):
    """{result id: (label, score)} of the span's relevance annotations once
    ``count`` are readable."""
    provider, project = _provider(telemetry, tenant_id)
    spans = _search_spans(telemetry, tenant_id, 1)
    spans = spans[spans["context.span_id"] == span_id]
    deadline = time.monotonic() + timeout
    while True:
        annotations = run_in_own_loop(
            provider.annotations.get_annotations(
                spans_df=spans, project=project, annotation_names=[RESULT_RELEVANCE]
            )
        )
        ratings = {
            TripletExtractor._annotation_result_id(row): (
                row["result.label"],
                row["result.score"],
            )
            for _, row in annotations.iterrows()
        }
        if len(ratings) >= count or time.monotonic() > deadline:
            return ratings
        time.sleep(2)


def _rate(runtime_url, key, span_id, result_id, relevance):
    return httpx.post(
        f"{runtime_url}/ag-ui/results/relevance",
        headers={"Authorization": f"Bearer {key}"},
        json={"span_id": span_id, "result_id": result_id, "relevance": relevance},
        timeout=120,
    )


def _annotation_writes(proxy, since):
    """The annotation writes the proxy forwarded after its ``since``-th request."""
    return [
        path
        for method, path, _ in proxy.requests[since:]
        if method == "POST" and "annotations" in path
    ]


def _run_search_in_browser(page: Page, web_url: str):
    page.goto(f"{web_url}/#/agents/search_agent")
    chat = page.get_by_placeholder("Ask Search…")
    chat.fill(QUERY)
    chat.press("Enter")
    results = page.get_by_role("complementary", name="Results")
    expect(results.get_by_role("group", name=f"Relevance of {POS_ID}")).to_be_visible(
        timeout=120_000
    )
    return results


def test_ratings_from_the_workspace_are_stored_and_mined_as_a_triplet(
    page, web_url, tenants, telemetry
):
    tenant, _ = tenants
    results = _run_search_in_browser(page, web_url)
    expect(results.locator(".result-title")).to_have_text(
        ["Fetch in the park", "Nap time"]
    )

    positive = results.get_by_role("group", name=f"Relevance of {POS_ID}")
    negative = results.get_by_role("group", name=f"Relevance of {NEG_ID}")
    positive.get_by_role("button", name="Highly Relevant").click()
    expect(positive.get_by_role("button", name="Highly Relevant")).to_have_attribute(
        "aria-pressed", "true"
    )
    negative.get_by_role("button", name="Not Relevant").click()
    expect(negative.get_by_role("button", name="Not Relevant")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(positive.get_by_role("button", name="Not Relevant")).to_have_attribute(
        "aria-pressed", "false"
    )

    # The run's span is the one search span of the tenant.
    spans = _search_spans(telemetry, tenant, 1)
    assert list(spans["name"]) == [SPAN_NAME]
    (span_id,) = spans["context.span_id"]
    assert _ratings(telemetry, tenant, span_id, 2) == {
        POS_ID: ("Highly Relevant", 1.0),
        NEG_ID: ("Not Relevant", 0.0),
    }

    provider, project = _provider(telemetry, tenant)
    triplets = run_in_own_loop(
        TripletExtractor(provider=provider).extract(
            project=project, modality="video", strategy="top_k", min_triplets=1
        )
    )
    assert [
        (t.anchor, t.positive, t.negative, t.metadata["span_id"]) for t in triplets
    ] == [(QUERY, POS_CONTENT, NEG_CONTENT, span_id)]


def test_a_rating_that_was_not_stored_shows_on_its_card(
    page, web_url, tenants, telemetry, phoenix_proxy
):
    tenant, _ = tenants
    results = _run_search_in_browser(page, web_url)
    phoenix_proxy.intercept = lambda method, path, body: (
        (503, {"detail": "unavailable"})
        if method == "POST" and "annotations" in path
        else None
    )
    positive = results.get_by_role("group", name=f"Relevance of {POS_ID}")
    positive.get_by_role("button", name="Highly Relevant").click()
    expect(positive.get_by_role("alert")).to_have_text(
        "The relevance of result id:video:video::vid_pos was not stored "
        "(HTTPStatusError). See server logs for detail."
    )
    expect(positive.get_by_role("button", name="Highly Relevant")).to_have_attribute(
        "aria-pressed", "false"
    )
    phoenix_proxy.intercept = None
    (span_id,) = _search_spans(telemetry, tenant, 1)["context.span_id"]
    assert _ratings(telemetry, tenant, span_id, 1, timeout=6) == {}


def test_a_span_of_another_tenant_is_not_rated(
    runtime_url, tenants, telemetry, phoenix_proxy
):
    tenant, other = tenants
    span_id = _search(telemetry, other)
    _search_spans(telemetry, other, 1)
    requests_before = len(phoenix_proxy.requests)

    response = _rate(runtime_url, KEY, span_id, POS_ID, "Highly Relevant")

    assert (response.status_code, response.json()) == (
        404,
        {
            "error": {
                "message": f"Search span {span_id} is not a span of this tenant.",
                "type": "invalid_request_error",
                "code": "span_not_found",
            }
        },
    )
    assert _annotation_writes(phoenix_proxy, requests_before) == []
    assert _ratings(telemetry, other, span_id, 1, timeout=6) == {}
    # The same rating with the span's own tenant's key is stored.
    stored = _rate(runtime_url, OTHER_KEY, span_id, POS_ID, "Highly Relevant")
    assert (stored.status_code, stored.json()) == (
        200,
        {
            "span_id": span_id,
            "result_id": POS_ID,
            "relevance": "Highly Relevant",
            "score": 1.0,
        },
    )
    assert _ratings(telemetry, other, span_id, 1) == {POS_ID: ("Highly Relevant", 1.0)}


def test_requests_without_a_key_or_with_a_bad_rating_are_refused(
    runtime_url, tenants, phoenix_proxy
):
    requests_before = len(phoenix_proxy.requests)
    unauthenticated = httpx.post(
        f"{runtime_url}/ag-ui/results/relevance",
        json={"span_id": "0" * 16, "result_id": POS_ID, "relevance": "Not Relevant"},
    )
    bad_label = _rate(runtime_url, KEY, "0" * 16, POS_ID, "Meh")
    bad_span = _rate(runtime_url, KEY, "not-a-span", POS_ID, "Not Relevant")

    assert (unauthenticated.status_code, unauthenticated.json()["error"]["code"]) == (
        401,
        openai_compat.UNAUTHORIZED["code"],
    )
    assert (bad_label.status_code, bad_label.json()["error"]["message"]) == (
        400,
        "Invalid relevance rating: relevance: Input should be 'Highly Relevant', "
        "'Somewhat Relevant' or 'Not Relevant'",
    )
    assert (bad_span.status_code, bad_span.json()["error"]["message"]) == (
        400,
        "Invalid relevance rating: span_id: String should match pattern "
        "'^[0-9a-f]{16}$'",
    )
    assert phoenix_proxy.requests[requests_before:] == []


def test_ratings_stored_at_once_each_keep_their_own_result(
    runtime_url, tenants, telemetry, phoenix_proxy
):
    tenant, _ = tenants
    hits = [(f"vid_{i}", 0.9 - i / 10, {"text_content": f"clip {i}"}) for i in range(6)]
    labels = ["Highly Relevant", "Somewhat Relevant", "Not Relevant"] * 2
    span_id = _search(telemetry, tenant, hits)
    _search_spans(telemetry, tenant, 1)
    barrier = threading.Barrier(len(hits), timeout=60)
    held = []

    def hold_annotation_writes(method, path, body):
        # Every rating's write reaches Phoenix together.
        if method == "POST" and "annotations" in path:
            held.append(json.loads(body))
            barrier.wait()
        return None

    phoenix_proxy.intercept = hold_annotation_writes

    async def rate_all():
        async with httpx.AsyncClient(timeout=120) as client:
            return await asyncio.gather(
                *(
                    client.post(
                        f"{runtime_url}/ag-ui/results/relevance",
                        headers={"Authorization": f"Bearer {KEY}"},
                        json={
                            "span_id": span_id,
                            "result_id": f"vid_{i}",
                            "relevance": labels[i],
                        },
                    )
                    for i in range(len(hits))
                )
            )

    responses = run_in_own_loop(rate_all())
    phoenix_proxy.intercept = None

    assert [response.status_code for response in responses] == [200] * len(hits)
    assert len(held) == len(hits)
    scores = {"Highly Relevant": 1.0, "Somewhat Relevant": 0.5, "Not Relevant": 0.0}
    assert _ratings(telemetry, tenant, span_id, len(hits)) == {
        f"vid_{i}": (labels[i], scores[labels[i]]) for i in range(len(hits))
    }


def test_an_unreadable_telemetry_backend_stores_nothing(
    runtime_url, tenants, telemetry, phoenix_proxy
):
    tenant, _ = tenants
    span_id = _search(telemetry, tenant)
    _search_spans(telemetry, tenant, 1)
    requests_before = len(phoenix_proxy.requests)
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})

    response = _rate(runtime_url, KEY, span_id, POS_ID, "Highly Relevant")
    phoenix_proxy.intercept = None

    assert (response.status_code, response.json()) == (
        502,
        {
            "error": {
                "message": f"The relevance of result {POS_ID} was not stored "
                "(HTTPStatusError). See server logs for detail.",
                "type": "server_error",
                "code": "annotation_not_stored",
                "error_type": "HTTPStatusError",
            }
        },
    )
    assert _annotation_writes(phoenix_proxy, requests_before) == []
    assert _ratings(telemetry, tenant, span_id, 1, timeout=6) == {}
