"""The deployed web client, driven in Chromium against the e2e cluster.

The web client (``clients/web``) is the Cogniverse UI: its Node server runs
in the cluster behind the ``web`` Service and every page action reaches the
live runtime through it. Each test reads its outcome back from the runtime,
Phoenix or the config store rather than from the page alone, and owns the
state it asserts on: tenants are minted per test or per class, and the web
tenant (the one the deployment's harness key maps to) is seeded idempotently
with the tracked sample video.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone

import httpx
import pytest
from playwright.sync_api import expect

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_PROFILE_SELECTION,
    SPAN_NAME_ROUTING,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.span_contract import RESULT_RELEVANCE
from cogniverse_runtime.routers import tenant as tenant_router
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.batch_optimization import argo_phases_between
from tests.e2e.cluster import RUNTIME, TENANT_DEPLOY_TIMEOUT_S
from tests.e2e.conftest import (
    GATEWAY_VIDEO_QUERIES,
    SAMPLE_VIDEO_CONTENT_ID,
    SAMPLE_VIDEO_PATH,
    WEB,
    _search_sample_content,
)
from tests.e2e.sample_corpus import (
    _expected_sample_documents_fed,
    _sample_video_media_type,
)
from tests.e2e.tenants import register_tenant_and_wait, unique_id
from tests.e2e.test_api_e2e import PROFILE, _deploy_profile_for_tenant
from tests.e2e.web_client import (
    OPS_VIEWS,
    RUN_TIMEOUT_MS,
    VIEW_TIMEOUT_MS,
    WEB_TENANT,
    agent_label,
    assistant_messages,
    choose_tenant,
    ensure_web_tenant_corpus,
    facts,
    open_agent,
    open_view,
    registered_agents,
    result_cards,
    run_agent_and_capture_state,
    saved_turns,
    send,
    sse_events,
    thread_of,
    user_messages,
    web_harness_key,
)

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

# The Ingestion view follows the job the worker runs to a terminal state; the
# worker extracts keyframes, transcribes and embeds the whole video first.
INGESTION_TIMEOUT_MS = 1_200_000
# Vespa answers a freshly fed document within seconds, but the suite's own
# ingestion helper allows a two-minute settle under load; match it.
SEARCH_INDEX_SETTLE_S = 180.0
# Span export is batched; the suite's other span readers wait this long.
SPAN_SETTLE_S = 300.0
# Phoenix serves an annotation shortly after it is written.
ANNOTATION_SETTLE_S = 90.0
VESPA_HTTP_PORT = 33080
ARGO_PHASES = ("Pending", "Running", "Succeeded", "Failed", "Error")


@pytest.fixture(scope="module")
def web_corpus() -> str:
    return ensure_web_tenant_corpus()


def _minted_tenant(prefix: str) -> str:
    tenant_id = canonical_tenant_id(unique_id(prefix))
    register_tenant_and_wait(tenant_id, created_by="e2e")
    return tenant_id


@pytest.fixture(scope="class")
def class_tenant() -> str:
    """A registered tenant this class owns and tears down."""
    return _minted_tenant("webe2e")


def _wait_for_span_count(
    phoenix_client, project: str, span_name: str, since: datetime, expected: int
) -> None:
    """Block until Phoenix holds exactly ``expected`` ``span_name`` spans, so
    the page's own count is an exact assertion rather than a race."""
    from phoenix.client.types.spans import SpanQuery

    query = SpanQuery().where(f"name == '{span_name}'")
    deadline = time.monotonic() + SPAN_SETTLE_S
    seen = -1
    while time.monotonic() < deadline:
        frame = phoenix_client.spans.get_spans_dataframe(
            project_identifier=project, query=query, start_time=since, timeout=30
        )
        seen = 0 if frame is None or frame.empty else len(frame)
        if seen == expected:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix holds {seen} {span_name} spans in project {project!r}; "
        f"expected {expected} within {SPAN_SETTLE_S:.0f}s"
    )


def _drive(agent: str, tenant_id: str) -> list[dict]:
    """Dispatch every suite video query to ``agent`` for ``tenant_id``."""
    bodies = []
    with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
        for query in GATEWAY_VIDEO_QUERIES:
            resp = client.post(
                f"/agents/{agent}/process",
                json={
                    "agent_name": agent,
                    "query": query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "success", body
            bodies.append(body)
    return bodies


class TestServingAndNavigation:
    def test_the_server_answers_health_the_page_and_the_runtimes_agents(self):
        health = httpx.get(f"{WEB}/healthz", timeout=30.0)
        assert (health.status_code, health.json()) == (200, {"status": "ok"})

        page = httpx.get(f"{WEB}/", timeout=30.0)
        assert page.status_code == 200
        assert page.headers["content-type"].split(";")[0] == "text/html"
        assert "<title>Cogniverse</title>" in page.text
        assert '<div id="root"></div>' in page.text

        listed = httpx.get(f"{WEB}/ui-api/agents", timeout=60.0)
        assert listed.status_code == 200, listed.text
        assert listed.json() == {"agents": registered_agents()}

    def test_the_sidebar_lists_every_agent_and_every_operations_view(self, page):
        agents = registered_agents()
        page.goto(f"{WEB}/", timeout=VIEW_TIMEOUT_MS)
        nav = page.get_by_role("navigation", name="Navigation")
        lists = nav.get_by_role("list")
        expect(lists).to_have_count(2, timeout=VIEW_TIMEOUT_MS)
        expect(nav.get_by_role("heading", level=2)).to_have_text(
            ["Agents", "Operations"]
        )
        expect(lists.nth(0).get_by_role("link")).to_have_text(
            [agent_label(name) for name in agents], timeout=VIEW_TIMEOUT_MS
        )
        expect(lists.nth(1).get_by_role("link")).to_have_text(
            [label for _, label in OPS_VIEWS]
        )
        # With no agent named in the address, the first registered one opens.
        expect(page.get_by_role("heading", level=1)).to_have_text(
            agent_label(agents[0]), timeout=VIEW_TIMEOUT_MS
        )
        assert thread_of(page, agents[0]) != ""

    def test_each_operations_view_opens_under_its_own_heading(self, page):
        page.goto(f"{WEB}/", timeout=VIEW_TIMEOUT_MS)
        nav = page.get_by_role("navigation", name="Navigation")
        for view, label in OPS_VIEWS:
            nav.get_by_role("link", name=label, exact=True).click()
            expect(page.get_by_role("heading", level=1)).to_have_text(label)
            assert page.evaluate("window.location.hash") == f"#/ops/{view}"
            expect(nav.get_by_role("link", name=label, exact=True)).to_have_attribute(
                "aria-current", "page"
            )

    def test_an_unknown_view_says_so(self, page):
        page.goto(f"{WEB}/#/ops/no-such-view", timeout=VIEW_TIMEOUT_MS)
        expect(page.locator("main .notice")).to_have_text(
            "There is no view named no-such-view."
        )


class TestAgentSearch:
    def test_a_search_renders_every_hit_of_the_run_as_a_card(self, page, web_corpus):
        """The cards are exactly the hits the run's final state carries, and
        the web tenant holds only the sample video, so every hit is of it."""
        open_agent(page, "search_agent")
        state = run_agent_and_capture_state(page, "search_agent", "sports activity")
        assert state["agent"] == "search_agent", state
        assert state["result"]["status"] == "success", state["result"]
        hits = state["result"]["results"]
        assert hits != [], state["result"]
        assert [SAMPLE_VIDEO_CONTENT_ID in json.dumps(hit) for hit in hits] == [
            True
        ] * len(hits), hits

        cards = result_cards(state)
        assert len(cards) == len(hits), (cards, hits)
        results = page.get_by_role("complementary", name="Results")
        expect(results.locator(".result-title")).to_have_text(
            [card["title"] for card in cards], timeout=RUN_TIMEOUT_MS
        )
        expect(results.locator(".result-score")).to_have_text(
            [card["score"] for card in cards if card["score"] is not None]
        )
        # One result list: a single agent's hits carry no group heading.
        expect(results.locator(".result-group")).to_have_count(0)
        span_id = state["result"]["span_id"]
        assert re.fullmatch(r"[0-9a-f]{16}", span_id), span_id
        expect(results.get_by_role("group")).to_have_count(len(cards))
        expect(results.get_by_role("group").first).to_have_attribute(
            "aria-label", f"Relevance of {cards[0]['rating_id']}"
        )
        assert results.get_by_role("group").evaluate_all(
            "groups => groups.map(g => g.getAttribute('aria-label'))"
        ) == [f"Relevance of {card['rating_id']}" for card in cards]
        # Every card offers exactly the three shipped ratings, none chosen yet.
        assert [
            group.get_by_role("button").all_inner_texts()
            for group in results.get_by_role("group").all()
        ] == [["Highly Relevant", "Somewhat Relevant", "Not Relevant"]] * len(cards)
        assert [
            [
                button.get_attribute("aria-pressed")
                for button in group.get_by_role("button").all()
            ]
            for group in results.get_by_role("group").all()
        ] == [["false", "false", "false"]] * len(cards)

    def test_a_rating_lands_on_the_searchs_span(
        self, page, web_corpus, phoenix_client_session
    ):
        open_agent(page, "search_agent")
        state = run_agent_and_capture_state(
            page, "search_agent", "sports throwing discus"
        )
        cards = result_cards(state)
        assert cards != [], state["result"]
        span_id = state["result"]["span_id"]
        rated = cards[0]["rating_id"]

        results = page.get_by_role("complementary", name="Results")
        group = results.get_by_role("group", name=f"Relevance of {rated}")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith("/ui-api/runtime/ag-ui/results/relevance")
            ),
            timeout=RUN_TIMEOUT_MS,
        ) as answered:
            group.get_by_role("button", name="Somewhat Relevant").click()
        stored = answered.value.json()
        assert answered.value.status == 200, stored
        assert {key: stored[key] for key in ("span_id", "result_id", "relevance")} == {
            "span_id": span_id,
            "result_id": rated,
            "relevance": "Somewhat Relevant",
        }
        expect(group.get_by_role("button", name="Somewhat Relevant")).to_have_attribute(
            "aria-pressed", "true"
        )
        expect(group.get_by_role("button", name="Highly Relevant")).to_have_attribute(
            "aria-pressed", "false"
        )
        expect(group.get_by_role("alert")).to_have_count(0)

        # The rating is the span's one relevance annotation, in the web
        # tenant's project, under the rated result.
        project = TelemetryConfig().get_project_name(WEB_TENANT)
        deadline = time.monotonic() + ANNOTATION_SETTLE_S
        ratings: dict = {}
        while time.monotonic() < deadline:
            frame = phoenix_client_session.spans.get_span_annotations_dataframe(
                span_ids=[span_id],
                project_identifier=project,
                include_annotation_names=[RESULT_RELEVANCE],
                timeout=30,
            )
            ratings = {
                TripletExtractor._annotation_result_id(row): (
                    row["result.label"],
                    row["result.score"],
                )
                for _, row in frame.iterrows()
            }
            if ratings:
                break
            time.sleep(3.0)
        assert ratings == {rated: ("Somewhat Relevant", stored["score"])}, ratings


class TestAgentConversation:
    @staticmethod
    def _plain_first_line(text: str) -> str:
        first = next(line for line in text.splitlines() if line.strip())
        return re.sub(r"[*_`#>]", "", first).strip()

    def test_a_turn_shows_the_message_and_the_reply_the_runtime_saved(
        self, page, web_corpus
    ):
        query = "What videos do you have about animals?"
        thread = open_agent(page, "gateway_agent")
        send(page, "gateway_agent", query)
        expect(assistant_messages(page)).to_have_count(1, timeout=RUN_TIMEOUT_MS)
        expect(user_messages(page)).to_have_text([query])

        saved = saved_turns(thread)
        assert (saved["thread_id"], saved["state"], saved["reason"]) == (
            thread,
            "loaded",
            None,
        ), saved
        assert [turn["role"] for turn in saved["turns"]] == ["user", "assistant"]
        assert saved["turns"][0]["content"] == query
        reply = saved["turns"][1]["content"]
        assert reply.strip() != "", saved
        # The reply is the gateway's rendered answer, never its raw payload.
        assert "document_id" not in reply, reply
        expect(assistant_messages(page)).to_contain_text(
            [self._plain_first_line(reply)]
        )

    def test_two_turns_survive_a_reload(self, page, web_corpus):
        turns = ("search for sports clips", "Tell me more about the first one")
        thread = open_agent(page, "gateway_agent")
        for count, text in enumerate(turns, start=1):
            send(page, "gateway_agent", text)
            expect(assistant_messages(page)).to_have_count(
                count, timeout=RUN_TIMEOUT_MS
            )
        expect(user_messages(page)).to_have_text(list(turns))

        saved = saved_turns(thread)
        assert [turn["role"] for turn in saved["turns"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ], saved
        assert [turn["content"] for turn in saved["turns"][0::2]] == list(turns)
        replies = [turn["content"] for turn in saved["turns"][1::2]]
        assert ["document_id" in reply for reply in replies] == [False, False]

        page.reload()
        expect(user_messages(page)).to_have_text(list(turns), timeout=VIEW_TIMEOUT_MS)
        expect(assistant_messages(page)).to_have_count(2)
        expect(assistant_messages(page)).to_contain_text(
            [self._plain_first_line(reply) for reply in replies]
        )
        assert thread_of(page, "gateway_agent") == thread

    def test_a_new_conversation_starts_empty_and_the_old_one_restores(
        self, page, web_corpus
    ):
        query = "search for sports clips"
        thread = open_agent(page, "gateway_agent")
        send(page, "gateway_agent", query)
        expect(assistant_messages(page)).to_have_count(1, timeout=RUN_TIMEOUT_MS)

        page.get_by_role("button", name="New conversation").click()
        expect(user_messages(page)).to_have_count(0)
        expect(assistant_messages(page)).to_have_count(0)
        fresh = thread_of(page, "gateway_agent")
        assert fresh != thread
        assert saved_turns(fresh)["turns"] == []

        page.go_back()
        expect(user_messages(page)).to_have_text([query], timeout=VIEW_TIMEOUT_MS)
        assert thread_of(page, "gateway_agent") == thread


class TestAgUiStreaming:
    """The runtime surface the web server relays: one AG-UI run per turn."""

    @staticmethod
    def _run(agent: str, text: str) -> tuple[str, list[dict]]:
        thread, run = f"web-e2e-{uuid.uuid4().hex}", f"run-{uuid.uuid4().hex}"
        response = httpx.post(
            f"{RUNTIME}/ag-ui/{agent}",
            json={
                "threadId": thread,
                "runId": run,
                "state": {},
                "messages": [{"id": "u1", "role": "user", "content": text}],
                "tools": [],
                "context": [],
                "forwardedProps": {},
            },
            headers={"Authorization": f"Bearer {web_harness_key()}"},
            timeout=900.0,
        )
        assert response.status_code == 200, response.text[:500]
        events = sse_events(response.text)
        assert events[0] == {"type": "RUN_STARTED", "threadId": thread, "runId": run}
        assert events[-1] == {
            "type": "RUN_FINISHED",
            "threadId": thread,
            "runId": run,
            "outcome": {"type": "success"},
        }, events[-3:]
        assert events[1:4] == [
            {"type": "STEP_STARTED", "stepName": "starting"},
            {
                "type": "CUSTOM",
                "name": "cogniverse.status",
                "value": {"phase": "starting", "message": f"Running {agent}"},
            },
            {"type": "STEP_FINISHED", "stepName": "starting"},
        ], events[:5]
        return thread, events

    @staticmethod
    def _reply(events: list[dict]) -> str:
        starts = [e for e in events if e["type"] == "TEXT_MESSAGE_START"]
        ends = [e for e in events if e["type"] == "TEXT_MESSAGE_END"]
        assert [(e["role"], e["messageId"]) for e in starts] == [
            ("assistant", ends[0]["messageId"])
        ], (starts, ends)
        assert len(ends) == 1, ends
        return "".join(
            e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"
        )

    def test_a_search_run_streams_its_phases_reply_and_results(self, web_corpus):
        thread, events = self._run("search_agent", "find nature videos")
        snapshots = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
        assert len(snapshots) == 1, [e["type"] for e in events]
        snapshot = snapshots[0]["snapshot"]
        assert snapshot["agent"] == "search_agent", snapshot
        assert snapshot["result"]["status"] == "success", snapshot["result"]
        assert [
            SAMPLE_VIDEO_CONTENT_ID in json.dumps(hit)
            for hit in snapshot["result"]["results"]
        ] == [True] * len(snapshot["result"]["results"])
        reply = self._reply(events)
        assert saved_turns(thread)["turns"] == [
            {"role": "user", "content": "find nature videos"},
            {"role": "assistant", "content": reply},
        ]

    def test_a_summarizer_run_streams_its_summary_as_one_reply(self, web_corpus):
        text = "summarize what video search technology does"
        thread, events = self._run("summarizer_agent", text)
        reply = self._reply(events)
        assert reply.strip() != "", events
        assert saved_turns(thread)["turns"] == [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ]


class TestTenantsView:
    def test_an_operator_creates_and_deletes_an_organization_and_its_tenant(self, page):
        org_id = unique_id("weborg")
        tenant_id = f"{org_id}:production"
        open_view(page, "tenants")

        form = page.get_by_role("form", name="Create organization")
        form.get_by_label("Organization ID").fill(org_id)
        form.get_by_label("Name").fill(f"Web e2e {org_id}")
        form.get_by_role("button", name="Create organization").click()
        expect(form.get_by_role("status")).to_have_text(
            f"Created organization {org_id}.", timeout=VIEW_TIMEOUT_MS
        )
        stored = httpx.get(f"{RUNTIME}/admin/organizations/{org_id}", timeout=30.0)
        assert stored.status_code == 200, stored.text
        assert (stored.json()["org_id"], stored.json()["org_name"]) == (
            org_id,
            f"Web e2e {org_id}",
        )
        org_row = (
            page.get_by_role("region", name="Organizations")
            .get_by_role("row")
            .filter(has=page.get_by_role("button", name=org_id, exact=True))
        )
        expect(org_row.get_by_role("cell")).to_have_text(
            [
                org_id,
                f"Web e2e {org_id}",
                "active",
                "0",
                re.compile(rf".+ by {re.escape(stored.json()['created_by'])}$"),
                "Delete",
            ]
        )

        org_row.get_by_role("button", name=org_id).click()
        tenants = page.get_by_role("region", name=f"Tenants of {org_id}")
        expect(tenants.get_by_text(f"No tenants in {org_id}.")).to_be_visible()
        tenant_form = page.get_by_role("form", name="Create tenant")
        tenant_form.get_by_label("Organization").fill(org_id)
        tenant_form.get_by_label("Tenant name").fill("production")
        tenant_form.get_by_role("button", name="Create tenant").click()
        expect(tenant_form.get_by_role("status")).to_contain_text(
            f"Created {tenant_id} with schemas ",
            timeout=TENANT_DEPLOY_TIMEOUT_S * 1000,
        )
        created = httpx.get(f"{RUNTIME}/admin/tenants/{tenant_id}", timeout=30.0)
        assert created.status_code == 200, created.text
        expect(tenant_form.get_by_role("status")).to_have_text(
            f"Created {tenant_id} with schemas "
            f"{', '.join(created.json()['schemas_deployed'])}."
        )
        expect(org_row.get_by_role("cell").nth(3)).to_have_text("1")

        tenant_row = tenants.get_by_role("row").filter(
            has=page.get_by_role("cell", name=tenant_id, exact=True)
        )
        tenant_row.get_by_role("button", name="Delete").click()
        confirm = tenant_row.get_by_role("button", name="Delete tenant")
        expect(confirm).to_be_disabled()
        tenant_row.get_by_label(f"Type {tenant_id} to delete this tenant").fill(
            tenant_id
        )
        confirm.click()
        expect(page.get_by_role("status").first).to_have_text(
            f"Deleted tenant {tenant_id} and organization {org_id}, which had "
            "no tenants left.",
            timeout=TENANT_DEPLOY_TIMEOUT_S * 1000,
        )
        expect(org_row).to_have_count(0)
        assert (
            httpx.get(f"{RUNTIME}/admin/tenants/{tenant_id}", timeout=30.0).status_code
            == 404
        )
        assert (
            httpx.get(
                f"{RUNTIME}/admin/organizations/{org_id}", timeout=30.0
            ).status_code
            == 404
        )


class TestConfigurationView:
    def test_the_forms_are_the_runtimes_config_sections(self, page, class_tenant):
        sections = httpx.get(f"{RUNTIME}/admin/config/sections", timeout=60.0)
        assert sections.status_code == 200, sections.text
        listed = sections.json()["sections"]
        open_view(page, "config")
        expect(
            page.get_by_role("form", name=re.compile(r"^Edit .+ config$"))
        ).to_have_count(
            len([s for s in listed if not s["tenant_scoped"]]),
            timeout=VIEW_TIMEOUT_MS,
        )
        for section in listed:
            if not section["tenant_scoped"]:
                expect(
                    page.get_by_role(
                        "form", name=f"Edit {section['title']} config", exact=True
                    )
                ).to_be_visible()

        choose_tenant(page, class_tenant, "Show configs")
        for section in listed:
            if section["tenant_scoped"] and section["service"] is not None:
                expect(
                    page.get_by_role(
                        "form",
                        name=f"Edit {section['title']} config of {class_tenant}",
                        exact=True,
                    )
                ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        agent_sections = [
            s for s in listed if s["tenant_scoped"] and s["service"] is None
        ]
        expect(
            page.get_by_role("region", name=f"Agents of {class_tenant}")
        ).to_have_count(len(agent_sections))

    def test_a_routing_edit_is_stored_and_an_earlier_version_restores(
        self, page, class_tenant
    ):
        open_view(page, "config")
        choose_tenant(page, class_tenant, "Show configs")
        form_name = f"Edit Routing config of {class_tenant}"
        for version, mode in enumerate(["direct", "adaptive"], start=1):
            form = page.get_by_role("form", name=form_name, exact=True)
            form.get_by_label("routing_mode", exact=True).fill(mode)
            form.get_by_role("button", name="Save").click()
            expect(page.get_by_role("status")).to_have_text(
                f"Saved Routing config of {class_tenant} as version {version}.",
                timeout=VIEW_TIMEOUT_MS,
            )

        stored = page.get_by_role("region", name=f"Stored configs of {class_tenant}")
        stored.get_by_role(
            "button", name="History of routing/gateway_agent/routing_config"
        ).click()
        history = stored.get_by_role(
            "region", name="History of routing/gateway_agent/routing_config"
        )
        summaries = history.locator("summary")
        expect(summaries).to_have_count(2)
        expect(summaries.nth(0)).to_contain_text("Version 2,")
        expect(summaries.nth(0)).to_contain_text("(current)")
        history.get_by_role("button", name="Restore version 1").click()
        expect(page.get_by_role("status")).to_have_text(
            "Restored version 1 of routing/gateway_agent/routing_config as version 3."
        )
        store = VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
        entry = store.get_config(
            tenant_id=class_tenant,
            scope=ConfigScope.ROUTING,
            service="gateway_agent",
            config_key="routing_config",
        )
        assert (entry.version, entry.config_value["routing_mode"]) == (3, "direct")
        expect(
            page.get_by_role("form", name=form_name, exact=True).get_by_label(
                "routing_mode", exact=True
            )
        ).to_have_value("direct")

    def test_an_export_imports_under_the_chosen_tenant_only(self, page, tmp_path):
        source = _minted_tenant("webcfgsrc")
        destination = _minted_tenant("webcfgdst")
        decoy = canonical_tenant_id(unique_id("webcfgdecoy"))
        store = VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
        written = {
            f"web_e2e_{index}": {"marker": f"{source}-{index}"} for index in range(3)
        }
        for key, value in written.items():
            store.set_config(
                tenant_id=source,
                scope=ConfigScope.AGENT,
                service="web_e2e",
                config_key=key,
                config_value=value,
            )

        def rows(tenant_id: str) -> set:
            return {
                (
                    entry.scope.value,
                    entry.service,
                    entry.config_key,
                    json.dumps(entry.config_value, sort_keys=True),
                )
                for entry in store.list_configs(tenant_id)
                if entry.scope != ConfigScope.SCHEMA
            }

        expected = {
            (ConfigScope.AGENT.value, "web_e2e", key, json.dumps(value, sort_keys=True))
            for key, value in written.items()
        }
        assert rows(source) == expected
        assert rows(destination) == set()

        open_view(page, "config")
        choose_tenant(page, source, "Show configs")
        transfer = page.get_by_role("region", name=f"Export and import for {source}")
        with page.expect_download() as download:
            transfer.get_by_role("group", name="Export configs").get_by_role(
                "button", name="Export configs"
            ).click()
        assert download.value.suggested_filename == (
            f"config_export_{source.replace(':', '_')}.json"
        )
        saved = tmp_path / "export.json"
        download.value.save_as(saved)
        exported = json.loads(saved.read_text())
        assert sorted(
            (c["scope"], c["service"], c["config_key"]) for c in exported["configs"]
        ) == sorted((ConfigScope.AGENT.value, "web_e2e", key) for key in written)

        # The file names a third tenant; the import files it under the one the
        # view shows.
        exported["tenant_id"] = decoy
        for entry in exported["configs"]:
            entry["tenant_id"] = decoy
        saved.write_text(json.dumps(exported))
        choose_tenant(page, destination, "Show configs")
        transfer = page.get_by_role(
            "region", name=f"Export and import for {destination}"
        )
        importer = transfer.get_by_role("form", name="Import configs")
        importer.get_by_label("Export file").set_input_files(str(saved))
        importer.get_by_role("button", name="Import configs").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Imported {len(written)} configs into {destination}.",
            timeout=VIEW_TIMEOUT_MS,
        )
        assert rows(destination) == expected
        assert rows(decoy) == set()
        assert rows(source) == expected

    def test_the_store_panel_reports_the_runtimes_store_facts(self, page):
        open_view(page, "config")
        panel = page.get_by_role("region", name="Config store").locator(
            'dl[aria-label="Config store facts"]'
        )
        expect(panel.locator("dd").first).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        stats = httpx.get(f"{RUNTIME}/admin/config/stats", timeout=60.0).json()
        shown = facts(panel)
        assert set(shown) == {"Backend", "Configs", "Versions", "Tenants", "By scope"}
        assert shown["Backend"] == stats["storage_backend"] == "vespa"

    def test_choosing_another_tenant_replaces_every_tenant_form(
        self, page, class_tenant
    ):
        other = _minted_tenant("webcfgother")
        open_view(page, "config")
        choose_tenant(page, class_tenant, "Show configs")
        expect(
            page.get_by_role(
                "form", name=f"Edit Routing config of {class_tenant}", exact=True
            )
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        choose_tenant(page, other, "Show configs")
        expect(
            page.get_by_role("form", name=f"Edit Routing config of {other}", exact=True)
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        expect(page.get_by_text(class_tenant)).to_have_count(0)


class TestBackendProfilesView:
    def test_the_web_tenants_profiles_are_the_runtimes(self, page, web_corpus):
        listed = httpx.get(
            f"{RUNTIME}/admin/profiles", params={"tenant_id": WEB_TENANT}, timeout=60.0
        )
        assert listed.status_code == 200, listed.text
        profiles = listed.json()["profiles"]
        assert PROFILE in [p["profile_name"] for p in profiles], profiles
        open_view(page, "profiles")
        choose_tenant(page, WEB_TENANT, "Show profiles")
        region = page.get_by_role("region", name=f"Profiles of {WEB_TENANT}")
        expect(region.locator("tbody tr")).to_have_count(
            len(profiles), timeout=VIEW_TIMEOUT_MS
        )
        assert [
            row.get_by_role("cell").all_inner_texts()
            for row in region.locator("tbody tr").all()
        ] == [
            [
                p["profile_name"],
                p["type"],
                p["schema_name"],
                p["embedding_model"],
                "yes" if p["schema_deployed"] else "no",
                p["description"],
            ]
            for p in profiles
        ]


class TestMemoryView:
    def test_an_operator_adds_finds_and_deletes_a_memory(self, page, class_tenant):
        text = f"web e2e memory {uuid.uuid4().hex[:8]} prefers dark mode"
        open_view(page, "memory")
        choose_tenant(page, class_tenant, "Show memories")
        panel = page.get_by_role(
            "region", name=f"Memories of _user_memories in {class_tenant}"
        )
        expect(panel.get_by_text("0 live, 0 archived.")).to_be_visible(
            timeout=VIEW_TIMEOUT_MS
        )

        add = page.get_by_role("form", name="Add memory")
        add.get_by_label("Memory").fill(text)
        add.get_by_label("Category").fill("ui")
        add.get_by_role("button", name="Save memory").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text("Saved memory ", timeout=VIEW_TIMEOUT_MS)
        memory_id = (
            notice.inner_text()
            .removeprefix("Saved memory ")
            .removesuffix(" to _user_memories.")
        )
        listing = httpx.get(
            f"{RUNTIME}/admin/tenant/{class_tenant}/memories", timeout=60.0
        )
        assert listing.status_code == 200, listing.text
        assert [m["id"] for m in listing.json()["memories"]] == [memory_id]

        panel = page.get_by_role(
            "region", name=f"Memories of _user_memories in {class_tenant}"
        )
        expect(panel.get_by_text("1 live, 0 archived.")).to_be_visible()
        rows = panel.get_by_role("table", name="Memories").locator("tbody tr")
        expect(rows.locator("td:nth-child(1)")).to_have_text([text])
        expect(rows.locator("td:nth-child(2)")).to_have_text(["ui"])
        expect(rows.locator("td:nth-child(4)")).to_have_text([memory_id])

        search = panel.get_by_role("form", name="Search memories")
        search.get_by_label("Query").fill("dark mode")
        search.get_by_role("button", name="Search").click()
        expect(rows.locator("td:nth-child(4)")).to_have_text(
            [memory_id], timeout=VIEW_TIMEOUT_MS
        )

        panel.get_by_role("button", name=f"Delete memory {memory_id}").click()
        panel.get_by_role("button", name=f"Confirm delete of {memory_id}").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Deleted memory {memory_id}.", timeout=VIEW_TIMEOUT_MS
        )
        listing = httpx.get(
            f"{RUNTIME}/admin/tenant/{class_tenant}/memories", timeout=60.0
        )
        assert [m["id"] for m in listing.json()["memories"]] == []

    def test_a_memory_stored_through_the_runtime_is_listed(self, page):
        tenant_id = _minted_tenant("webmem")
        text = f"web e2e listed memory {uuid.uuid4().hex[:8]}"
        seed = httpx.post(
            f"{RUNTIME}/admin/tenant/{tenant_id}/memories",
            json={"text": text},
            timeout=60.0,
        )
        assert seed.status_code == 200, seed.text[:300]
        assert seed.json()["status"] == "saved", seed.json()
        memory_id = seed.json()["id"]

        open_view(page, "memory")
        choose_tenant(page, tenant_id, "Show memories")
        panel = page.get_by_role(
            "region", name=f"Memories of _user_memories in {tenant_id}"
        )
        rows = panel.get_by_role("table", name="Memories").locator("tbody tr")
        expect(rows.locator("td:nth-child(4)")).to_have_text(
            [memory_id], timeout=VIEW_TIMEOUT_MS
        )
        expect(rows.locator("td:nth-child(1)")).to_have_text([text])


class TestIngestionView:
    def test_an_upload_is_followed_to_the_jobs_own_outcome(self, page):
        tenant_id = _minted_tenant("webingest")
        with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
        media_type = _sample_video_media_type(SAMPLE_VIDEO_PATH)
        expected_fed = _expected_sample_documents_fed(
            SAMPLE_VIDEO_PATH, PROFILE, media_type
        )
        filename = SAMPLE_VIDEO_PATH.name

        open_view(page, "ingestion")
        choose_tenant(page, tenant_id, "Use tenant")
        expect(page.get_by_role("region", name=f"Upload to {tenant_id}")).to_be_visible(
            timeout=VIEW_TIMEOUT_MS
        )
        form = page.get_by_role("form", name="Upload content")
        form.get_by_label("File", exact=True).set_input_files(str(SAMPLE_VIDEO_PATH))
        form.get_by_label("Profile").fill(PROFILE)
        form.get_by_role("button", name="Upload and ingest").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(
            f"Queued {filename} as ingest ", timeout=VIEW_TIMEOUT_MS
        )
        ingest_id = (
            notice.inner_text()
            .removeprefix(f"Queued {filename} as ingest ")
            .rstrip(".")
        )
        assert re.fullmatch(r"ingest_[0-9a-f]{32}", ingest_id), ingest_id

        row = (
            page.get_by_role("region", name="Ingests")
            .get_by_role("row")
            .filter(has=page.get_by_role("cell", name=ingest_id, exact=True))
        )
        expect(row.get_by_role("cell").nth(3)).to_have_text(
            "complete", timeout=INGESTION_TIMEOUT_MS
        )
        status = httpx.get(f"{RUNTIME}/ingestion/{ingest_id}/status", timeout=30.0)
        assert status.status_code == 200, status.text
        assert status.json()["state"] == "complete", status.json()
        result = status.json()["latest"]["result"]
        assert (result["video_id"], result["documents_fed"]) == (
            SAMPLE_VIDEO_CONTENT_ID,
            expected_fed,
        ), result
        outcome = (
            f"{SAMPLE_VIDEO_CONTENT_ID}: {result.get('chunks') or 0} chunks, "
            f"{expected_fed} documents fed."
        )
        if "graph_nodes" in result:
            outcome += (
                f" Graph: {result['graph_nodes']} nodes, "
                f"{result.get('graph_edges') or 0} edges."
            )
        expect(row.get_by_role("cell")).to_have_text(
            [ingest_id, filename, PROFILE, "complete", outcome]
        )

        # The documents the page reported are the documents the tenant serves.
        deadline = time.monotonic() + SEARCH_INDEX_SETTLE_S
        matches: list = []
        error = None
        while time.monotonic() < deadline:
            found, error = _search_sample_content(
                content_id=SAMPLE_VIDEO_CONTENT_ID,
                tenant_id=tenant_id,
                profile=PROFILE,
                suffix=SAMPLE_VIDEO_PATH.suffix,
                media_type=media_type,
            )
            matches = found or []
            if len(matches) == expected_fed:
                break
            time.sleep(2.0)
        assert error is None, error
        assert len(matches) == expected_fed

        # The same bytes again are the same ingest, followed rather than re-fed.
        form.get_by_label("File", exact=True).set_input_files(str(SAMPLE_VIDEO_PATH))
        form.get_by_role("button", name="Upload and ingest").click()
        expect(page.get_by_role("status")).to_have_text(
            f"{filename} matches ingest {ingest_id} (complete); following it.",
            timeout=VIEW_TIMEOUT_MS,
        )
        expect(
            page.get_by_role("region", name="Ingests").get_by_role("row")
        ).to_have_count(2)

    def test_the_upload_form_is_the_chosen_tenants(self, page):
        first = _minted_tenant("webingesta")
        second = _minted_tenant("webingestb")
        open_view(page, "ingestion")
        expect(page.get_by_role("form", name="Upload content")).to_have_count(0)
        choose_tenant(page, first, "Use tenant")
        expect(page.get_by_role("region", name=f"Upload to {first}")).to_be_visible(
            timeout=VIEW_TIMEOUT_MS
        )
        choose_tenant(page, second, "Use tenant")
        expect(page.get_by_role("region", name=f"Upload to {second}")).to_be_visible(
            timeout=VIEW_TIMEOUT_MS
        )
        expect(page.get_by_role("region", name=f"Upload to {first}")).to_have_count(0)
        expect(page.get_by_role("form", name="Upload content")).to_have_count(1)


class TestOptimizationRunsView:
    def test_a_started_run_is_listed_with_argos_phase(self, page):
        tenant_id = _minted_tenant("webopt")
        mode = "gateway-thresholds"
        open_view(page, "optimization")
        choose_tenant(page, tenant_id, "Show runs")
        runs = page.get_by_role("region", name=f"Optimization runs of {tenant_id}")
        expect(
            runs.get_by_text(f"No optimization runs for {tenant_id}.")
        ).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        form = page.get_by_role("form", name="Start optimization")
        expect(form.get_by_label("Mode").locator("option")).to_have_text(
            sorted(tenant_router._MANUAL_OPTIMIZE_MODES)
        )
        form.get_by_label("Mode").select_option(mode)
        form.get_by_role("button", name="Start run").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text(
            f"Started a {mode} run: manual-optimize-{mode}-", timeout=VIEW_TIMEOUT_MS
        )
        name = re.fullmatch(
            rf"Started a {mode} run: (manual-optimize-{mode}-\w+)\.",
            notice.inner_text(),
        ).group(1)

        def status() -> dict:
            response = httpx.get(
                f"{RUNTIME}/admin/tenant/{tenant_id}/optimize/runs/{name}",
                timeout=120.0,
            )
            assert response.status_code == 200, response.text
            return response.json()

        before = status()["phase"]
        row = runs.get_by_role("row").filter(
            has=page.get_by_role("button", name=name, exact=True)
        )
        expect(row.get_by_role("cell").nth(0)).to_have_text(name)
        expect(row.get_by_role("cell").nth(1)).to_have_text(mode)
        expect(row.get_by_role("cell").nth(2)).to_have_text("manual")
        detail = page.get_by_role("region", name=f"Run {name}")
        shown = detail.locator("dt:text-is('Phase') + dd")
        expect(shown).to_have_text(re.compile(rf"^({'|'.join(ARGO_PHASES)})$"))
        rendered = shown.inner_text()
        after = status()["phase"]
        assert rendered in argo_phases_between(before, after), (before, rendered, after)


class TestReviewViews:
    def test_an_empty_review_queue_says_so(self, page, class_tenant):
        open_view(page, "approvals")
        choose_tenant(page, class_tenant, "Show review queue")
        queue = page.get_by_role("region", name=f"Awaiting review in {class_tenant}")
        expect(queue.locator("p.muted")).to_have_text(
            f"Nothing awaits review in {class_tenant}.", timeout=VIEW_TIMEOUT_MS
        )
        served = httpx.get(
            f"{RUNTIME}/admin/tenant/{class_tenant}/approvals", timeout=60.0
        )
        assert (served.status_code, served.json()) == (200, {"items": []})

    def test_a_tenant_without_workflows_says_so(self, page, class_tenant):
        open_view(page, "workflows")
        choose_tenant(page, class_tenant, "Show workflows")
        panel = page.get_by_role("region", name=f"Workflows of {class_tenant}")
        expect(panel.locator("p.muted")).to_have_text(
            "No orchestration workflows in this window.", timeout=VIEW_TIMEOUT_MS
        )
        expect(panel.get_by_role("alert")).to_have_count(0)


class TestTelemetryViews:
    def test_profile_metrics_count_the_tenants_selections(
        self, page, phoenix_client_session
    ):
        tenant_id = _minted_tenant("webprofile")
        with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
        since = datetime.now(timezone.utc)
        modalities = {
            body["modality"] for body in _drive("profile_selection_agent", tenant_id)
        }
        assert modalities == {"video"}, modalities
        project = TelemetryConfig().get_project_name(tenant_id)
        count = len(GATEWAY_VIDEO_QUERIES)
        _wait_for_span_count(
            phoenix_client_session, project, SPAN_NAME_PROFILE_SELECTION, since, count
        )

        open_view(page, "profile-metrics")
        choose_tenant(page, tenant_id, "Show metrics")
        panel = page.get_by_role(
            "region", name=f"Profile selections of {tenant_id}", exact=True
        )
        table = panel.get_by_role("table", name="Selections by modality", exact=True)
        expect(table.locator("tbody tr")).to_have_count(1, timeout=VIEW_TIMEOUT_MS)
        expect(table.locator("tbody tr td").nth(0)).to_have_text("video")
        expect(table.locator("tbody tr td").nth(1)).to_have_text(str(count))
        expect(table.locator("tbody tr td").nth(5)).to_have_text("100.0%")
        figure = page.get_by_role("figure", name="Selections per modality", exact=True)
        expect(figure.locator(".bar-label")).to_have_text(["video"])
        expect(figure.locator(".bar-value")).to_have_text([str(count)])

    def test_routing_evaluation_counts_the_tenants_decisions(
        self, page, phoenix_client_session
    ):
        tenant_id = _minted_tenant("webrouting")
        with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
        since = datetime.now(timezone.utc)
        decisions = _drive("gateway_agent", tenant_id)
        assert [
            (body["gateway"]["complexity"], body["gateway"]["routed_to"])
            for body in decisions
        ] == [("simple", "search_agent")] * len(GATEWAY_VIDEO_QUERIES)
        project = TelemetryConfig().get_project_name(tenant_id)
        _wait_for_span_count(
            phoenix_client_session,
            project,
            SPAN_NAME_ROUTING,
            since,
            len(GATEWAY_VIDEO_QUERIES),
        )

        open_view(page, "routing")
        choose_tenant(page, tenant_id, "Show decisions")
        panel = page.get_by_role("region", name=f"Routing decisions of {tenant_id}")
        summary = panel.locator('dl[aria-label="Routing summary"]')
        expect(summary.locator("dd").first).to_have_text(
            str(len(GATEWAY_VIDEO_QUERIES)), timeout=VIEW_TIMEOUT_MS
        )
        shown = facts(summary)
        assert shown["Decisions"] == str(len(GATEWAY_VIDEO_QUERIES)), shown
        assert shown["Unreadable"] == "0", shown
        expect(
            panel.get_by_text("No routing decisions were recorded in this window.")
        ).to_have_count(0)

    def test_analytics_of_a_quiet_tenant_says_there_are_no_traces(
        self, page, class_tenant
    ):
        open_view(page, "analytics")
        choose_tenant(page, class_tenant, "Show traces")
        region = page.get_by_role(
            "region", name=f"Traces of {class_tenant}", exact=True
        )
        expect(region.locator("p.muted")).to_have_text(
            "No traces match in this window.", timeout=VIEW_TIMEOUT_MS
        )
        expect(region.get_by_role("alert")).to_have_count(0)

    def test_evaluation_without_a_golden_set_says_how_to_upload_one(
        self, page, class_tenant
    ):
        open_view(page, "evaluation")
        choose_tenant(page, class_tenant, "Evaluate")
        panel = page.get_by_role(
            "region", name=f"Golden set evaluation of {class_tenant}", exact=True
        )
        expect(panel.get_by_role("alert")).to_have_text(
            f"Tenant {class_tenant} has no golden set. Upload one with "
            f"PUT /admin/tenants/{class_tenant}/golden_set_ground_truth.",
            timeout=VIEW_TIMEOUT_MS,
        )
        expect(panel.locator('dl[aria-label="Evaluation summary"]')).to_have_count(0)

    def test_rlm_ab_without_comparisons_says_how_to_record_them(
        self, page, class_tenant
    ):
        open_view(page, "rlm-ab")
        choose_tenant(page, class_tenant, "Show comparisons")
        panel = page.get_by_role(
            "region", name=f"RLM A/B comparisons of {class_tenant}", exact=True
        )
        expect(panel.locator("p.muted")).to_have_text(
            "No comparisons in this window. Run cogniverse-optim --mode ab-compare "
            "for this tenant to record some.",
            timeout=VIEW_TIMEOUT_MS,
        )

    def test_the_atlas_shows_what_the_runtime_maps(self, page, web_corpus):
        served = httpx.get(
            f"{RUNTIME}/admin/tenant/{WEB_TENANT}/embeddings/atlas",
            params={"profile": PROFILE, "limit": 500},
            timeout=300.0,
        )
        open_view(page, "atlas")
        choose_tenant(page, WEB_TENANT, "Use tenant")
        form = page.get_by_role("form", name="Map documents")
        expect(form).to_be_visible(timeout=VIEW_TIMEOUT_MS)
        form.get_by_label("Documents").fill("500")
        form.get_by_label("Profile").select_option(PROFILE)
        form.get_by_role("button", name="Show map").click()
        if served.status_code != 200:
            # The page carries the runtime's own refusal, word for word.
            expect(form.get_by_role("alert")).to_have_text(
                served.json()["detail"], timeout=RUN_TIMEOUT_MS
            )
            expect(page.get_by_role("figure")).to_have_count(0)
            return
        expected = served.json()
        panel = page.get_by_role("region", name=f"Profile {PROFILE} of {WEB_TENANT}")
        table = panel.get_by_role("table", name="Mapped documents")
        expect(table.locator("tbody tr td:first-child")).to_have_text(
            sorted(point["title"] for point in expected["points"]),
            timeout=RUN_TIMEOUT_MS,
        )
        shown = facts(panel)
        assert (shown["Schema"], shown["Documents mapped"]) == (
            expected["schema_name"],
            str(len(expected["points"])),
        ), shown
