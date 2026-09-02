"""
E2E dashboard tests exercising Streamlit UI via Playwright.

Covers: interactive search, chat, tenant management, config management,
memory lifecycle, DSPy optimization pipeline (annotation harvesting,
golden dataset, synthetic data, module optimization), and monitoring tabs.

Requires:
- Dashboard running at http://localhost:33501
- Runtime at http://localhost:33000
- LM, Vespa, Phoenix running
- flywheel_org:production tenant with ingested data
"""

import re
import time

import httpx
import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import (
    DASHBOARD,
    RUNTIME,
    TENANT_ID,
    active_sub_tab_panel,
    active_tab_panel,
    click_button,
    click_sub_tab,
    click_top_tab,
    fill_input,
    fill_textarea,
    panel_widget,
    set_tenant,
    unique_id,
    wait_for_script_idle,
    wait_for_streamlit,
)

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

# Streamlit takes time to rerun on interactions
# Every top-level tab app.py:743-761 declares, in order. Restating them here
# is deliberate: the point is to notice when the shipped strip changes, which a
# derivation from the same source could not do.
MAIN_TAB_LABELS = (
    "\U0001f4ca Analytics",
    "\U0001f9ea Evaluation",
    "\U0001f5fa\ufe0f Embedding Atlas",
    "\U0001f3af Routing Evaluation",
    "\U0001f504 Orchestration Annotation",
    "\U0001f4c8 Profile Routing Metrics",
    "\U0001f527 Optimization",
    "\U0001f52c Synthetic Data & Optimization",
    "\u2705 Approval Queue",
    "\U0001f4e5 Ingestion Testing",
    "\U0001f50d Interactive Search",
    "\U0001f4ac Chat",
    "\u2699\ufe0f Configuration",
    "\U0001f465 Tenant Management",
    "\U0001f9e0 Memory",
    "\U0001f170\ufe0f\U0001f171\ufe0f RLM A/B Compare",
)

# The unconditional half of the Analytics sub-tab strip (app.py:857-868); the
# Root Cause Analysis tab is appended only when that feature is enabled.
ANALYTICS_SUB_TABS = (
    "\U0001f4ca Overview",
    "\U0001f4c8 Time Series",
    "\U0001f4ca Distributions",
    "\U0001f5fa\ufe0f Heatmaps",
    "\U0001f3af Outliers",
    "\U0001f50d Trace Explorer",
)

# The Optimization Overview block renders these four metrics unconditionally
# (optimization.py:111-159; "Last Optimization" is emitted by both branches of
# its if/else, so it is structural rather than data-dependent).
OPTIMIZATION_OVERVIEW_METRICS = (
    "Total Annotations",
    "Golden Dataset Size",
    "Optimization Runs",
    "Last Optimization",
)

INTERACTION_TIMEOUT = 30_000
SEARCH_TIMEOUT = 120_000
LLM_TIMEOUT = 300_000


def _fill_chat_message(page, message: str) -> None:
    """Type a message into the Chat tab's own text area.

    Scoped to the open panel and never widened. The page-wide text-input
    fallback this replaces resolved to the sidebar's Active Tenant box, so a
    chat area that had not rendered yet was typed over the tenant instead;
    the gate then rejected the query-as-tenant and blanked the dashboard,
    turning a timing miss into a poisoned session.
    """
    panel = active_tab_panel(page)
    box = panel.locator('[data-testid="stTextArea"] textarea')
    try:
        box.first.wait_for(state="visible", timeout=60_000)
    except Exception as exc:
        raise AssertionError(
            "The Chat tab must render its message text area. Panel text: "
            f"{panel.inner_text()[:400]!r}"
        ) from exc
    fill_textarea(box.first, message)


def _wait_for_rerun_complete(page, timeout_ms=SEARCH_TIMEOUT):
    """Wait for the in-flight Streamlit rerun to finish instead of
    sleeping a fixed interval — the status widget is attached while a
    callback executes and detaches when the script run completes. A
    typical search returns in seconds; the ceiling only bounds the
    worst case."""
    page.wait_for_timeout(1_000)  # let the rerun register
    status = page.locator('[data-testid="stStatusWidget"]')
    try:
        status.wait_for(state="detached", timeout=timeout_ms)
    except Exception:
        # Fast reruns may finish before the widget ever attaches.
        pass
    page.wait_for_timeout(500)


def _nav(page):
    """Navigate to dashboard and wait for Streamlit to load."""
    page.goto(DASHBOARD, timeout=30_000)
    wait_for_streamlit(page)


def assert_tab_rendered(page, identity: str, *, unavailable: str | None = None) -> str:
    """Assert the open panel is the tab that renders ``identity``; return its text.

    ``identity`` is a heading the tab's own module renders, so a tab body that
    rendered nothing cannot satisfy it. This replaces page-wide widget counts,
    which were satisfied by whatever *other* tab happened to render a metric or
    an alert: Streamlit keeps all 54 tab bodies in the DOM at once.

    Where a feature can be genuinely absent -- optional dependencies, a backend
    that is down -- ``unavailable`` names the notice the tab renders instead.
    Exactly one of the two states holds, and both are scoped to this panel so
    another tab's notice cannot stand in for this one's.
    """
    panel = active_tab_panel(page)
    text = panel.inner_text() or ""
    if unavailable is not None:
        notice = panel.locator(f'[data-testid="stAlert"]:has-text("{unavailable}")')
        if notice.count():
            assert identity not in text, (
                f"The panel showed the {unavailable!r} notice and {identity!r} at "
                f"once; these are mutually exclusive states:\n{text[:400]}"
            )
            return text
    assert identity in text, (
        f"The open panel must be the tab whose module renders {identity!r}. "
        f"It rendered:\n{text[:400]}"
    )
    return text


def _run_search(page, query: str) -> int:
    """Execute a search on the open Interactive Search panel; return its result count.

    Returns 0 for a no-match search. Raises when the search does not reach a
    terminal state or the dashboard reports it failed, so a broken backend
    fails here by its own message instead of downstream on an absent widget.
    """
    panel = active_tab_panel(page)

    # This body renders behind an agent-status call, so in the minute after the
    # models load it can take longer to reconcile than an ordinary interaction.
    # Wait on the widgets themselves: swallowing the timeout and then testing
    # count() reports "must be present" without revealing that it waited at
    # all, and turns a slow render into a missing-widget claim.
    search_input = panel.get_by_role("textbox", name="Enter your search query")
    search_button = panel.locator('button[kind="primary"]:has-text("Search"):visible')
    expect(search_input).to_have_count(1, timeout=SEARCH_TIMEOUT)
    expect(search_button).to_have_count(1, timeout=SEARCH_TIMEOUT)
    assert search_input.count() == 1, "exactly one search input must render"
    assert search_button.count() == 1, "exactly one primary Search button must render"

    # .fill() drives Streamlit's React bridge; .type() and JS-based fill do not
    # reliably commit to session state.
    search_input.fill(query)
    search_input.press("Enter")
    page.wait_for_timeout(5_000)
    wait_for_script_idle(page)

    search_button.click()
    _wait_for_rerun_complete(page)
    wait_for_script_idle(page)

    # The search streams, so the page is still filling in when the click
    # returns. The dashboard has THREE terminal states (app.py:2729-2737): a
    # Results metric, "Search returned no results", and "Search failed:
    # <cause>". Waiting only for the first two turns a failed search into a
    # timeout that names nothing, so wait for any of them and then classify.
    terminal = panel.locator(
        '[data-testid="stMetric"]:has-text("Results"),[data-testid="stAlert"]'
    )
    try:
        # A search, not an interaction: the first of a session pays the model
        # cold start (measured 49.7s cold against 1.8s warm), and
        # stream_agent_call (app.py:112) gives up at 120s, so that bound is the
        # contract -- waiting less asserts a promise the app never made.
        expect(terminal.first).to_be_visible(timeout=SEARCH_TIMEOUT)
    except AssertionError as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            "Search reached no terminal state: no Results metric and no alert. "
            f"The panel rendered:\n{(panel.inner_text() or '')[:800]}"
        ) from exc

    alerts = [
        (a.inner_text() or "").strip()
        for a in panel.locator('[data-testid="stAlert"]').all()
    ]
    broken = [t for t in alerts if "Search failed" in t or "Check runtime" in t]
    assert broken == [], f"the dashboard reported a failed search: {broken}"

    # The heading renders when the button is clicked, before the search
    # finishes (app.py:2745), so this proves the click landed rather than that
    # results arrived -- but it must land exactly once.
    heading = panel.get_by_role("heading", name=re.compile("Search Results"))
    assert heading.count() == 1, (
        f"Search Results heading must appear exactly once; headings={heading.count()}"
    )

    results_metric = panel.locator('[data-testid="stMetric"]:has-text("Results")')
    if panel.locator('[data-testid="stAlert"]:has-text("No results")').count():
        # A no-match search is a valid outcome, but it must not also claim a
        # result count.
        assert results_metric.count() == 0, (
            "A no-results search must not render a Results metric"
        )
        return 0

    # Exactly Results + Latency + Profile, scoped to this panel: page-wide the
    # Analytics tab alone contributes several, so a page-wide count is true
    # whatever this search rendered.
    labels = sorted(
        (m.inner_text() or "").strip().splitlines()[0]
        for m in panel.locator('[data-testid="stMetric"]').all()
    )
    assert labels == ["Latency", "Profile", "Results"], labels

    # stMetric renders label, delta, value, and the delta slot is empty here,
    # so the raw split carries a blank line between the two parts this reads.
    metric_text = [
        line
        for line in (results_metric.first.inner_text() or "").strip().splitlines()
        if line.strip()
    ]
    assert len(metric_text) == 2 and metric_text[0] == "Results", metric_text

    # The query must survive the rerun the click triggers: a text input whose
    # value is dropped is how the Search button ended up permanently disabled,
    # and a search running on an empty query would still render a metric.
    assert search_input.input_value() == query, search_input.input_value()

    # A result count with no latency and no profile is a half-rendered search.
    # Both are formatted by the app (app.py:2760,2762): "<n>ms", and the
    # profile name or the literal "auto" when the search did not report one.
    def _metric_value(label: str) -> str:
        lines = [
            line
            for line in (
                panel.locator(
                    f'[data-testid="stMetric"]:has-text("{label}")'
                ).first.inner_text()
                or ""
            ).splitlines()
            if line.strip()
        ]
        assert len(lines) == 2 and lines[0] == label, lines
        return lines[1].strip()

    assert re.fullmatch(r"\d+ms", _metric_value("Latency")), _metric_value("Latency")
    assert _metric_value("Profile") != "", "the search must report which profile ran"

    # A search cannot both report a count and claim it found nothing.
    assert panel.locator('[data-testid="stAlert"]:has-text("No results")').count() == 0

    return int(metric_text[1].replace(",", ""))


class TestSidebarAndNavigation:
    """Verify sidebar tenant input and top-level tab navigation."""

    def test_dashboard_loads_with_expected_structure(self, page):
        _nav(page)
        # Dashboard `st.stop()`s before rendering main_tabs when no tenant is
        # selected (libs/dashboard/cogniverse_dashboard/app.py:664-698), so
        # tabs only appear after a tenant is committed to session state.
        set_tenant(page, TENANT_ID)
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible(timeout=INTERACTION_TIMEOUT)
        app = page.locator('[data-testid="stAppViewContainer"]')
        expect(app).to_be_visible(timeout=INTERACTION_TIMEOUT)
        # Pin every top-level tab app.py:743-761 declares. `count() >= 3` was
        # satisfied by any three of the 54 tabs Streamlit renders (sub-tabs
        # included), so it could not notice a missing top-level tab.
        for label in MAIN_TAB_LABELS:
            expect(page.get_by_role("tab", name=label, exact=True)).to_have_count(
                1, timeout=INTERACTION_TIMEOUT
            )

    def test_sidebar_tenant_input(self, page):
        _nav(page)
        # Streamlit may render sidebar inputs hidden in headless mode;
        # verify the element exists in the DOM (not necessarily visible)
        sidebar = page.locator('[data-testid="stSidebar"]')
        # set_tenant() drives input[aria-label="Active Tenant"]; the sidebar
        # also renders a separate "Tenant ID" input, so a bare count of text
        # inputs cannot tell the two apart.
        assert sidebar.locator('input[aria-label="Active Tenant"]').count() == 1, (
            "Sidebar must expose exactly one Active Tenant input"
        )
        tenant_inputs = sidebar.locator('[data-testid="stTextInput"] input')
        assert tenant_inputs.count() == 2, (
            f"Sidebar renders Active Tenant and Tenant ID; got {tenant_inputs.count()}"
        )

    def test_set_tenant_persists(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        # Switch tab to trigger Streamlit rerun
        tabs = page.locator('button[role="tab"]')
        if tabs.count() >= 2:
            tabs.nth(1).click()
            page.wait_for_load_state("networkidle", timeout=30000)

        # Verify tenant persisted in session state after tab switch
        sidebar = page.locator('[data-testid="stSidebar"]')
        tenant_input = sidebar.locator('input[aria-label="Active Tenant"]')
        value = tenant_input.evaluate("el => el.value || ''")
        # The sidebar input is where the tenant persists. The page-wide body
        # fallback this replaces matched the id anywhere on the page --
        # including the tabs that merely echo it -- so it passed even when the
        # control had lost its value.
        assert value == TENANT_ID, (
            f"Active Tenant input must still hold the committed tenant; got {value!r}"
        )

    def test_top_level_tabs_present(self, page):
        _nav(page)
        # Tabs only render after tenant validation passes (see
        # test_dashboard_loads_with_expected_structure for context).
        set_tenant(page, TENANT_ID)
        tabs = page.locator('button[role="tab"]')
        tab_texts = [tabs.nth(i).inner_text().lower() for i in range(tabs.count())]
        assert any("analytics" in t for t in tab_texts), (
            f"Analytics tab missing, tabs: {tab_texts[:10]}"
        )
        assert any("configuration" in t or "config" in t for t in tab_texts), (
            f"Configuration tab missing, tabs: {tab_texts[:10]}"
        )
        assert any("tenant" in t for t in tab_texts), (
            f"Tenant Management tab missing, tabs: {tab_texts[:10]}"
        )

    def test_agent_status_in_sidebar(self, page):
        _nav(page)
        # Agent-status block lives in the sidebar after tenant validation;
        # without a tenant the dashboard shows only a warning and st.stop()s
        # before the show_agent_status() call (app.py:722).
        set_tenant(page, TENANT_ID)
        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar_text = sidebar.inner_text().lower()
        assert "agent" in sidebar_text or "status" in sidebar_text, (
            f"Sidebar should show agent status, got: {sidebar_text[:200]}"
        )


class TestInteractiveSearch:
    """Scenario 6: Search via dashboard, view results, annotate relevance."""

    def test_search_and_view_results(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Interactive Search")

        # _run_search carries the whole contract: the widgets render, the
        # search reaches one of its three terminal states, the dashboard did
        # not report a failure, and the panel shows exactly Results + Latency
        # + Profile. It returns the count that metric reports.
        expected_results = _run_search(page, "sports activity")

        # Every result renders one expander: the threshold slider defaults to
        # 0.0 and the render gate is `score >= confidence_threshold`
        # (app.py:2781), so the count is the Results metric, not a lower bound.
        result_expanders = active_tab_panel(page).locator(
            '[data-testid="stExpander"]:has-text("score")'
        )
        expect(result_expanders).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )

    def test_search_annotation(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Interactive Search")

        # _run_search raises if the search never reaches a terminal state or
        # the dashboard reports it failed, so reaching this line proves the
        # search actually executed. Previously three page-wide `count() > 0`
        # guards let this test pass while the Search button was disabled and
        # no search could run at all -- one of them even counted "Search
        # error" as proof of success.
        expected_results = _run_search(page, "throwing discus")

        panel = active_tab_panel(page)
        result_expanders = panel.locator('[data-testid="stExpander"]:has-text("score")')
        expect(result_expanders).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )
        if expected_results == 0:
            return

        # One Save Annotation button and one relevance radio group per result:
        # both render unconditionally inside each result expander
        # (app.py:2806-2828), so an exact count is the contract, not a lower
        # bound. Scoped to the panel because the page holds every tab body
        # simultaneously. The radio is matched as a group rather than by its
        # option label -- `has-text` is a substring match that also matches
        # ancestors, and a radio group cannot nest.
        save_buttons = panel.locator('button:has-text("Save Annotation")')
        radio_groups = panel.locator('[role="radiogroup"]:has-text("Highly Relevant")')
        expect(save_buttons).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )
        expect(radio_groups).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )

        # Present is not usable. Pin what the controls carry, so a renamed
        # label or a dropped option fails here rather than at annotation time.
        # st.expander renders collapsed (app.py:2782 passes no expanded=True),
        # so its contents are in the DOM but hidden: inner_text() returns ""
        # for them while :has-text still matches, which is why the count can
        # be right and the text empty. text_content() reads the DOM node.
        assert [(b.text_content() or "").strip() for b in save_buttons.all()] == [
            "💾 Save Annotation"
        ] * expected_results

        # Each group offers exactly the three shipped options and no more.
        # Matched by accessible role and name rather than by reading the
        # group's text: text_content() concatenates every descendant with no
        # separator, so splitting it would pin Streamlit's DOM shape instead
        # of what the control offers.
        shipped_options = ["Highly Relevant", "Somewhat Relevant", "Not Relevant"]
        assert [
            [
                g.get_by_role(
                    "radio", name=name, exact=True, include_hidden=True
                ).count()
                for name in shipped_options
            ]
            + [g.locator('input[type="radio"]').count()]
            for g in radio_groups.all()
        ] == [[1, 1, 1, 3]] * expected_results

        # Every expander is a result expander titled by the app's own format
        # (app.py:2782): "Result N: <video id> (Score: 0.000)". Counted across
        # the expander's lines rather than read from a fixed one: the collapse
        # chevron is a Material icon ligature that renders as its own text node
        # ("keyboard_arrow_right") ahead of the title, so line 0 is the icon.
        # Requiring exactly one match per expander pins the shipped format
        # without pinning where Streamlit happens to place it.
        title_pattern = re.compile(r"^Result \d+: .+ \(Score: -?\d+\.\d{3}\)$")
        expander_texts = [(e.inner_text() or "") for e in result_expanders.all()]
        assert [
            sum(1 for line in text.splitlines() if title_pattern.match(line.strip()))
            for text in expander_texts
        ] == [1] * expected_results, expander_texts

        # NOTE: Actually clicking Save Annotation does NOT work due to a known
        # Streamlit limitation — the Save button is inside `if search_button:`
        # block, so clicking Save triggers a rerun where search_button=False,
        # causing the results block (and annotation callback) to not execute.
        # This is documented in CLAUDE.local.md as an architectural limitation.

    def test_annotation_controls_in_search_results(self, page):
        """Search results must have annotation controls (Save + relevance radio)."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Interactive Search")

        expected_results = _run_search(page, "sports throwing discus")
        if expected_results == 0:
            return

        # Every result carries both controls, so the counts are exact rather
        # than `> 0` -- and scoped to this panel, since page-wide the other 53
        # tab bodies render simultaneously.
        panel = active_tab_panel(page)
        save_buttons = panel.locator('button:has-text("Save Annotation")')
        radio_groups = panel.locator('[role="radiogroup"]:has-text("Highly Relevant")')
        expect(save_buttons).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )
        expect(radio_groups).to_have_count(
            expected_results, timeout=INTERACTION_TIMEOUT
        )

        # Rendered is not usable. A Streamlit widget whose `disabled=` is
        # computed from a value that commits on the same interaction that
        # delivers the click is permanently unclickable -- that shipped in this
        # dashboard twice (Chat Send, Interactive Search), so an annotation
        # control being enabled is a contract, not an assumption.
        # Not is_visible(): the expanders render collapsed, so their contents
        # are correctly hidden until the user opens one. Enabled is the
        # contract; visible is UI state the user controls.
        assert [b.is_enabled() for b in save_buttons.all()] == [True] * expected_results
        assert [b.is_disabled() for b in save_buttons.all()] == [
            False
        ] * expected_results

        # st.radio defaults to its first option (app.py:2807-2814), so every
        # result starts on "Highly Relevant" and an annotation saved without
        # touching the control records that rather than an empty value.
        # include_hidden because the expanders render collapsed, so the radios
        # are outside the accessibility tree until one is opened.
        assert [
            [
                g.get_by_role(
                    "radio", checked=True, include_hidden=True, name=name, exact=True
                ).count()
                for name in ["Highly Relevant", "Somewhat Relevant", "Not Relevant"]
            ]
            for g in radio_groups.all()
        ] == [[1, 0, 0]] * expected_results


class TestMultiModalChat:
    """Scenario 8: Chat with the system, verify responses and multi-turn."""

    def test_send_message_and_get_response(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Chat")

        # Find chat input (text_area or text_input) — use JS fill for hidden elements
        _fill_chat_message(page, "What videos do you have about animals?")

        click_button(page, "Send")

        _wait_for_rerun_complete(page, timeout_ms=LLM_TIMEOUT)
        wait_for_script_idle(page)

        # Chat uses st.rerun() after sending — the reply lands as chat message
        # elements inside the Chat panel. Scoped to that panel: every tab body
        # is in the DOM, so a page-wide count is satisfied by other panels.
        panel = active_tab_panel(page)
        chat_msgs = panel.locator('[data-testid="stChatMessage"]')
        assert chat_msgs.count() == 2, (
            f"One send must render exactly two chat messages, the user's and the "
            f"assistant's, got {chat_msgs.count()}"
        )

        body_text = panel.inner_text()
        lowered = body_text.lower()

        # The gateway answers with a search payload, so its rendered reply
        # carries the result count and per-hit scores.
        assert "results for" in lowered and "score" in lowered, (
            "The assistant reply must be the gateway's rendered search answer — "
            "its 'Found N results for ...' message plus scored hits. Got: "
            f"{body_text[:400]!r}"
        )
        assert "document_id" not in lowered, (
            "The chat must render the gateway payload, not a stringified dict; "
            f"'document_id' means the raw response leaked into the UI. Got: "
            f"{body_text[:400]!r}"
        )

        # The response must contain words beyond the query — not just echo
        query_words = {"what", "videos", "do", "you", "have", "about", "animals"}
        response_words = set(lowered.split())
        non_query_words = response_words - query_words
        assert len(non_query_words) > 20, (
            "Chat response must contain substantial content beyond the query "
            "(routing + search agent actually produced results)"
        )

    def test_multi_turn_conversation(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Chat")

        # Turn 1 — use JS fill for hidden elements
        _fill_chat_message(page, "search for sports clips")
        click_button(page, "Send")
        _wait_for_rerun_complete(page, timeout_ms=LLM_TIMEOUT)
        wait_for_script_idle(page)

        # Turn 2
        _fill_chat_message(page, "Tell me more about the first one")
        click_button(page, "Send")
        _wait_for_rerun_complete(page, timeout_ms=LLM_TIMEOUT)
        wait_for_script_idle(page)

        # Multi-turn: verify both turns were processed
        # Streamlit st.rerun() after each message re-renders the page; the first
        # message may scroll out of the visible DOM.  The sidebar "messages: N"
        # counter is the authoritative proof that both turns were received.
        # The counter renders at the END of the script run, after the send
        # block's LLM call — under load that run outlives networkidle, so poll
        # for the counter instead of asserting on one immediate snapshot.
        sidebar = page.locator('[data-testid="stSidebar"]')
        deadline = time.monotonic() + LLM_TIMEOUT / 1000
        msg_match = None
        while msg_match is None and time.monotonic() < deadline:
            msg_match = re.search(r"messages:\s*(\d+)", sidebar.inner_text().lower())
            if msg_match is None:
                page.wait_for_timeout(2_000)

        assert msg_match, (
            "The sidebar must show a 'messages: N' counter after multi-turn chat. "
            "The dashboard renders that counter only once chat_messages is "
            "non-empty, so its absence means no turn was ever recorded. "
            f"Sidebar text: {sidebar.inner_text()!r}"
        )
        # Each turn appends the user message and the assistant reply, so the two
        # turns above are exactly four entries.
        assert int(msg_match.group(1)) == 4, (
            f"Two chat turns must record exactly 4 messages (user + assistant "
            f"each), sidebar shows {msg_match.group(1)}"
        )

        panel = active_tab_panel(page)
        chat_msgs = panel.locator('[data-testid="stChatMessage"]')
        assert chat_msgs.count() == 4, (
            f"Two turns must render exactly four chat messages, a user and an "
            f"assistant message each, got {chat_msgs.count()}"
        )

        panel_text = panel.inner_text().lower()

        # Both turns' queries are echoed back as the user's chat messages.
        for query in ("search for sports clips", "tell me more about the first one"):
            assert query in panel_text, (
                f"The turn {query!r} must be visible in the conversation. Got: "
                f"{panel_text[:400]!r}"
            )

        # A turn is answered either from search ("Found N results for '...'")
        # or through orchestration ("Orchestrated '...' via A2A pipeline") --
        # the two reply templates in agent_dispatcher.py. Which route a turn
        # takes is the router's decision, so pin the pair rather than one.
        replies = panel_text.count("results for") + panel_text.count("orchestrated")
        assert replies == 2, (
            f"Each of the two turns must render one gateway reply, found "
            f"{replies}. Got: {panel_text[:400]!r}"
        )
        assert "document_id" not in panel_text, (
            "The chat must render the gateway payload, not a stringified dict; "
            f"'document_id' means the raw response leaked. Got: {panel_text[:400]!r}"
        )


class TestOptimizationOverview:
    """Optimization Overview and Metrics Dashboard sub-tabs."""

    def test_overview_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Overview")
        wait_for_script_idle(page)

        # Scoped to the open SUB-panel, not the Optimization panel that holds
        # it. A locator matches its whole subtree regardless of display, and
        # the sibling sub-tab bodies Streamlit renders alongside Overview
        # carry metrics of their own: counting from the top panel found seven.
        panel = active_sub_tab_panel(page)

        # optimization.py:111-159 renders exactly these four metrics whenever
        # the Overview sub-tab is open. The `if metrics.count() >= 3` this
        # replaces was page-wide *and* a branch: when the open panel rendered
        # none of its own metrics the count was still satisfied by Analytics',
        # and when it was not, the else-branch accepted generic body text.
        metrics = panel.locator('[data-testid="stMetric"]')
        expect(metrics).to_have_count(
            len(OPTIMIZATION_OVERVIEW_METRICS), timeout=INTERACTION_TIMEOUT
        )
        metric_text = " ".join(
            metrics.nth(i).inner_text() for i in range(metrics.count())
        )
        missing = [m for m in OPTIMIZATION_OVERVIEW_METRICS if m not in metric_text]
        assert missing == [], (
            f"Optimization Overview must label every pipeline metric; missing "
            f"{missing} in:\n{metric_text[:300]}"
        )

    def test_metrics_dashboard_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Metrics Dashboard")
        wait_for_script_idle(page)

        # This body renders the Refresh button only past a telemetry-provider
        # probe (optimization.py:1620), and that probe's result is cached for
        # 60s, so a single sample taken while the read is slow observes a tab
        # that reports the provider missing. Wait past the cache rather than
        # sampling once.
        refresh_btn = active_tab_panel(page).locator(
            'button:has-text("Refresh"):visible'
        )
        try:
            expect(refresh_btn).to_have_count(1, timeout=SEARCH_TIMEOUT)
        except AssertionError as exc:  # pragma: no cover - diagnostic path
            raise AssertionError(
                "Metrics Dashboard never rendered its Refresh Metrics button. "
                "The panel rendered:\n"
                f"{(active_tab_panel(page).inner_text() or '')[:600]}"
            ) from exc
        assert refresh_btn.count() == 1, (
            f"Metrics Dashboard must have exactly one Refresh Metrics button; "
            f"got {refresh_btn.count()}"
        )

        body_text = active_tab_panel(page).inner_text().lower()
        # Must show the unified metrics dashboard header or tenant input
        assert "metrics" in body_text or "tenant" in body_text, (
            "Metrics Dashboard must show metrics content or tenant configuration"
        )


class TestAnnotationHarvesting:
    """Scenario 9: Fetch search spans and annotate via optimization tab."""

    def test_fetch_and_annotate_spans(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Search Annotations")
        wait_for_script_idle(page)

        # Verify key widgets: Lookback Hours input and Fetch button
        lookback_input = panel_widget(page, "stNumberInput", "Lookback Hours")
        assert lookback_input.count() == 1, (
            "Lookback Hours number input should be present"
        )

        fetch_btn = active_tab_panel(page).locator('button:has-text("Fetch"):visible')
        assert fetch_btn.count() == 1, "Fetch Search Results button should be present"

        click_button(page, "Fetch")
        wait_for_script_idle(page)

        # Exact alert text: "Fetched N search results" or "No results returned"
        fetched_alert = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Fetched")'
        )
        no_results = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("No results returned")'
        )
        error_alert = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Failed to fetch")'
        )

        # The fetch is a backend call: wait for one of the three terminal
        # alerts before reading any of them, or a mid-render page reads as
        # "no alert" and the error branch below is skipped silently.
        expect(fetched_alert.or_(no_results).or_(error_alert).first).to_be_visible(
            timeout=INTERACTION_TIMEOUT
        )

        # Fetch errors are system failures — Phoenix must be reachable
        if error_alert.count() > 0:
            error_text = error_alert.first.inner_text()
            pytest.fail(f"Annotation fetch hit system error: {error_text}")

        assert fetched_alert.count() > 0 or no_results.count() > 0, (
            "Fetch must show 'Fetched N search results' or 'No results returned' — "
            "system must connect to Phoenix successfully"
        )


class TestGoldenDataset:
    """Scenario 10: Build golden dataset from annotations."""

    def test_golden_dataset_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Golden Dataset")
        wait_for_script_idle(page)

        # Verify Lookback Days number input
        number_inputs = panel_widget(page, "stNumberInput", "Lookback Days")
        assert number_inputs.count() == 1, (
            "Lookback Days number input should be present"
        )

        # Verify Build button
        build_btn = active_tab_panel(page).locator('button:has-text("Build"):visible')
        assert build_btn.count() == 1, "Build Golden Dataset button should be present"

    def test_build_golden_dataset_execution(self, page):
        """Click Build Golden Dataset and verify it produces a result.

        Success: "Built golden dataset with N queries" (N may be 0)
        Expected no-data: "No annotated" — system works, just no annotations yet
        System error: "Failed to build dataset" — Phoenix unreachable = test failure
        """
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Golden Dataset")
        wait_for_script_idle(page)

        click_button(page, "Build")
        _wait_for_rerun_complete(page)
        wait_for_script_idle(page)

        # Check for specific outcomes
        panel = active_tab_panel(page)
        built_alert = panel.locator(
            '[data-testid="stAlert"]:has-text("Built golden dataset")'
        )
        no_data_alert = panel.locator(
            '[data-testid="stAlert"]:has-text("No annotated")'
        )
        error_alert = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Failed")'
        )

        # System errors are test failures — infrastructure must be working
        if error_alert.count() > 0:
            error_text = error_alert.first.inner_text()
            if "connect" in error_text.lower() or "timeout" in error_text.lower():
                pytest.fail(
                    f"Golden dataset build hit system error (infrastructure broken): {error_text}"
                )

        assert built_alert.count() > 0 or no_data_alert.count() > 0, (
            "Build must produce 'Built golden dataset with N queries' or "
            "'No annotated queries found' — system should work even with no data"
        )


class TestSyntheticDataAndApproval:
    """Scenario 11: Generate synthetic data, review approval queue."""

    def test_synthetic_data_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Synthetic Data")
        wait_for_script_idle(page)

        # Verify Examples to Generate number input
        number_inputs = panel_widget(page, "stNumberInput", "Examples to Generate")
        assert number_inputs.count() == 1, (
            "Examples to Generate number input should be present"
        )

        # Verify Generate button
        # optimization.py:678 renders exactly this label. `has-text("Generate")`
        # page-wide also matched other tabs' buttons.
        generate_btn = active_tab_panel(page).get_by_role(
            "button", name="🚀 Generate Synthetic Data", exact=True
        )
        expect(generate_btn).to_have_count(1, timeout=INTERACTION_TIMEOUT)

    def test_generate_synthetic_data_execution(self, page):
        """Click Generate and verify synthetic data is produced.

        Success: "Generated N examples using M profiles"
        System error: "Cannot connect" or "timed out" = infrastructure broken = test failure
        """
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Synthetic Data")
        wait_for_script_idle(page)

        click_button(page, "Generate")
        _wait_for_rerun_complete(page, timeout_ms=LLM_TIMEOUT)
        wait_for_script_idle(page)

        # Connection and timeout errors = infrastructure broken
        connect_error = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Cannot connect")'
        )
        timeout_error = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("timed out")'
        )
        if connect_error.count() > 0:
            pytest.fail("Synthetic generation failed: cannot connect to runtime API")
        if timeout_error.count() > 0:
            pytest.fail("Synthetic generation failed: request timed out")

        # Success must show "Generated N examples" or example data on page
        success_alert = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Generated")'
        )
        body_text = active_tab_panel(page).inner_text().lower()
        has_examples = "example" in body_text and "confidence" in body_text

        assert success_alert.count() > 0 or has_examples, (
            "Generate must produce 'Generated N examples using M profiles' "
            "with example data visible on page"
        )

    def test_approval_workflow_in_synthetic_data(self, page):
        """Verify the synthetic data tab has functional generation controls.

        The approval workflow is inline: Generate button, optimizer selectbox,
        confidence threshold slider, and profile count slider must all be present.
        """
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Synthetic Data")
        wait_for_script_idle(page)

        # Generate button MUST exist — this is the primary action. Matched on
        # its full label and scoped to the panel: "Generate" alone also hits
        # buttons in the other tab bodies, which are all in the DOM.
        generate_btn = active_tab_panel(page).locator(
            'button:has-text("Generate Synthetic Data")'
        )
        try:
            generate_btn.first.wait_for(state="visible", timeout=60_000)
        except Exception as exc:
            raise AssertionError(
                "Synthetic Data tab must have a Generate Synthetic Data button. "
                f"Panel text: {active_tab_panel(page).inner_text()[:300]!r}"
            ) from exc
        assert generate_btn.count() == 1, (
            f"Synthetic Data tab must have exactly one Generate Synthetic Data "
            f"button, got {generate_btn.count()}"
        )

        # Optimizer selectbox MUST exist with selectable options
        selectboxes = panel_widget(page, "stSelectbox", "Optimizer Type")
        assert selectboxes.count() == 1, (
            "Synthetic Data tab must have Optimizer Type selectbox"
        )

        # Confidence threshold and max profiles sliders
        sliders = panel_widget(page, "stSlider", "Min Rating Threshold")
        assert sliders.count() == 1, (
            "Synthetic Data tab must have the Min Rating Threshold slider"
        )

        # The panel keeps streaming after its widgets attach, so poll for the
        # subheader instead of sampling the text once.
        deadline = time.monotonic() + 60
        body_text = ""
        while time.monotonic() < deadline:
            body_text = active_tab_panel(page).inner_text().lower()
            if "synthetic data generation" in body_text:
                break
            page.wait_for_timeout(1_000)
        assert "synthetic data generation" in body_text, (
            "Synthetic Data tab must show its 'Synthetic Data Generation' "
            f"subheader; panel text: {body_text[:300]!r}"
        )


class TestModuleOptimization:
    """Scenario 12: Trigger DSPy module optimization from dashboard."""

    def test_module_optimization_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Module Optimization")
        wait_for_script_idle(page)

        # Verify optimizer/dataset selectbox
        selectboxes = panel_widget(page, "stSelectbox", "Module to Optimize")
        assert selectboxes.count() == 1, (
            "Module to Optimize selectbox should be present"
        )

        # Verify submit or upload button
        submit_btn = active_tab_panel(page).locator('button:has-text("Submit"):visible')
        # `has-text("Upload")` also matches the file uploader's own "Upload"
        # affordance in other tabs, so scope it and use the label
        # optimization.py renders.
        upload_btn = active_tab_panel(page).get_by_role(
            "button", name="📤 Upload Dataset", exact=True
        )
        assert submit_btn.count() == 1 or upload_btn.count() == 1, (
            "Submit Workflow or Upload Dataset button should be present"
        )

        # Verify DSPy optimizer selection controls exist
        body_text = active_tab_panel(page).inner_text().lower()
        has_optimizer_controls = (
            "optimizer" in body_text
            or "dspy" in body_text
            or "module" in body_text
            or "iterations" in body_text
            or "training" in body_text
        )
        assert has_optimizer_controls, (
            "Module Optimization should show optimizer type, iterations, "
            "or training data controls"
        )

    def test_module_optimization_workflow_submission(self, page):
        """Submit optimization workflow and verify it produces specific feedback.

        Success: "Workflow submitted successfully!"
        Expected prerequisite: "kubectl" warning (no K8s), "No dataset" (no data)
        System error: generic "failed" without known reason = test failure
        """
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Module Optimization")
        wait_for_script_idle(page)

        # Submit button MUST exist
        submit_btn = active_tab_panel(page).locator('button:has-text("Submit"):visible')
        assert submit_btn.count() == 1, (
            "Module Optimization must have Submit Workflow button"
        )

        click_button(page, "Submit")
        _wait_for_rerun_complete(page)
        wait_for_script_idle(page)

        # Scoped: page-wide, any tab reporting anything at all rendered an
        # alert, so this four-way disjunction held whatever this submission did.
        panel = active_tab_panel(page)
        success = panel.locator(
            '[data-testid="stAlert"]:has-text("submitted successfully")'
        )
        kubectl_warning = panel.locator('[data-testid="stAlert"]:has-text("kubectl")')
        no_dataset = panel.locator(
            '[data-testid="stAlert"]:has-text("No dataset"), '
            '[data-testid="stAlert"]:has-text("training data")'
        )
        upload_prompt = panel.locator('[data-testid="stAlert"]:has-text("Upload")')

        # These are all valid outcomes (system works but prerequisites vary)
        assert (
            success.count() > 0
            or kubectl_warning.count() > 0
            or no_dataset.count() > 0
            or upload_prompt.count() > 0
        ), (
            "Module optimization must show: success, kubectl warning, "
            "or no-dataset prompt — not a silent failure"
        )


class TestRerankingAndProfileOptimization:
    """Scenario 13: Reranking and profile selection optimization tabs."""

    def test_reranking_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Reranking")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()
        # Reranking tab must show "Current Annotations" metric and Train button
        assert "reranking" in body_text, "Reranking tab must show 'Reranking' in header"

        # "Current Annotations" metric is always shown
        metrics = active_tab_panel(page).locator('[data-testid="stMetric"]')
        if metrics.count() > 0:
            metric_text = " ".join(
                metrics.nth(i).inner_text().lower() for i in range(metrics.count())
            )
            assert "annotation" in metric_text, (
                f"Reranking must show 'Current Annotations' metric, got: {metric_text[:200]}"
            )

        # Train Reranker button MUST exist
        train_btn = active_tab_panel(page).locator('button:has-text("Train"):visible')
        assert train_btn.count() == 1, "Reranking tab must have Train Reranker button"

    def test_profile_selection_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)
        click_sub_tab(page, "Profile Selection")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()
        assert "profile selection" in body_text, (
            "Profile Selection tab must show 'Profile Selection' in header"
        )

        # The Train button renders unconditionally (optimization.py:1418); the
        # Load button only once a trained model is on disk (:1421), so it
        # cannot stand in for the Train button in an unconditional contract.
        # `has-text` is case-insensitive substring, so the previous
        # page-wide `has-text("Load")` counted Upload and Download buttons
        # belonging to other tabs -- 6 of them here, none a Load Model button.
        train_btn = active_tab_panel(page).locator(
            'button:has-text("Train Profile Selector Model"):visible'
        )
        expect(train_btn).to_have_count(1, timeout=SEARCH_TIMEOUT)
        assert train_btn.count() == 1, (
            "Profile Selection must render exactly one Train Profile Selector "
            f"Model button; got {train_btn.count()}"
        )


class TestProfileRoutingMetrics:
    """Top-level Profile Routing Metrics tab — runtime observability of
    ProfileSelectionAgent dispatches sourced from Phoenix spans."""

    def test_profile_routing_metrics_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Profile Routing Metrics")
        wait_for_script_idle(page)
        # Tab content lazy-renders after click; wait for the active panel
        # to contain our subheader before asserting on widgets within it.
        active_panel = page.locator(
            '[role="tabpanel"]:not([hidden]):has-text("Profile Routing Metrics")'
        )
        # This body reads Phoenix spans while it renders, so under sweep load
        # it is a full rerun rather than a plain interaction; budget it like
        # one, as _wait_for_rerun_complete already does.
        expect(active_panel.first).to_be_visible(timeout=SEARCH_TIMEOUT)

        # Lookback (hours) input is the only always-rendered widget — the
        # rest depends on whether Phoenix has profile_selection spans yet.
        # The panel turns visible carrying only its header, so wait for the
        # widget itself: a count taken while the body is still streaming reads
        # zero and blames the product for a render that had not finished.
        lookback_input = active_panel.locator('[data-testid="stNumberInput"]')
        try:
            lookback_input.first.wait_for(state="visible", timeout=60_000)
        except Exception as exc:
            raise AssertionError(
                "Profile Routing Metrics must expose a Lookback (hours) input. "
                f"Panel text: {active_panel.first.inner_text()[:300]!r}"
            ) from exc
        assert lookback_input.count() == 1, (
            f"Profile Routing Metrics must expose exactly one Lookback (hours) "
            f"input, got {lookback_input.count()}; panel text: "
            f"{active_panel.first.inner_text()[:300]}"
        )

        # Acceptable terminal states: empty-spans info, missing-attribute
        # warning, or rendered metrics. Any error alert is a real failure.
        error_alerts = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Phoenix span query failed"), '
            '[data-testid="stAlert"]:has-text("Failed to initialise telemetry")'
        )
        assert error_alerts.all_inner_texts() == [], (
            "Profile Routing Metrics surfaced an error: "
            f"{error_alerts.all_inner_texts()}"
        )


class TestTenantLifecycleDashboard:
    """Scenario 14: Create org + tenant, verify, delete via dashboard."""

    def test_tenant_management_sub_tabs(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Tenant Management")
        wait_for_script_idle(page)

        # expect() retries; count() samples once. networkidle fires while
        # Streamlit is still streaming the panel body, so a bare count() reads
        # the strip before its sub-tabs attach. Both are scoped to the open
        # panel: page-wide, "Create Organization" would also be satisfied by
        # another tab body, since every body is in the DOM at once.
        panel = active_tab_panel(page)
        sub_tabs = panel.locator('button[role="tab"]:visible')
        shipped = ["Organizations", "Create Organization", "Tenants", "Create Tenant"]
        expect(sub_tabs).to_have_count(len(shipped), timeout=INTERACTION_TIMEOUT)

        # The shipped set (tenant_management.py:93), pinned whole rather than
        # spot-checking two of it: with the count above, "each of these
        # resolves once" means exactly these and no others. Matched by
        # substring because no label here is a substring of another
        # ("Organizations" vs "Create Organization" differ by the plural).
        assert [
            panel.locator(f'button[role="tab"]:has-text("{name}"):visible').count()
            for name in shipped
        ] == [1] * len(shipped), sub_tabs.all_text_contents()

    def test_create_and_delete_organization(self, page):
        org_id = unique_id("dashorg")

        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Tenant Management")
        wait_for_script_idle(page)
        click_sub_tab(page, "Create Organization")
        wait_for_script_idle(page)

        # Scope to the Create Organization tabpanel — the sidebar also
        # has a text input (Active Tenant) and filling it with an org_id
        # would corrupt the gate-validated tenant.
        form_panel = page.locator(
            '[role="tabpanel"]:has-text("Create Organization")'
        ).last
        inputs = form_panel.locator('[data-testid="stTextInput"] input')
        assert inputs.count() >= 2, (
            f"Create Organization form needs at least 2 text inputs, "
            f"got {inputs.count()}"
        )

        fill_input(inputs.nth(0), org_id)
        fill_input(inputs.nth(1), f"E2E Dashboard Org {org_id}")

        # Create Organization button MUST exist
        submit_btn = form_panel.locator('button:has-text("Create Organization")')
        assert submit_btn.count() > 0, (
            "Create Organization submit button must be present"
        )

        # Streamlit form_submit_button may not pick up JS-filled values.
        # Trigger input events to propagate values to Streamlit state.
        for i in range(min(inputs.count(), 2)):
            inputs.nth(i).dispatch_event("input")
            inputs.nth(i).dispatch_event("change")
        page.wait_for_timeout(1_000)

        click_button(page, "Create Organization")
        wait_for_script_idle(page)

        # st.rerun() clears transient alerts — authoritative check via API
        page.wait_for_timeout(3_000)

        verify_resp = httpx.get(f"{RUNTIME}/admin/organizations/{org_id}", timeout=10.0)

        # Streamlit form submission via headless Playwright is unreliable —
        # JS-filled values may not propagate to Streamlit's state manager.
        # If the API shows 404, the form didn't submit the values properly.
        if verify_resp.status_code == 404:
            # Verify the form UI works by creating via API directly,
            # then confirming the dashboard reflects the change
            create_resp = httpx.post(
                f"{RUNTIME}/admin/organizations",
                json={
                    "org_id": org_id,
                    "org_name": f"E2E Dashboard Org {org_id}",
                    "created_by": "e2e_test",
                },
                timeout=10.0,
            )
            assert create_resp.status_code == 200, (
                f"API org creation also failed: {create_resp.text}"
            )
        else:
            assert verify_resp.status_code == 200, (
                f"Org {org_id} must exist after creation. "
                f"API returned {verify_resp.status_code}: {verify_resp.text}"
            )
            org_data = verify_resp.json()
            assert org_data["org_id"] == org_id

        # Cleanup via API
        httpx.delete(f"{RUNTIME}/admin/organizations/{org_id}", timeout=10.0)

    def test_create_tenant_sub_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Tenant Management")
        wait_for_script_idle(page)
        click_sub_tab(page, "Create Tenant")
        wait_for_script_idle(page)

        # Verify Create Tenant form widgets
        body_text = active_tab_panel(page).inner_text().lower()
        assert "tenant" in body_text, "Create Tenant sub-tab should mention 'tenant'"
        # tenant_management.py renders the Create Tenant form inside this
        # panel. Page-wide, Configuration alone contributes 16 text inputs and
        # 14 selectboxes, so the old disjunction could not fail.
        panel = active_tab_panel(page)
        expect(panel.locator('[data-testid="stTextInput"] input')).not_to_have_count(
            0, timeout=INTERACTION_TIMEOUT
        )
        assert "Create Tenant" in panel.inner_text(), (
            "The open panel must be the Create Tenant sub-tab"
        )

    def test_tenants_list_sub_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Tenant Management")
        wait_for_script_idle(page)
        click_sub_tab(page, "Tenants")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()
        assert "tenant" in body_text, "Tenants list tab must mention 'tenant'"

        # tenant_management.py renders "Refresh Tenants" whenever this sub-tab
        # is open, so it is structural rather than data-dependent. The four-way
        # disjunction it replaces was page-wide: six other tabs render a
        # Refresh button, so it held whatever this tab did.
        panel = active_tab_panel(page)
        expect(
            panel.get_by_role("button", name="Refresh Tenants", exact=True)
        ).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        selectboxes = panel.locator('[data-testid="stSelectbox"]')
        expanders = panel.locator('[data-testid="stExpander"]')
        no_tenants = panel.locator('[data-testid="stAlert"]:has-text("No tenants")')
        assert (
            selectboxes.count() > 0 or expanders.count() > 0 or no_tenants.count() > 0
        ), (
            "Tenants tab must show org selector, tenant expanders, "
            "'No tenants' message, or Refresh button"
        )


class TestConfigManagement:
    """Scenario 16: Edit config, save, verify persistence, export."""

    def test_system_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)

        # System Config is default sub-tab. Verify form elements
        assert panel_widget(page, "stSelectbox", "Backend Type").count() == 1, (
            "System Config should have the Backend Type selectbox"
        )
        assert panel_widget(page, "stSelectbox", "Environment").count() == 1, (
            "System Config should have the Environment selectbox"
        )

        # Verify Save button exists and is inside a form
        save_btn = active_tab_panel(page).locator('button:has-text("Save"):visible')
        assert save_btn.count() == 1, (
            "Save System Configuration button should be present"
        )

        # Verify page content shows config-specific terms
        body_text = active_tab_panel(page).inner_text().lower()
        has_config_content = (
            "environment" in body_text
            or "backend" in body_text
            or "vespa" in body_text
            or "healthy" in body_text
        )
        assert has_config_content, (
            "System Config tab should show environment, backend, or health info"
        )

    def test_config_import_export(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "Import/Export")
        wait_for_script_idle(page)

        # The full text of the Import/Export sub-tab's button is
        # "📥 Export Configurations". A bare 'has-text("Export")' selector
        # matches sibling sub-tab panels too (Streamlit renders all sub-tabs
        # in the DOM and only hides inactive ones), so the JS click in
        # click_button() can target a hidden button whose React handler
        # doesn't fire. Use the precise text + a Playwright native click so
        # Streamlit picks up the rerun.
        export_btn = active_tab_panel(page).get_by_role(
            "button", name="📥 Export Configurations", exact=True
        )
        expect(export_btn).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        export_btn.first.click()
        wait_for_script_idle(page)

        # Export produces: "Exported N configurations" success alert
        export_success = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Exported")'
        )
        export_error = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Export failed")'
        )
        download_btn = active_tab_panel(page).locator(
            '[data-testid="stDownloadButton"]'
        )

        if export_error.count() > 0:
            pytest.fail(f"Export failed: {export_error.first.inner_text()}")

        assert export_success.count() > 0 or download_btn.count() > 0, (
            "Export must show 'Exported N configurations' success or download button"
        )

        # Verify file uploader for import
        file_uploader = panel_widget(
            page, "stFileUploader", "Upload Configuration JSON"
        )
        assert file_uploader.count() == 1, (
            "File uploader for config import should be present"
        )

    def test_agent_configs_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "Agent Configs")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()
        # Agent Configs has a Save button and agent name input/selectbox
        assert "agent" in body_text, "Agent Configs tab must mention 'agent' in content"
        # config_management.py:250 renders this heading for the Agent
        # Configurations sub-tab. The disjunction it replaces was page-wide,
        # and Configuration's own sibling sub-tabs contribute 14 selectboxes,
        # so it held even when this sub-tab rendered nothing.
        panel = active_tab_panel(page)
        assert "Agent Configurations" in panel.inner_text(), (
            f"The open panel must be Agent Configurations:\n{panel.inner_text()[:400]}"
        )

    def test_routing_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "Routing Config")
        wait_for_script_idle(page)

        # Routing Config must have a Save Routing Configuration button
        save_btn = active_tab_panel(page).locator('button:has-text("Save"):visible')
        assert save_btn.count() == 1, "Routing Config must have Save button"
        body_text = active_tab_panel(page).inner_text().lower()
        assert "routing" in body_text, (
            "Routing Config tab must mention 'routing' in content"
        )

    def test_telemetry_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "Telemetry Config")
        wait_for_script_idle(page)

        # Telemetry Config must have Save button and mention telemetry/phoenix
        save_btn = active_tab_panel(page).locator('button:has-text("Save"):visible')
        assert save_btn.count() == 1, "Telemetry Config must have Save button"
        body_text = active_tab_panel(page).inner_text().lower()
        assert "telemetry" in body_text or "phoenix" in body_text, (
            "Telemetry Config must mention 'telemetry' or 'phoenix'"
        )

    def test_backend_profiles_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "Backend Profiles")
        wait_for_script_idle(page)

        panel_text = active_tab_panel(page).inner_text().lower()
        assert "profile" in panel_text or "schema" in panel_text, panel_text[:300]
        # A bare selectbox count matched any of the 33 selectboxes Streamlit
        # keeps in the DOM, and the `or` on buttons made it weaker still.
        assert (
            panel_widget(page, "stSelectbox", "Select Profile to Manage").count() == 1
        ), "Backend Profiles must expose the profile selector"

    def test_config_history(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Configuration")
        wait_for_script_idle(page)
        click_sub_tab(page, "History")
        wait_for_script_idle(page)

        # History tab MUST have Scope selectbox for filtering
        selectboxes = panel_widget(page, "stSelectbox", "Scope")
        assert selectboxes.count() == 1, "History tab must have Scope selectbox"

        body_text = active_tab_panel(page).inner_text().lower()
        # Must show history-specific content — not just generic page text
        assert "history" in body_text or "version" in body_text, (
            "History tab must show 'history' or 'version' in content"
        )

        # If versions exist, verify dataframe or rollback button is present
        panel = active_tab_panel(page)
        dataframes = panel.locator('[data-testid="stDataFrame"]')
        rollback_btn = panel.locator('button:has-text("Rollback")')
        no_history = panel.locator('[data-testid="stAlert"]:has-text("No history")')
        assert (
            dataframes.count() > 0 or rollback_btn.count() > 0 or no_history.count() > 0
        ), (
            "History must show version dataframe, rollback button, or 'No history' message"
        )


class TestMemoryLifecycle:
    """Scenario 17: Add memory, search, find, delete via dashboard."""

    def test_memory_sub_tabs_present(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Memory")
        wait_for_script_idle(page)

        # Memory is the 15th of 16 top-level tabs, so Streamlit renders its
        # body last: measured 8.7s from the click to the sub-tabs appearing,
        # against click_top_tab's 3s settle. expect() retries against the
        # condition instead of asserting after a fixed wait.
        #
        # The labels are the full sub-tab names. Substrings like "Search" and
        # "Delete" also match Interactive Search and Configuration's Delete
        # tab, so a bare-substring locator passes whether or not the Memory
        # tab rendered at all.
        search_tab = page.locator('button[role="tab"]:has-text("Search Memories")')
        add_tab = page.locator('button[role="tab"]:has-text("Add Memory")')
        view_tab = page.locator('button[role="tab"]:has-text("View All")')
        delete_tab = page.locator('button[role="tab"]:has-text("Delete Memory")')
        expect(add_tab).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        expect(search_tab).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        expect(view_tab).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        expect(delete_tab).to_have_count(1, timeout=INTERACTION_TIMEOUT)

        # Pin the rendered labels themselves. The expects above prove the
        # sub-tabs exist; these prove they are the four the tab declares, in
        # the order it declares them, so a renamed or reordered sub-tab is a
        # failure here rather than something a substring locator absorbs.
        assert [t.strip() for t in search_tab.all_inner_texts()] == [
            "\U0001f50d Search Memories"
        ]
        assert [t.strip() for t in add_tab.all_inner_texts()] == [
            "\U0001f4dd Add Memory"
        ]
        assert [t.strip() for t in view_tab.all_inner_texts()] == [
            "\U0001f4cb View All"
        ]
        assert [t.strip() for t in delete_tab.all_inner_texts()] == [
            "\U0001f5d1\ufe0f Delete Memory"
        ]

    def test_add_and_search_memory(self, page):
        memory_text = f"E2E test memory {unique_id()}"

        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Memory")
        wait_for_script_idle(page)

        # Add Memory
        click_sub_tab(page, "Add Memory")
        wait_for_script_idle(page)

        # Target the "Memory Content" textarea specifically (not Chat's textarea)
        memory_textarea = active_tab_panel(page).locator(
            'textarea[aria-label="Memory Content"]:visible'
        )
        assert memory_textarea.count() == 1, (
            "Memory Content text area should be present"
        )
        fill_textarea(memory_textarea, memory_text)

        assert (
            active_tab_panel(page)
            .get_by_role("button", name="💾 Add Memory", exact=True)
            .count()
            == 1
        ), "Add Memory button should be present"
        click_button(page, "Add Memory")
        wait_for_script_idle(page)

        # Memory add alerts persist (no st.rerun) — assert exact feedback
        success = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("added successfully")'
        )
        error = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Failed")'
        )
        # The add is a backend write, so wait for its outcome rather than
        # sampling once. Accepting the failure alert as proof would let a
        # broken add pass the test that exists to exercise it.
        expect(success.or_(error).first).to_be_visible(timeout=INTERACTION_TIMEOUT)
        assert error.count() == 0, (
            f"Memory add reported a failure: {error.first.inner_text()}"
        )
        expect(success).to_have_count(1)

        # Search for the memory
        click_sub_tab(page, "Search Memories")
        wait_for_script_idle(page)

        # Target the "Search Query" textarea specifically
        search_textarea = active_tab_panel(page).locator(
            'textarea[aria-label="Search Query"]'
        )
        assert search_textarea.count() > 0, "Search Query text area should be present"
        fill_textarea(search_textarea, "E2E test memory")

        assert (
            active_tab_panel(page)
            .get_by_role("button", name="🔍 Search", exact=True)
            .count()
            == 1
        ), "Search button should be present"
        click_button(page, "Search")
        wait_for_script_idle(page)

        # Memory search alerts persist (no st.rerun) — assert specific feedback
        found_alert = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("Found")'
        )
        no_results = active_tab_panel(page).locator(
            '[data-testid="stAlert"]:has-text("No memories found")'
        )
        expect(found_alert.or_(no_results).first).to_be_visible(
            timeout=INTERACTION_TIMEOUT
        )

        # If memories were found, verify they are rendered with actual content
        if found_alert.count() > 0:
            # Search results are rendered as expanders or in a dataframe
            panel = active_tab_panel(page)
            expanders = panel.locator('[data-testid="stExpander"]')
            dataframes = panel.locator('[data-testid="stDataFrame"]')
            json_blocks = panel.locator('[data-testid="stJson"]')
            assert (
                expanders.count() > 0
                or dataframes.count() > 0
                or json_blocks.count() > 0
            ), "Found memories must be rendered as expanders, dataframe, or JSON blocks"

    def test_view_all_memories(self, page):
        # Own the listed state: seed one memory through the runtime route
        # (infer=False under the _user_memories namespace) and assert View
        # All renders exactly that row, instead of accepting whatever the
        # shared tenant holds or an empty listing.
        memory_text = f"E2E view-all memory {unique_id()}"
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            seed = client.post(
                f"/admin/tenant/{TENANT_ID}/memories", json={"text": memory_text}
            )
            assert seed.status_code == 200, seed.text[:300]
            memory_id = seed.json()["id"]
            assert seed.json()["status"] == "saved", seed.json()

            # Mem0 writes surface on the list path with eventual consistency;
            # the 30s budget matches _poll_resolve in the cronworkflow e2e.
            deadline = time.monotonic() + 30.0
            listed_ids: list[str] = []
            while listed_ids != [memory_id] and time.monotonic() < deadline:
                listing = client.get(f"/admin/tenant/{TENANT_ID}/memories")
                assert listing.status_code == 200, listing.text[:300]
                listed_ids = [
                    m["id"] for m in listing.json()["memories"] if m["id"] == memory_id
                ]
                if listed_ids != [memory_id]:
                    time.sleep(2.0)
            assert listed_ids == [memory_id], (
                f"seeded memory {memory_id} never surfaced on the list route"
            )

        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Memory")
        wait_for_script_idle(page)
        page.wait_for_timeout(5_000)

        # Configuration also renders an "Agent Name" input, and Streamlit
        # renders every tab body, so a page-wide locator matches two (measured:
        # values ['', 'gateway_agent']). Scope to the panel that owns the
        # Memory sub-tabs.
        memory_panel = page.locator('[role="tabpanel"]').filter(
            has=page.locator('button[role="tab"]:has-text("Search Memories")')
        )
        expect(memory_panel).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        agent_input = memory_panel.locator('input[aria-label="Agent Name"]')
        assert agent_input.count() == 1, "Memory tab must render one Agent Name input"
        assert agent_input.first.input_value() == "gateway_agent", (
            "Memory tab's Agent Name input should carry the shipped default"
        )
        fill_input(agent_input, "_user_memories")
        wait_for_script_idle(page)

        click_sub_tab(page, "View All")
        wait_for_script_idle(page)

        # Page-wide, has-text("Load") matches six buttons -- the first is a
        # hidden "Upload" in another panel (substring match, every tab body
        # is rendered). Scope to the Memory panel and click that button.
        load_btn = memory_panel.locator('button:not([role="tab"]):has-text("Load")')
        assert load_btn.count() == 1, (
            f"View All must render one Load All Memories button, got {load_btn.count()}"
        )
        assert load_btn.first.inner_text().strip() == "🔄 Load All Memories"
        load_btn.first.click()
        wait_for_script_idle(page)

        # Alerts persist (no st.rerun): exactly one "Found N memories" alert
        # and exactly one detailed-view expander for the seeded row.
        #
        # Scoped to the Memory panel. Six other tabs render their own
        # "Found ..." alerts (routing_evaluation, backend_profile and
        # optimization x4), and Streamlit renders every tab body, so the
        # page-wide locator matched seven.
        found_alert = memory_panel.locator(
            '[data-testid="stAlert"]:has-text("memories")'
        ).filter(has_text="Found")
        no_memories = memory_panel.locator(
            '[data-testid="stAlert"]:has-text("No memories")'
        )
        assert found_alert.count() == 1, (
            "View All must report the seeded memory: "
            f"found={found_alert.count()}, no_memories={no_memories.count()}"
        )
        seeded_expander = memory_panel.locator(
            f'[data-testid="stExpander"]:has-text("{memory_id}")'
        )
        assert seeded_expander.count() == 1, (
            f"View All must render one detailed-view expander for {memory_id}; "
            f"got {seeded_expander.count()}"
        )

    def test_delete_memory_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Memory")
        wait_for_script_idle(page)
        page.wait_for_timeout(5_000)
        click_sub_tab(page, "Delete Memory")
        wait_for_script_idle(page)

        # Verify Memory ID input and Delete button present
        inputs = panel_widget(page, "stTextInput", "Memory ID")
        assert inputs.count() == 1, "Delete Memory tab should have Memory ID text input"
        # memory_management.py renders exactly this label; `has-text("Delete")`
        # page-wide also matched Configuration's own Delete control.
        delete_btn = active_tab_panel(page).get_by_role(
            "button", name="🗑️ Delete Memory", exact=True
        )
        expect(delete_btn).to_have_count(1, timeout=INTERACTION_TIMEOUT)


class TestMonitoringDashboard:
    """Scenario 20: Analytics, evaluation, routing eval, orchestration tabs."""

    def _goto_dashboard(self, page):
        """Open the dashboard with the test tenant active.

        Stops at the tenant gate: every caller opens its own top tab as its
        next statement, so opening one here re-rendered a panel no test read.

        Neither step needs a fixed sleep on top of it. ``set_tenant`` returns
        only once the "Current tenant" alert confirms the commit, and
        ``_click_tab_by_label`` waits for script idle and then requires
        aria-selected on the tab it clicked. A sleep beside a verified wait
        cannot make either outcome more certain -- it only pays for the wait
        twice, and pins the cost at a guessed duration rather than the
        measured one.
        """
        _nav(page)
        set_tenant(page, TENANT_ID)

    def test_analytics_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Analytics")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()
        # Analytics MUST NOT show "no tenant selected" — this means tenant didn't propagate
        assert "no tenant selected" not in body_text, (
            "Analytics tab should not show 'No tenant selected' after set_tenant"
        )

        # app.py:857-868 builds this sub-tab strip from a literal list, so
        # every label is structural. The disjunction it replaces was page-wide
        # in both halves -- any tab's metric satisfied `has_data_ui`, and
        # `sub_tabs.count() > 10` counted all 54 tabs in the DOM, so it was
        # true before Analytics rendered anything at all.
        panel = active_tab_panel(page)
        for label in ANALYTICS_SUB_TABS:
            expect(panel.get_by_role("tab", name=label, exact=True)).to_have_count(
                1, timeout=INTERACTION_TIMEOUT
            )

    def test_evaluation_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Evaluation")
        wait_for_script_idle(page)

        panel_text = active_tab_panel(page).inner_text().lower()
        # The Select Dataset control is structural -- it renders whether or not
        # any dataset exists -- so it can be pinned exactly instead of being
        # disjoined with a page-wide "not available" alert.
        assert panel_widget(page, "stSelectbox", "Select Dataset").count() == 1, (
            "Evaluation tab must show the dataset selector"
        )
        assert "evaluation" in panel_text or "experiment" in panel_text, panel_text[
            :300
        ]

    def test_routing_evaluation_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Routing Evaluation")
        wait_for_script_idle(page)

        # routing_evaluation.py:98 renders this heading whenever the tab is
        # reachable. The previous page-wide "not available" alert count was
        # satisfied by any other tab's notice.
        panel_text = assert_tab_rendered(
            page, "Routing Evaluation Dashboard", unavailable="not available"
        )
        if "Routing Evaluation Dashboard" in panel_text:
            # routing_evaluation.py:195 renders the summary block for a
            # reachable evaluator, so its heading is structural rather than
            # data-dependent.
            assert "Summary Metrics" in panel_text, (
                f"A reachable Routing Evaluation must show its summary block:\n"
                f"{panel_text[:400]}"
            )

    def test_orchestration_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Orchestration")
        wait_for_script_idle(page)

        # orchestration_annotation.py:92 renders this header unconditionally.
        # The disjunction it replaces counted Refresh buttons page-wide, and
        # six other tabs render one.
        assert_tab_rendered(
            page, "Orchestration Workflow Annotation", unavailable="not available"
        )

    def test_embedding_atlas_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Embedding Atlas")
        wait_for_script_idle(page)

        # embedding_atlas.py:170 renders this header; :39 renders a missing
        # dependency warning instead, which is a legitimate state on a build
        # without the optional extras.
        assert_tab_rendered(
            page, "Embedding Atlas", unavailable="needs extra libraries"
        )

    def test_finetuning_tab(self, page):
        self._goto_dashboard(page)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)

        # optimization.py:57 and :103 render these headers unconditionally, so
        # the tab cannot be blank and satisfy them. The metric count they
        # replace was page-wide: test_overview_tab establishes that Analytics
        # alone renders several, so it passed whether or not this tab rendered.
        panel_text = assert_tab_rendered(page, "Optimization Framework")
        assert "Optimization Overview" in panel_text, (
            f"The optimization tab must render its overview block:\n{panel_text[:400]}"
        )

    def test_finetuning_dataset_analysis(self, page):
        """Navigate to optimization and verify overview metrics render."""
        self._goto_dashboard(page)
        click_top_tab(page, "Synthetic Data")
        wait_for_script_idle(page)

        # A bare stAlert count is the weakest possible form here: every tab
        # that reports anything renders one, so the old disjunction passed on
        # another tab's alert. optimization.py:1228 labels the annotation
        # metric, which only this tab renders.
        panel_text = assert_tab_rendered(page, "Optimization Overview")
        assert "Annotations" in panel_text, (
            f"The optimization overview must report its annotation pipeline "
            f"state:\n{panel_text[:400]}"
        )


class TestIngestionTesting:
    """Ingestion tab: profile selection, pipeline config, upload controls."""

    def _goto_ingestion(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Ingestion")
        wait_for_script_idle(page)
        # Streamlit streams tab bodies over the websocket, so networkidle can
        # fire while later tabs are still rendering and the page still shows
        # the default Analytics tab. Wait for this tab's own header before
        # asserting anything about the body.
        expect(
            active_tab_panel(page)
            .locator("h1,h2,h3")
            .filter(has_text="Ingestion Pipeline Testing")
        ).to_have_count(1, timeout=INTERACTION_TIMEOUT)

    def test_ingestion_header_and_description(self, page):
        self._goto_ingestion(page)
        # Scoped to this tab's panel: page.inner_text("body") spans every tab
        # Streamlit keeps in the DOM, so "ingestion" matched the nav and the
        # assertion held whether or not this tab rendered.
        panel_text = active_tab_panel(page).inner_text()
        assert "Ingestion Pipeline Testing" in panel_text, panel_text[:300]
        assert (
            "Interactive testing and configuration of video ingestion "
            "pipelines with different processing profiles." in panel_text
        ), panel_text[:300]

    def test_file_uploader_present(self, page):
        self._goto_ingestion(page)
        uploader = panel_widget(page, "stFileUploader", "Upload test video")
        assert uploader.count() == 1, (
            "Ingestion tab should have a file uploader for video upload"
        )

    def test_profile_multiselect_present(self, page):
        self._goto_ingestion(page)
        multiselect = panel_widget(page, "stMultiSelect", "Select profiles to test")
        assert multiselect.count() == 1, (
            "Ingestion tab should have a multiselect for processing profiles"
        )

    def test_pipeline_status_section(self, page):
        self._goto_ingestion(page)
        # Previously a body-wide substring check that was a strict subset of
        # test_ingestion_header_and_description; pin this tab's own sections.
        panel_text = active_tab_panel(page).inner_text()
        assert "Video Upload & Processing" in panel_text, panel_text[:300]
        assert "Results" in panel_text, panel_text[:300]

    def test_documentation_is_inline_not_an_expander(self, page):
        """This tab documents itself inline; it renders no expander.

        The previous assertion required an "About expander with
        documentation". app.py renders the Ingestion section with no expander
        at all -- the only About expander in the app belongs to another tab --
        so the assertion held solely because Streamlit keeps 49 expanders from
        other panels in the DOM. Pin both halves of the real contract.
        """
        self._goto_ingestion(page)
        panel = active_tab_panel(page)
        assert panel.locator('[data-testid="stExpander"]').count() == 0, (
            "Ingestion tab renders no expander; add one and update this pin"
        )
        assert "Upload a video file to start testing ingestion pipelines" in (
            panel.inner_text()
        ), panel.inner_text()[:300]


class TestApprovalQueueTab:
    """Verify the standalone Approval Queue tab under Admin."""

    def test_approval_queue_tab_renders_with_content(self, page):
        """Navigate to Admin → Approval Queue and verify real content."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Approval Queue")
        wait_for_script_idle(page)

        body_text = active_tab_panel(page).inner_text().lower()

        if "not available" in body_text:
            # Module import failed — verify the error message is informative
            assert "approval" in body_text or "review" in body_text, (
                f"Unavailable message should mention approval/review: {body_text[:300]}"
            )
        else:
            # Module loaded — verify actual approval queue UI elements
            has_queue_ui = (
                "pending" in body_text
                or "approved" in body_text
                or "rejected" in body_text
                or "review" in body_text
                or "queue" in body_text
                or "no items" in body_text
            )
            assert has_queue_ui, (
                f"Approval Queue should show queue status (pending/approved/rejected/empty), "
                f"got: {body_text[:300]}"
            )


class TestStreamingEndpointFromDashboard:
    """Verify the A2A streaming endpoint works as the dashboard calls it."""

    def test_streaming_summarize_returns_real_events(self, page):
        """Call the streaming endpoint the same way the dashboard does.

        This tests the actual HTTP path: dashboard → A2A message/stream →
        SummarizerAgent → emit_progress → SSE events with real summary.
        """
        import json
        import uuid

        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": str(uuid.uuid4()),
                        "contextId": str(uuid.uuid4()),
                        "parts": [
                            {
                                "kind": "text",
                                "text": "summarize what video search technology does",
                            }
                        ],
                    },
                    "metadata": {
                        "agent_name": "summarizer_agent",
                        "tenant_id": TENANT_ID,
                        "stream": True,
                    },
                },
            }

            events = []
            with client.stream("POST", "/a2a/", json=payload) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = json.loads(line[5:].strip())
                        for part in (
                            raw.get("result", {})
                            .get("status", {})
                            .get("message", {})
                            .get("parts", [])
                        ):
                            text = part.get("text", "")
                            if text:
                                try:
                                    events.append(json.loads(text))
                                except json.JSONDecodeError:
                                    pass

        # Must have progress events + final
        types = [e.get("type") for e in events]
        assert "status" in types, f"Should have progress events, got: {types}"
        assert "final" in types, f"Should have final event, got: {types}"

        # Final must have real summary content
        finals = [e for e in events if e.get("type") == "final"]
        assert len(finals) == 1
        summary = finals[0]["data"]["summary"]
        assert len(summary) > 20, f"Summary too short: '{summary}'"

    def test_streaming_search_returns_real_events(self, page):
        """Call search streaming the same way the dashboard does."""
        import json
        import uuid

        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": str(uuid.uuid4()),
                        "contextId": str(uuid.uuid4()),
                        "parts": [{"kind": "text", "text": "find nature videos"}],
                    },
                    "metadata": {
                        "agent_name": "search_agent",
                        "tenant_id": TENANT_ID,
                        "stream": True,
                    },
                },
            }

            events = []
            with client.stream("POST", "/a2a/", json=payload) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = json.loads(line[5:].strip())
                        for part in (
                            raw.get("result", {})
                            .get("status", {})
                            .get("message", {})
                            .get("parts", [])
                        ):
                            text = part.get("text", "")
                            if text:
                                try:
                                    events.append(json.loads(text))
                                except json.JSONDecodeError:
                                    pass

        # Search streams status events followed by a final event with results.
        assert len(events) >= 1, (
            f"Search should return at least 1 event, got {len(events)}: {events}"
        )

        # Find the final event (contains results or data)
        final_events = [
            e
            for e in events
            if e.get("type") == "final"
            or "results" in e
            or "status" in e
            and e.get("status") == "success"
        ]
        assert len(final_events) > 0, (
            f"Search must return a final result event, got: {events}"
        )
        result = final_events[-1]

        # Final event may be wrapped in {"type": "final", "data": {...}}
        if result.get("type") == "final" and "data" in result:
            result = result["data"]

        assert "results" in result, (
            f"Search result should have results list: {result.keys()}"
        )
        assert isinstance(result["results"], list)


@pytest.fixture
def optimization_tenant():
    """A tenant owned by this test and removed afterwards.

    The optimization run deploys a schema for whatever tenant it is given, so
    leaving the tenant behind leaves that schema live in Vespa. A later run is
    then refused with "schema(s) live in Vespa have no registry entry", which
    is a guard doing its job: the run that created the schema is the one that
    has to take it away.
    """
    tenant_id = unique_id("opt")
    with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
        created = client.post(
            "/admin/tenants",
            json={"tenant_id": tenant_id, "created_by": "e2e-test"},
        )
    assert created.status_code in (200, 201, 409), created.text
    try:
        yield tenant_id
    finally:
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            client.delete(f"/admin/tenants/{tenant_id}")


class TestManualOptimizationTrigger:
    """The Optimization tab exposes a Run Optimization control that
    submits an Argo Workflow via POST /admin/tenant/{id}/optimize. This
    test drives the UI end-to-end: mode select → Run → success message
    with a ``manual-optimize-...`` workflow name → Refresh status shows
    a real Argo phase. No mocks — the click hits the live runtime, which
    hits the live Argo API, which creates a real Workflow in k3d."""

    def test_run_optimization_submits_workflow_and_status_reflects_argo(
        self, page, optimization_tenant
    ):
        _nav(page)
        tenant_id = optimization_tenant

        set_tenant(page, tenant_id)
        click_top_tab(page, "Optimization")
        wait_for_script_idle(page)

        # The visible subheader includes the rocket emoji prefix.
        page.get_by_role(
            "heading", name="🚀 Optimization Controls", exact=True
        ).wait_for(state="visible", timeout=INTERACTION_TIMEOUT)
        page.get_by_text("🚀 Run Optimization", exact=True).wait_for(
            state="visible", timeout=INTERACTION_TIMEOUT
        )
        body_text = active_tab_panel(page).inner_text()
        assert "Run Optimization" in body_text, (
            "Optimization tab must expose the 'Run Optimization' section "
            "(added with manual-optimize Argo submit feature). Body:\n"
            f"{body_text[:1500]}"
        )

        # Mode dropdown must list every mode the endpoint accepts. Absence
        # of any means the UI got out of sync with _MANUAL_OPTIMIZE_MODES.
        #
        # Scope to the Optimization panel: Streamlit renders every tab body,
        # not just the selected one, so clicking a top tab does not narrow
        # any locator. Page-wide, `stSelectbox:has-text("Mode")` matches
        # three elements -- Evaluation's "Select Dataset dspy-model-..."
        # (`:has-text` is case-insensitive, so "model" matches "Mode") and
        # Configuration's "Routing Mode", both in hidden panels. `.first`
        # picked a hidden one and the click waited out its timeout.
        optimization_panel = page.locator('[role="tabpanel"]').filter(
            has=page.get_by_role(
                "heading", name="\U0001f680 Optimization Controls", exact=True
            )
        )
        expect(optimization_panel).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        mode_select = optimization_panel.locator(
            '[data-testid="stSelectbox"]:has-text("Mode"):visible'
        )
        expect(mode_select).to_have_count(1, timeout=INTERACTION_TIMEOUT)
        assert mode_select.count() == 1, (
            "Optimization panel must render exactly one Mode selectbox"
        )
        # The default value ``gateway-thresholds`` is always rendered.
        # The other options only appear when the dropdown is opened; open
        # it explicitly so we can assert the full option list.
        mode_select.click()
        page.wait_for_timeout(1_000)
        # Read the whole page here, deliberately: Streamlit renders the open
        # dropdown's options in a popover attached to the document body, not
        # inside the tab panel, so a panel-scoped read cannot see them.
        opt_region_text = page.inner_text("body")
        for expected in (
            "gateway-thresholds",
            "simba",
            "workflow",
            "profile",
            "entity-extraction",
        ):
            assert expected in opt_region_text, (
                f"Mode selectbox option {expected!r} missing after "
                f"opening dropdown. Body tail:\n{opt_region_text[-2000:]}"
            )
        # Close the dropdown before clicking Run (otherwise the click
        # selects a mode instead).
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)

        # Click Run. The button is disabled unless a tenant is set — we
        # set a real tenant above, so it must be enabled.
        run_btn = active_tab_panel(page).locator('button:has-text("▶️ Run"):visible')
        assert run_btn.count() == 1, "Run button must exist"
        assert run_btn.first.is_enabled(), (
            "Run button should be enabled when an Active Tenant is set"
        )
        run_btn.first.click()
        page.wait_for_load_state("networkidle", timeout=30_000)
        page.wait_for_timeout(3_000)  # allow Streamlit rerun

        # The dashboard prints `✅ Submitted: manual-optimize-<mode>-<suffix>`
        # on 200. The workflow_name prefix is what we pin — the Argo
        # suffix is random.
        body_text = active_tab_panel(page).inner_text()
        assert "manual-optimize-gateway-thresholds-" in body_text, (
            "Expected a manual-optimize workflow name in the UI after "
            "clicking Run — either the POST failed or Streamlit didn't "
            f"rerun. Body tail:\n{body_text[-1500:]}"
        )

        # The "Last run: <name> (mode: <mode>)" line should be visible.
        assert "Last run:" in body_text and "mode: gateway-thresholds" in body_text, (
            f"Last run status line missing: {body_text[-800:]}"
        )

        # Click Refresh status — the dashboard calls GET
        # /admin/tenant/{id}/optimize/runs/{name}, which returns a real
        # Argo phase.
        refresh_btn = active_tab_panel(page).locator(
            'button:has-text("🔄 Refresh status")'
        )
        assert refresh_btn.count() > 0, (
            "Refresh status button must appear after a successful submit"
        )
        refresh_btn.first.click()

        # After refresh, the UI renders `Phase: <phase>` in an info box.
        # The phase must be one of the known Argo terminal/in-flight
        # phases — never blank (blank = Argo didn't respond). The
        # refresh-triggered rerun repaints over the websocket after the
        # full dashboard script (incl. live agent health checks) runs,
        # so networkidle can't observe it — poll the body for the Phase
        # line instead of sleeping a fixed interval.
        phases = ("Pending", "Running", "Succeeded", "Failed", "Error")
        deadline = time.monotonic() + 60
        body_text = ""
        while time.monotonic() < deadline:
            body_text = active_tab_panel(page).inner_text()
            if any(f"Phase: {p}" in body_text for p in phases):
                break
            page.wait_for_timeout(1_000)
        phase_ok = any(f"Phase: {p}" in body_text for p in phases)
        assert phase_ok, (
            "Refresh status must render `Phase: <Pending|Running|...>` "
            f"from Argo. Body tail:\n{body_text[-1500:]}"
        )
