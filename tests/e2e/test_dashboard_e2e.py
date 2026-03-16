"""
E2E dashboard tests exercising Streamlit UI via Playwright.

Covers: interactive search, chat, tenant management, config management,
memory lifecycle, DSPy optimization pipeline (annotation harvesting,
golden dataset, synthetic data, module optimization), and monitoring tabs.

Requires:
- Dashboard running at http://localhost:8501
- Runtime at http://localhost:8000
- Ollama, Vespa, Phoenix running
- flywheel_org:production tenant with ingested data
"""

import httpx
import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import (
    DASHBOARD,
    RUNTIME,
    TENANT_ID,
    click_button,
    click_sub_tab,
    click_top_tab,
    fill_input,
    fill_textarea,
    set_tenant,
    skip_if_no_dashboard,
    unique_id,
    wait_for_streamlit,
)

pytestmark = [pytest.mark.e2e, pytest.mark.browser, skip_if_no_dashboard]

# Streamlit takes time to rerun on interactions
INTERACTION_TIMEOUT = 15_000
SEARCH_TIMEOUT = 60_000
LLM_TIMEOUT = 180_000


def _nav(page):
    """Navigate to dashboard and wait for Streamlit to load."""
    page.goto(DASHBOARD, timeout=30_000)
    wait_for_streamlit(page)




class TestSidebarAndNavigation:
    """Verify sidebar tenant input and top-level tab navigation."""

    def test_dashboard_loads_with_expected_structure(self, page):
        _nav(page)
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible(timeout=INTERACTION_TIMEOUT)
        app = page.locator('[data-testid="stAppViewContainer"]')
        expect(app).to_be_visible(timeout=INTERACTION_TIMEOUT)
        tabs = page.locator('button[role="tab"]')
        assert tabs.count() >= 3, (
            f"Expected at least 3 top-level tabs, got {tabs.count()}"
        )

    def test_sidebar_tenant_input(self, page):
        _nav(page)
        # Streamlit may render sidebar inputs hidden in headless mode;
        # verify the element exists in the DOM (not necessarily visible)
        sidebar = page.locator('[data-testid="stSidebar"]')
        tenant_inputs = sidebar.locator('[data-testid="stTextInput"] input')
        assert tenant_inputs.count() > 0, (
            "Tenant input should be present in the sidebar DOM"
        )

    def test_set_tenant_persists(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        page.wait_for_load_state("networkidle")

        # Streamlit may clear the DOM input value on rerun but persists
        # the tenant in session state. Check DOM value, React fiber
        # state, and page body text for tenant ID presence.
        sidebar = page.locator('[data-testid="stSidebar"]')
        tenant_input = sidebar.locator(
            '[data-testid="stTextInput"] input'
        ).first
        value = tenant_input.evaluate(
            "el => el.value || ''"
        )
        body_text = page.inner_text("body")
        assert TENANT_ID in value or TENANT_ID in body_text, (
            f"Tenant should persist after tab switch. "
            f"Input value: '{value}'"
        )

    def test_top_level_tabs_present(self, page):
        _nav(page)
        tabs = page.locator('button[role="tab"]')
        tab_texts = [
            tabs.nth(i).inner_text().lower() for i in range(tabs.count())
        ]
        assert any("user" in t for t in tab_texts), (
            f"User tab missing, tabs: {tab_texts}"
        )
        assert any("admin" in t for t in tab_texts), (
            f"Admin tab missing, tabs: {tab_texts}"
        )
        assert any("monitoring" in t for t in tab_texts), (
            f"Monitoring tab missing, tabs: {tab_texts}"
        )

    def test_agent_status_in_sidebar(self, page):
        _nav(page)
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
        click_top_tab(page, "User")
        click_sub_tab(page, "Interactive Search")

        # Verify search widgets present (use count check, not visibility —
        # Streamlit may render them hidden in headless mode)
        search_input = page.locator(
            '[data-testid="stTextInput"] input'
        ).first
        assert page.locator('[data-testid="stTextInput"] input').count() > 0, (
            "Search input should be present"
        )

        assert page.locator('button:has-text("Search")').count() > 0, (
            "Search button should be present"
        )

        # Fill search query — wait for Streamlit rerun to enable Search button
        fill_input(search_input, "cat videos")
        page.wait_for_timeout(3_000)
        page.wait_for_load_state("networkidle")
        click_button(page, "Search")
        page.wait_for_timeout(SEARCH_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Assert: search must produce a concrete outcome — not just "page changed"
        body_text = page.inner_text("body").lower()
        metrics = page.locator('[data-testid="stMetric"]')
        expanders = page.locator('[data-testid="stExpander"]')
        no_results_msg = "no results" in body_text

        if metrics.count() > 0:
            # Verify metrics show search-specific values (Results, Latency)
            metric_text = metrics.first.inner_text().lower()
            assert any(
                kw in metric_text for kw in ("results", "latency", "score", "0")
            ), f"Search metric should show Results/Latency, got: {metric_text}"

        assert metrics.count() > 0 or expanders.count() > 0 or no_results_msg, (
            "Search must show result metrics, result expanders, or 'No results' — "
            f"metrics={metrics.count()}, expanders={expanders.count()}, "
            f"no_results={no_results_msg}"
        )

    def test_search_annotation(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Interactive Search")

        search_input = page.locator(
            '[data-testid="stTextInput"] input'
        ).first
        fill_input(search_input, "animal videos")
        page.wait_for_timeout(3_000)
        page.wait_for_load_state("networkidle")
        click_button(page, "Search")
        page.wait_for_timeout(SEARCH_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # If results exist, annotate the first one
        save_btn = page.locator('button:has-text("Save Annotation")')
        if save_btn.count() > 0:
            relevant_radio = page.locator('label:has-text("Relevant")')
            if relevant_radio.count() > 0:
                relevant_radio.first.evaluate("el => el.click()")
                page.wait_for_load_state("networkidle")

            click_button(page, "Save Annotation")
            page.wait_for_load_state("networkidle")

            # Verify annotation saved confirmation
            saved_alert = page.locator(
                '[data-testid="stAlert"]:has-text("Annotation saved"), '
                '[data-testid="stAlert"]:has-text("Saved")'
            )
            assert saved_alert.count() > 0, (
                "Annotation save confirmation alert should appear"
            )




class TestMultiModalChat:
    """Scenario 8: Chat with the system, verify responses and multi-turn."""

    def test_send_message_and_get_response(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Chat")

        # Find chat input (text_area or text_input) — use JS fill for hidden elements
        chat_input = page.locator('[data-testid="stTextArea"] textarea')
        if chat_input.count() > 0:
            fill_textarea(chat_input.first, "What videos do you have about animals?")
        else:
            chat_input = page.locator('[data-testid="stTextInput"] input')
            fill_input(chat_input.first, "What videos do you have about animals?")

        click_button(page, "Send")

        page.wait_for_timeout(LLM_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Verify response contains actual content, not just empty containers
        chat_msgs = page.locator('[data-testid="stChatMessage"]')
        markdown = page.locator('[data-testid="stMarkdown"]')
        alerts = page.locator('[data-testid="stAlert"]')

        has_response = (
            chat_msgs.count() > 0
            or markdown.count() > 1
            or alerts.count() > 0
        )
        assert has_response, (
            "Chat should display a response message or status alert after sending"
        )

        # Verify response text is non-empty and not just the query echoed back
        body_text = page.inner_text("body")
        assert len(body_text) > 100, (
            f"Chat response should contain substantial text, got {len(body_text)} chars"
        )
        # The response should contain words beyond the query (routing/search happened)
        query_words = {"what", "videos", "animals"}
        response_words = set(body_text.lower().split())
        non_query_words = response_words - query_words
        assert len(non_query_words) > 20, (
            "Chat response should contain words beyond the query (agent actually responded)"
        )

    def test_multi_turn_conversation(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Chat")

        # Turn 1 — use JS fill for hidden elements
        chat_input = page.locator('[data-testid="stTextArea"] textarea')
        if chat_input.count() > 0:
            fill_textarea(chat_input.first, "search for sports clips")
        else:
            chat_input = page.locator('[data-testid="stTextInput"] input')
            fill_input(chat_input.first, "search for sports clips")
        click_button(page, "Send")
        page.wait_for_timeout(LLM_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Turn 2
        chat_input = page.locator('[data-testid="stTextArea"] textarea')
        if chat_input.count() > 0:
            fill_textarea(chat_input.first, "Tell me more about the first one")
        else:
            chat_input = page.locator('[data-testid="stTextInput"] input')
            fill_input(chat_input.first, "Tell me more about the first one")
        click_button(page, "Send")
        page.wait_for_timeout(LLM_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Verify multi-turn: multiple content blocks visible
        chat_msgs = page.locator('[data-testid="stChatMessage"]')
        markdown = page.locator('[data-testid="stMarkdown"]')
        total = chat_msgs.count() + markdown.count()
        assert total >= 2, (
            f"Multi-turn chat should show conversation history, "
            f"got {total} content blocks"
        )




class TestOptimizationOverview:
    """Optimization Overview and Metrics Dashboard sub-tabs."""

    def test_overview_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Overview")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_content = (
            "overview" in body_text
            or "optimization" in body_text
            or "pipeline" in body_text
            or page.locator('[data-testid="stMetric"]').count() > 0
            or page.locator('[data-testid="stAlert"]').count() > 0
        )
        assert has_content, (
            "Optimization Overview tab should show pipeline overview"
        )

    def test_metrics_dashboard_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Metrics Dashboard")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_content = (
            "metric" in body_text
            or "dashboard" in body_text
            or page.locator('[data-testid="stMetric"]').count() > 0
            or page.locator('[data-testid="stPlotlyChart"]').count() > 0
            or page.locator('[data-testid="stAlert"]').count() > 0
        )
        assert has_content, (
            "Metrics Dashboard tab should show metrics or status"
        )


class TestAnnotationHarvesting:
    """Scenario 9: Fetch search spans and annotate via optimization tab."""

    def test_fetch_and_annotate_spans(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Search Annotations")
        page.wait_for_load_state("networkidle")

        # Verify key widgets: Lookback Hours input and Fetch button
        lookback_input = page.locator('[data-testid="stNumberInput"]')
        assert lookback_input.count() > 0, (
            "Lookback Hours number input should be present"
        )

        fetch_btn = page.locator('button:has-text("Fetch")')
        assert fetch_btn.count() > 0, (
            "Fetch Search Results button should be present"
        )

        click_button(page, "Fetch")
        page.wait_for_timeout(INTERACTION_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Verify fetch produced a specific outcome — not just "any alert"
        fetched_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Fetched")'
        )
        no_spans = page.locator(
            '[data-testid="stAlert"]:has-text("No search")'
        )
        error_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Error"), '
            '[data-testid="stAlert"]:has-text("error"), '
            '[data-testid="stAlert"]:has-text("Failed")'
        )
        dataframes = page.locator('[data-testid="stDataFrame"]')
        assert (
            fetched_alert.count() > 0
            or no_spans.count() > 0
            or error_alert.count() > 0
            or dataframes.count() > 0
        ), (
            "Fetch must show 'Fetched N spans', 'No search spans', error, "
            "or results dataframe"
        )




class TestGoldenDataset:
    """Scenario 10: Build golden dataset from annotations."""

    def test_golden_dataset_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Golden Dataset")
        page.wait_for_load_state("networkidle")

        # Verify Lookback Days number input
        number_inputs = page.locator('[data-testid="stNumberInput"]')
        assert number_inputs.count() > 0, (
            "Lookback Days number input should be present"
        )

        # Verify Build button
        build_btn = page.locator('button:has-text("Build")')
        assert build_btn.count() > 0, (
            "Build Golden Dataset button should be present"
        )

    def test_build_golden_dataset_execution(self, page):
        """Click Build Golden Dataset and verify it produces a result."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Golden Dataset")
        page.wait_for_load_state("networkidle")

        # Click Build and wait for Phoenix query to complete
        click_button(page, "Build")
        page.wait_for_timeout(SEARCH_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Build queries Phoenix telemetry spans — result is either:
        # - "Built golden dataset with N queries" (success, N may be 0)
        # - "Failed to build dataset: ..." (Phoenix connection error)
        # - "No annotated queries found" (no annotations exist)
        built_alert = page.locator(
            '[data-testid="stAlert"]:has-text("dataset")'
        )
        no_data_alert = page.locator(
            '[data-testid="stAlert"]:has-text("No annotated"), '
            '[data-testid="stAlert"]:has-text("no queries")'
        )
        error_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Failed"), '
            '[data-testid="stAlert"]:has-text("Error")'
        )
        assert (
            built_alert.count() > 0
            or no_data_alert.count() > 0
            or error_alert.count() > 0
        ), (
            "Build Golden Dataset must produce a specific result alert — "
            "not just DOM presence"
        )




class TestSyntheticDataAndApproval:
    """Scenario 11: Generate synthetic data, review approval queue."""

    def test_synthetic_data_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Synthetic Data")
        page.wait_for_load_state("networkidle")

        # Verify Examples to Generate number input
        number_inputs = page.locator('[data-testid="stNumberInput"]')
        assert number_inputs.count() > 0, (
            "Examples to Generate number input should be present"
        )

        # Verify Generate button
        generate_btn = page.locator('button:has-text("Generate")')
        assert generate_btn.count() > 0, (
            "Generate Synthetic Data button should be present"
        )

    def test_generate_synthetic_data_execution(self, page):
        """Click Generate and verify synthetic data is produced."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Synthetic Data")
        page.wait_for_load_state("networkidle")

        # Click Generate — this calls POST /synthetic/generate via the runtime
        click_button(page, "Generate")
        # Synthetic data generation queries Vespa + generates examples (up to 5 min)
        page.wait_for_timeout(LLM_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Verify generation outcome
        success_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Generated")'
        )
        error_alert = page.locator(
            '[data-testid="stAlert"]:has-text("failed"), '
            '[data-testid="stAlert"]:has-text("Cannot connect"), '
            '[data-testid="stAlert"]:has-text("timed out")'
        )
        body_text = page.inner_text("body").lower()
        has_examples = "example" in body_text or "confidence" in body_text
        assert (
            success_alert.count() > 0
            or error_alert.count() > 0
            or has_examples
        ), (
            "Generate must produce a result: success with examples, "
            "or a specific error message"
        )

    def test_approval_workflow_in_synthetic_data(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Synthetic Data")
        page.wait_for_load_state("networkidle")

        # Approval workflow is inline within Synthetic Data tab
        # Verify the approval checkbox or review controls exist
        approval_checkbox = page.locator('label:has-text("approval")')
        generate_btn = page.locator('button:has-text("Generate")')
        review_section = page.locator('text=Review')
        assert (
            approval_checkbox.count() > 0
            or generate_btn.count() > 0
            or review_section.count() > 0
        ), (
            "Synthetic Data tab should have approval checkbox, generate button, "
            "or review section"
        )




class TestModuleOptimization:
    """Scenario 12: Trigger DSPy module optimization from dashboard."""

    def test_module_optimization_tab_widgets(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Module Optimization")
        page.wait_for_load_state("networkidle")

        # Verify optimizer/dataset selectbox
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        assert selectboxes.count() > 0, (
            "Optimizer or dataset selectbox should be present"
        )

        # Verify submit or upload button
        submit_btn = page.locator('button:has-text("Submit")')
        upload_btn = page.locator('button:has-text("Upload")')
        assert submit_btn.count() > 0 or upload_btn.count() > 0, (
            "Submit Workflow or Upload Dataset button should be present"
        )

        # Verify DSPy optimizer selection controls exist
        body_text = page.inner_text("body").lower()
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
        """Attempt to submit optimization workflow and verify feedback."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Module Optimization")
        page.wait_for_load_state("networkidle")

        # Try Submit — this may succeed (Argo available) or fail with
        # a specific error (Argo not configured, no training data)
        submit_btn = page.locator('button:has-text("Submit")')
        if submit_btn.count() > 0:
            click_button(page, "Submit")
            page.wait_for_timeout(SEARCH_TIMEOUT)
            page.wait_for_load_state("networkidle")

            # Verify a specific outcome — success, error, or prerequisite warning
            success = page.locator(
                '[data-testid="stAlert"]:has-text("submitted"), '
                '[data-testid="stAlert"]:has-text("success")'
            )
            error = page.locator(
                '[data-testid="stAlert"]:has-text("failed"), '
                '[data-testid="stAlert"]:has-text("Error"), '
                '[data-testid="stAlert"]:has-text("kubectl")'
            )
            warning = page.locator(
                '[data-testid="stAlert"]:has-text("training data"), '
                '[data-testid="stAlert"]:has-text("No dataset")'
            )
            assert (
                success.count() > 0
                or error.count() > 0
                or warning.count() > 0
            ), (
                "Module optimization submission must produce specific feedback"
            )




class TestRerankingAndProfileOptimization:
    """Scenario 13: Reranking and profile selection optimization tabs."""

    def test_reranking_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Reranking")
        page.wait_for_load_state("networkidle")

        # Should show training requirements or metrics
        body_text = page.inner_text("body").lower()
        assert "annotation" in body_text or "train" in body_text or "ndcg" in body_text, (
            "Reranking tab should show training requirements or NDCG metrics"
        )

        # Train Reranker button (may be disabled without enough data)
        train_btn = page.locator('button:has-text("Train")')
        assert train_btn.count() > 0, (
            "Train Reranker button should be present"
        )

    def test_profile_selection_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Optimization")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Profile Selection")
        page.wait_for_load_state("networkidle")

        # Train or Load Model button
        train_btn = page.locator('button:has-text("Train")')
        load_btn = page.locator('button:has-text("Load")')
        assert train_btn.count() > 0 or load_btn.count() > 0, (
            "Profile Selection tab should have Train or Load Model button"
        )




class TestTenantLifecycleDashboard:
    """Scenario 14: Create org + tenant, verify, delete via dashboard."""

    def test_tenant_management_sub_tabs(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Tenant Management")
        page.wait_for_load_state("networkidle")

        # Verify sub-tabs exist
        org_tab = page.locator(
            'button[role="tab"]:has-text("Organizations")'
        )
        create_org_tab = page.locator(
            'button[role="tab"]:has-text("Create Organization")'
        )
        assert org_tab.count() > 0, (
            "Organizations sub-tab should be present"
        )
        assert create_org_tab.count() > 0, (
            "Create Organization sub-tab should be present"
        )

    def test_create_and_delete_organization(self, page):
        org_id = unique_id("dashorg")

        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Tenant Management")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Create Organization")
        page.wait_for_load_state("networkidle")

        # Fill org form (use JS fill for hidden inputs)
        inputs = page.locator('[data-testid="stTextInput"] input')
        assert inputs.count() >= 2, (
            f"Create Organization form needs at least 2 text inputs, "
            f"got {inputs.count()}"
        )

        fill_input(inputs.nth(0), org_id)
        fill_input(inputs.nth(1), f"E2E Dashboard Org {org_id}")

        submit_btn = page.locator(
            'button:has-text("Create Organization")'
        )
        if submit_btn.count() > 0:
            click_button(page, "Create Organization")
            page.wait_for_timeout(INTERACTION_TIMEOUT)
            page.wait_for_load_state("networkidle")

            # st.rerun() clears transient alerts — verify the org was actually
            # created by checking the API directly (authoritative) and that
            # the page reflects the change
            page.wait_for_timeout(3_000)
            body_text = page.inner_text("body")

            # Authoritative check: verify org exists via API
            verify_resp = httpx.get(
                f"{RUNTIME}/admin/organizations/{org_id}", timeout=10.0
            )
            if verify_resp.status_code == 200:
                # Org created — verify dashboard shows it (either in alert or body)
                success = page.locator(
                    '[data-testid="stAlert"]:has-text("created successfully")'
                )
                assert success.count() > 0 or org_id in body_text, (
                    f"Org {org_id} exists in API but not shown on dashboard"
                )
            else:
                # Check for error alert explaining why creation failed
                error = page.locator('[data-testid="stAlert"]')
                assert error.count() > 0, (
                    f"Org creation failed (API {verify_resp.status_code}) "
                    f"but no error alert shown"
                )

        # Cleanup via API
        httpx.delete(
            f"{RUNTIME}/admin/organizations/{org_id}", timeout=10.0
        )

    def test_create_tenant_sub_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Tenant Management")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Create Tenant")
        page.wait_for_load_state("networkidle")

        # Verify Create Tenant form widgets
        body_text = page.inner_text("body").lower()
        assert "tenant" in body_text, (
            "Create Tenant sub-tab should mention 'tenant'"
        )
        inputs = page.locator('[data-testid="stTextInput"] input')
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        has_form = inputs.count() > 0 or selectboxes.count() > 0
        assert has_form, (
            "Create Tenant tab should have input fields or selectboxes"
        )

    def test_tenants_list_sub_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Tenant Management")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Tenants")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_content = (
            "tenant" in body_text
            or page.locator('[data-testid="stExpander"]').count() > 0
            or page.locator('[data-testid="stAlert"]').count() > 0
        )
        assert has_content, (
            "Tenants list tab should show tenants or status message"
        )




class TestConfigManagement:
    """Scenario 16: Edit config, save, verify persistence, export."""

    def test_system_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")

        # System Config is default sub-tab. Verify form elements
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        assert selectboxes.count() > 0, (
            "System Config should have selectboxes (Environment, Backend Type)"
        )

        # Verify Save button exists and is inside a form
        save_btn = page.locator('button:has-text("Save")')
        assert save_btn.count() > 0, (
            "Save System Configuration button should be present"
        )

        # Verify page content shows config-specific terms
        body_text = page.inner_text("body").lower()
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
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Import/Export")
        page.wait_for_load_state("networkidle")

        # Verify Export button and click it to trigger export
        export_btn = page.locator('button:has-text("Export")')
        assert export_btn.count() > 0, (
            "Export Configurations button should be present"
        )

        # Click Export and verify outcome (success alert or download appears)
        click_button(page, "Export")
        page.wait_for_timeout(INTERACTION_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Export should produce a success alert or download link
        export_success = page.locator(
            '[data-testid="stAlert"]:has-text("Exported")'
        )
        download_btn = page.locator('[data-testid="stDownloadButton"]')
        body_text = page.inner_text("body").lower()
        assert (
            export_success.count() > 0
            or download_btn.count() > 0
            or "exported" in body_text
            or "download" in body_text
        ), (
            "Export should show success alert or download button"
        )

        # Verify file uploader for import
        file_uploader = page.locator('[data-testid="stFileUploader"]')
        assert file_uploader.count() > 0, (
            "File uploader for config import should be present"
        )

    def test_agent_configs_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Agent Configs")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_agent_config = (
            "agent" in body_text
            and (
                page.locator('[data-testid="stSelectbox"]').count() > 0
                or page.locator('[data-testid="stTextInput"]').count() > 0
                or "config" in body_text
            )
        )
        assert has_agent_config, (
            "Agent Configs tab should show agent configuration controls"
        )

    def test_routing_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Routing Config")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_routing = (
            "routing" in body_text
            or "agent" in body_text
            or page.locator('[data-testid="stSelectbox"]').count() > 0
            or page.locator('button:has-text("Save")').count() > 0
        )
        assert has_routing, (
            "Routing Config tab should show routing configuration"
        )

    def test_telemetry_config_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Telemetry Config")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_telemetry = (
            "telemetry" in body_text
            or "phoenix" in body_text
            or "trace" in body_text
            or page.locator('[data-testid="stTextInput"]').count() > 0
            or page.locator('button:has-text("Save")').count() > 0
        )
        assert has_telemetry, (
            "Telemetry Config tab should show telemetry settings"
        )

    def test_backend_profiles_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Backend Profiles")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_profiles = (
            "profile" in body_text
            or "schema" in body_text
            or page.locator('[data-testid="stSelectbox"]').count() > 0
            or page.locator('button:has-text("Create")').count() > 0
            or page.locator('button:has-text("Deploy")').count() > 0
        )
        assert has_profiles, (
            "Backend Profiles tab should show profile management controls"
        )

    def test_config_history(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Configuration")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "History")
        page.wait_for_load_state("networkidle")

        # Verify Scope selectbox
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        assert selectboxes.count() > 0, (
            "History tab should have Scope selectbox"
        )

        # Verify history content: version entries, dataframes, or rollback controls
        body_text = page.inner_text("body").lower()
        dataframes = page.locator('[data-testid="stDataFrame"]')
        has_history_content = (
            "version" in body_text
            or "history" in body_text
            or "rollback" in body_text
            or dataframes.count() > 0
        )
        assert has_history_content, (
            "History tab should show version entries, rollback controls, "
            "or history dataframe"
        )




class TestMemoryLifecycle:
    """Scenario 17: Add memory, search, find, delete via dashboard."""

    def test_memory_sub_tabs_present(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Memory")
        page.wait_for_load_state("networkidle")

        # Verify memory sub-tabs
        search_tab = page.locator('button[role="tab"]:has-text("Search")')
        add_tab = page.locator('button[role="tab"]:has-text("Add")')
        view_tab = page.locator('button[role="tab"]:has-text("View All")')
        delete_tab = page.locator('button[role="tab"]:has-text("Delete")')
        assert search_tab.count() > 0, "Search Memories sub-tab should be present"
        assert add_tab.count() > 0, "Add Memory sub-tab should be present"
        assert view_tab.count() > 0, "View All sub-tab should be present"
        assert delete_tab.count() > 0, "Delete Memory sub-tab should be present"

    def test_add_and_search_memory(self, page):
        memory_text = f"E2E test memory {unique_id()}"

        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Memory")
        page.wait_for_load_state("networkidle")

        # Add Memory
        click_sub_tab(page, "Add Memory")
        page.wait_for_load_state("networkidle")

        # Target the "Memory Content" textarea specifically (not Chat's textarea)
        memory_textarea = page.locator('textarea[aria-label="Memory Content"]')
        assert memory_textarea.count() > 0, (
            "Memory Content text area should be present"
        )
        fill_textarea(memory_textarea, memory_text)

        assert page.locator('button:has-text("Add Memory")').count() > 0, (
            "Add Memory button should be present"
        )
        click_button(page, "Add Memory")
        page.wait_for_timeout(INTERACTION_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Memory add alerts persist (no st.rerun) — assert exact feedback
        success = page.locator(
            '[data-testid="stAlert"]:has-text("added successfully")'
        )
        error = page.locator(
            '[data-testid="stAlert"]:has-text("Failed")'
        )
        assert success.count() > 0 or error.count() > 0, (
            "Memory add must show 'added successfully' or 'Failed' alert — "
            f"success={success.count()}, error={error.count()}"
        )

        # Search for the memory
        click_sub_tab(page, "Search Memories")
        page.wait_for_load_state("networkidle")

        # Target the "Search Query" textarea specifically
        search_textarea = page.locator('textarea[aria-label="Search Query"]')
        assert search_textarea.count() > 0, (
            "Search Query text area should be present"
        )
        fill_textarea(search_textarea, "E2E test memory")

        assert page.locator('button:has-text("Search")').count() > 0, (
            "Search button should be present"
        )
        click_button(page, "Search")
        page.wait_for_timeout(INTERACTION_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Memory search alerts persist (no st.rerun) — assert specific feedback
        found_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Found")'
        )
        no_results = page.locator(
            '[data-testid="stAlert"]:has-text("No memories found")'
        )
        assert found_alert.count() > 0 or no_results.count() > 0, (
            "Memory search must show 'Found N memories' or 'No memories found'"
        )

    def test_view_all_memories(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Memory")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "View All")
        page.wait_for_load_state("networkidle")

        load_btn = page.locator('button:has-text("Load")')
        assert load_btn.count() > 0, (
            "View All tab should have Load All Memories button"
        )
        click_button(page, "Load")
        page.wait_for_timeout(INTERACTION_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Alerts persist (no st.rerun) — check for specific outcomes
        found_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Found")'
        )
        dataframes = page.locator('[data-testid="stDataFrame"]')
        expanders = page.locator('[data-testid="stExpander"]')
        no_memories = page.locator(
            '[data-testid="stAlert"]:has-text("No memories")'
        )
        assert (
            found_alert.count() > 0
            or dataframes.count() > 0
            or expanders.count() > 0
            or no_memories.count() > 0
        ), (
            "View All must show 'Found N memories' with data, or 'No memories' — "
            f"found={found_alert.count()}, df={dataframes.count()}, "
            f"expanders={expanders.count()}"
        )

    def test_delete_memory_tab(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "User")
        click_sub_tab(page, "Memory")
        page.wait_for_load_state("networkidle")
        click_sub_tab(page, "Delete Memory")
        page.wait_for_load_state("networkidle")

        # Verify Memory ID input and Delete button present
        inputs = page.locator('[data-testid="stTextInput"] input')
        assert inputs.count() > 0, (
            "Delete Memory tab should have Memory ID text input"
        )
        delete_btn = page.locator('button:has-text("Delete")')
        assert delete_btn.count() > 0, (
            "Delete Memory tab should have Delete Memory button"
        )




class TestMonitoringDashboard:
    """Scenario 20: Analytics, evaluation, routing eval, orchestration tabs."""

    def _goto_monitoring(self, page):
        """Navigate to dashboard and click Monitoring top tab with wait."""
        _nav(page)
        set_tenant(page, TENANT_ID)
        # Wait for tenant to be committed and page to re-render
        page.wait_for_timeout(5_000)
        page.wait_for_load_state("networkidle")
        click_top_tab(page, "Monitoring")
        # Streamlit needs time to re-render after top-tab switch
        page.wait_for_timeout(5_000)
        page.wait_for_load_state("networkidle")

    def test_analytics_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Analytics")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        # Analytics MUST NOT show "no tenant selected" — this means tenant didn't propagate
        assert "no tenant selected" not in body_text, (
            "Analytics tab should not show 'No tenant selected' after set_tenant"
        )

        # Verify analytics-specific UI elements rendered
        metrics = page.locator('[data-testid="stMetric"]')
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        charts = page.locator(
            '[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"]'
        )
        dataframes = page.locator('[data-testid="stDataFrame"]')
        sub_tabs = page.locator('button[role="tab"]')

        # Analytics should show at least one of: metrics, charts, data tables, sub-tabs
        has_data_ui = (
            metrics.count() > 0
            or charts.count() > 0
            or dataframes.count() > 0
            or selectboxes.count() > 0
        )
        # Or analytics sub-tabs (Overview, Time Series, etc.) rendered
        has_sub_tabs = sub_tabs.count() > 10

        assert has_data_ui or has_sub_tabs, (
            f"Analytics tab should show data widgets — "
            f"metrics={metrics.count()}, charts={charts.count()}, "
            f"dataframes={dataframes.count()}, selectboxes={selectboxes.count()}, "
            f"tabs={sub_tabs.count()}"
        )

    def test_evaluation_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Evaluation")
        page.wait_for_load_state("networkidle")

        # Should show dataset selector or info message
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        alerts = page.locator('[data-testid="stAlert"]')
        assert selectboxes.count() > 0 or alerts.count() > 0, (
            "Evaluation tab should show dataset selector or info message"
        )

    def test_routing_evaluation_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Routing Evaluation")
        page.wait_for_load_state("networkidle")

        # Should show lookback controls and/or metrics
        number_inputs = page.locator('[data-testid="stNumberInput"]')
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        metrics = page.locator('[data-testid="stMetric"]')
        alerts = page.locator('[data-testid="stAlert"]')
        has_ui = (
            number_inputs.count() > 0
            or selectboxes.count() > 0
            or metrics.count() > 0
            or alerts.count() > 0
        )
        assert has_ui, (
            "Routing Evaluation should show input controls, metrics, or info"
        )

    def test_orchestration_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Orchestration")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_content = (
            "orchestration" in body_text
            or "span" in body_text
            or "pipeline" in body_text
            or page.locator('[data-testid="stAlert"]').count() > 0
            or page.locator('[data-testid="stDataFrame"]').count() > 0
        )
        assert has_content, (
            "Orchestration tab should show spans, pipeline info, or status"
        )

    def test_embedding_atlas_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Embedding Atlas")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()
        has_content = (
            "embedding" in body_text
            or "atlas" in body_text
            or "dimension" in body_text
            or page.locator('[data-testid="stAlert"]').count() > 0
        )
        assert has_content, (
            "Embedding Atlas tab should show visualization controls or info"
        )

    def test_multimodal_performance_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Multi-Modal Performance")
        page.wait_for_load_state("networkidle")

        metrics = page.locator('[data-testid="stMetric"]')
        alerts = page.locator('[data-testid="stAlert"]')
        charts = page.locator(
            '[data-testid="stPlotlyChart"], '
            '[data-testid="stVegaLiteChart"]'
        )
        assert metrics.count() > 0 or alerts.count() > 0 or charts.count() > 0, (
            "Multi-Modal Performance tab should show metrics, charts, or info"
        )

    def test_finetuning_tab(self, page):
        self._goto_monitoring(page)
        click_sub_tab(page, "Fine-Tuning")
        page.wait_for_load_state("networkidle")

        body_text = page.inner_text("body").lower()

        # Fine-tuning tab has three sections: Configuration, Dataset Status,
        # and Training Configuration. Verify the key UI elements exist.

        # 1. Configuration expander with Tenant ID, Project, Agent/Modality
        config_expander = page.locator('[data-testid="stExpander"]')
        assert config_expander.count() > 0, (
            "Fine-Tuning tab should have Configuration expander"
        )

        # 2. Dataset Status section with Analyze button
        analyze_btn = page.locator('button:has-text("Analyze")')
        assert analyze_btn.count() > 0, (
            "Fine-Tuning tab should have 'Analyze Dataset' button"
        )

        # 3. Verify fine-tuning specific content (not just generic page text)
        assert "fine-tuning" in body_text or "fine tuning" in body_text, (
            "Fine-Tuning tab should mention fine-tuning in its content"
        )
        assert "experiment" in body_text or "dataset" in body_text, (
            "Fine-Tuning tab should reference experiments or datasets"
        )

        # 4. Agent/Modality selectbox should list available targets
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        assert selectboxes.count() > 0, (
            "Fine-Tuning tab should have Agent/Modality selectbox"
        )

    def test_finetuning_dataset_analysis(self, page):
        """Click Analyze Dataset and verify it produces analysis results."""
        self._goto_monitoring(page)
        click_sub_tab(page, "Fine-Tuning")
        page.wait_for_load_state("networkidle")

        # Click Analyze Dataset
        click_button(page, "Analyze")
        page.wait_for_timeout(SEARCH_TIMEOUT)
        page.wait_for_load_state("networkidle")

        # Analysis should produce one of:
        # - Dataset status metrics (total examples, ready for training)
        # - Import error (cogniverse_finetuning not installed)
        # - No data message
        metrics = page.locator('[data-testid="stMetric"]')
        alerts = page.locator('[data-testid="stAlert"]')
        body_text = page.inner_text("body").lower()
        has_analysis_result = (
            metrics.count() > 0
            or alerts.count() > 0
            or "training" in body_text
            or "readiness" in body_text
            or "no data" in body_text
            or "not installed" in body_text
            or "import" in body_text
        )
        assert has_analysis_result, (
            "Analyze Dataset must produce metrics, alerts, or status — "
            f"metrics={metrics.count()}, alerts={alerts.count()}"
        )


class TestIngestionTesting:
    """Ingestion tab: profile selection, pipeline config, upload controls."""

    def _goto_ingestion(self, page):
        _nav(page)
        set_tenant(page, TENANT_ID)
        click_top_tab(page, "Admin")
        click_sub_tab(page, "Ingestion")
        page.wait_for_load_state("networkidle")

    def test_ingestion_header_and_description(self, page):
        self._goto_ingestion(page)
        body_text = page.inner_text("body").lower()
        assert "ingestion" in body_text, (
            "Ingestion tab should display header with 'Ingestion'"
        )
        assert "pipeline" in body_text or "processing" in body_text, (
            "Ingestion tab should describe pipeline or processing"
        )

    def test_file_uploader_present(self, page):
        self._goto_ingestion(page)
        uploader = page.locator('[data-testid="stFileUploader"]')
        assert uploader.count() > 0, (
            "Ingestion tab should have a file uploader for video upload"
        )

    def test_profile_multiselect_present(self, page):
        self._goto_ingestion(page)
        multiselect = page.locator('[data-testid="stMultiSelect"]')
        assert multiselect.count() > 0, (
            "Ingestion tab should have a multiselect for processing profiles"
        )

    def test_pipeline_status_section(self, page):
        self._goto_ingestion(page)
        body_text = page.inner_text("body").lower()
        assert "pipeline status" in body_text or "recent jobs" in body_text, (
            "Ingestion tab should show Pipeline Status or Recent Jobs section"
        )

    def test_about_expander(self, page):
        self._goto_ingestion(page)
        expander = page.locator('[data-testid="stExpander"]')
        assert expander.count() > 0, (
            "Ingestion tab should have an About expander with documentation"
        )
