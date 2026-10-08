"""The web client's Configuration view, driven in Chromium.

The built client's Node server forwards to the runtime's config routes on a
real uvicorn socket over the real Vespa config store. Each save, restore and
import is read back from the store through a ConfigManager of its own.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import RoutingConfigUnified
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture
def reader(vespa_instance):
    """A manager of its own over the store: what another process reads."""
    return ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        ),
        scoped_config_refresh_s=0,
        scoped_config_max_staleness_s=0,
        system_config_refresh_s=0,
        system_config_max_staleness_s=0,
    )


@pytest.fixture
def tenant():
    return f"webcfg{uuid.uuid4().hex[:8]}:main"


@pytest.fixture
def web_url(built_client, runtime_url):
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


@pytest.fixture
def page(browser):
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture
def system_restored(config_manager):
    before = config_manager.store.get_config(
        "_system", ConfigScope.SYSTEM, "system", "system_config"
    )
    yield
    after = config_manager.store.get_config(
        "_system", ConfigScope.SYSTEM, "system", "system_config"
    )
    config_manager.compare_and_set_entry(
        "_system",
        ConfigScope.SYSTEM,
        "system",
        "system_config",
        before.config_value,
        expected_version=after.version,
    )


def _config_view(page: Page, web_url: str, tenant: str | None = None) -> None:
    page.goto(f"{web_url}/#/ops/config")
    expect(page.get_by_role("heading", name="Configuration", level=1)).to_be_visible()
    expect(page.get_by_role("form", name="Edit System config")).to_be_visible()
    if tenant:
        chooser = page.get_by_role("form", name="Choose tenant")
        chooser.get_by_label("Tenant ID").fill(tenant)
        chooser.get_by_role("button", name="Show configs").click()
        expect(
            page.get_by_role("form", name=f"Edit Routing config of {tenant}")
        ).to_be_visible()


def _routing_form(page: Page, tenant: str):
    return page.get_by_role("form", name=f"Edit Routing config of {tenant}")


class TestSectionForms:
    def test_a_routing_edit_lands_in_the_store_with_every_other_field_kept(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        form = _routing_form(page, tenant)
        expect(form.get_by_text("Not saved yet; showing the defaults.")).to_be_visible()
        default = RoutingConfigUnified(tenant_id=tenant)
        expect(form.get_by_label("routing_mode", exact=True)).to_have_value(
            default.routing_mode
        )

        form.get_by_label("routing_mode", exact=True).fill("direct")
        form.get_by_label("min_unique_queries", exact=True).fill("9")
        form.get_by_label("enable_fast_path", exact=True).uncheck()
        form.get_by_label("optimizer_floors", exact=True).fill(
            json.dumps({"routing": {"min_examples": 4}})
        )
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Routing config of {tenant} as version 1."
        )

        stored = reader.get_routing_config(tenant)
        assert (
            stored.tenant_id,
            stored.routing_mode,
            stored.min_unique_queries,
            stored.enable_fast_path,
            stored.optimizer_floors,
            stored.gliner_model,
            stored.optimization_interval_seconds,
        ) == (
            tenant,
            "direct",
            9,
            False,
            {"routing": {"min_examples": 4}},
            default.gliner_model,
            default.optimization_interval_seconds,
        )
        form = _routing_form(page, tenant)
        expect(form.get_by_text("Version 1, saved ")).to_be_visible()
        expect(form.get_by_label("min_unique_queries", exact=True)).to_have_value("9")

    def test_an_unreadable_input_and_a_refused_value_store_nothing(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        form = _routing_form(page, tenant)
        form.get_by_label("min_unique_queries", exact=True).fill("lots")
        form.get_by_role("button", name="Save").click()
        expect(form.get_by_role("alert")).to_have_text(
            "min_unique_queries must be a whole number."
        )

        form.get_by_label("min_unique_queries", exact=True).fill("3")
        form.get_by_label("optimizer_floors", exact=True).fill('{"routing": 5}')
        form.get_by_role("button", name="Save").click()
        expect(form.get_by_role("alert")).to_have_text(
            "The routing config was not saved.: optimizer_floors.routing: Input "
            "should be a valid dictionary"
        )
        assert (
            reader.store.get_config(
                tenant, ConfigScope.ROUTING, "gateway_agent", "routing_config"
            )
            is None
        )

    def test_an_agent_config_is_created_for_the_named_agent(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        agents = page.get_by_role("region", name=f"Agents of {tenant}")
        expect(
            agents.get_by_text("No agent configs saved for this tenant.")
        ).to_be_visible()
        chooser = agents.get_by_role("form", name="Choose agent")
        chooser.get_by_label("Agent").fill("search_agent")
        chooser.get_by_role("button", name="Edit agent config").click()

        form = page.get_by_role(
            "form", name=f"Edit Agent search_agent config of {tenant}"
        )
        form.get_by_label("agent_description", exact=True).fill("Finds video")
        module = form.get_by_role("group", name="module_config")
        module.get_by_label("module_type", exact=True).select_option("chain_of_thought")
        module.get_by_label("signature", exact=True).fill("question -> answer")
        form.get_by_label("llm_api_key", exact=True).fill("agent-secret")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Agent search_agent config of {tenant} as version 1."
        )

        stored = reader.get_agent_config(tenant, "search_agent")
        assert (
            stored.agent_name,
            stored.agent_description,
            stored.module_config.module_type.value,
            stored.module_config.signature,
            stored.llm_api_key,
        ) == (
            "search_agent",
            "Finds video",
            "chain_of_thought",
            "question -> answer",
            "agent-secret",
        )
        agents = page.get_by_role("region", name=f"Agents of {tenant}")
        expect(agents.get_by_text("Configured: search_agent.")).to_be_visible()

    def test_the_system_api_key_is_written_kept_and_cleared_never_shown(
        self, page, web_url, reader, system_restored
    ):
        base = reader.store.get_config(
            "_system", ConfigScope.SYSTEM, "system", "system_config"
        ).version
        _config_view(page, web_url)
        form = page.get_by_role("form", name="Edit System config")
        form.get_by_label("llm_api_key", exact=True).fill("sk-web-1")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved System config as version {base + 1}."
        )
        assert reader.get_system_config().llm_api_key == "sk-web-1"

        form = page.get_by_role("form", name="Edit System config")
        key = form.get_by_label("llm_api_key", exact=True)
        expect(key).to_have_value("")
        expect(key).to_have_attribute("placeholder", "set; leave blank to keep")
        form.get_by_label("application_name", exact=True).fill("web-edited")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved System config as version {base + 2}."
        )
        stored = reader.get_system_config()
        assert (stored.llm_api_key, stored.application_name) == (
            "sk-web-1",
            "web-edited",
        )
        assert "sk-web-1" not in page.content()

        form = page.get_by_role("form", name="Edit System config")
        form.get_by_label("Clear llm_api_key").check()
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved System config as version {base + 3}."
        )
        assert reader.get_system_config().llm_api_key is None


class TestHistoryAndTransfer:
    def test_an_earlier_version_is_restored_from_the_history(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        for version, mode in enumerate(["direct", "adaptive"], start=1):
            form = _routing_form(page, tenant)
            form.get_by_label("routing_mode", exact=True).fill(mode)
            form.get_by_role("button", name="Save").click()
            expect(page.get_by_role("status")).to_have_text(
                f"Saved Routing config of {tenant} as version {version}."
            )

        stored = page.get_by_role("region", name=f"Stored configs of {tenant}")
        expect(stored.get_by_role("row")).to_have_count(2)
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
        summaries.nth(1).click()
        expect(history.locator("pre").nth(1)).to_contain_text(
            '"routing_mode": "direct"'
        )
        history.get_by_role("button", name="Restore version 1").click()
        expect(page.get_by_role("status")).to_have_text(
            "Restored version 1 of routing/gateway_agent/routing_config as version 3."
        )
        assert reader.get_routing_config(tenant).routing_mode == "direct"
        expect(
            _routing_form(page, tenant).get_by_label("routing_mode", exact=True)
        ).to_have_value("direct")

    def test_an_export_downloads_and_imports_into_another_tenant(
        self, page, web_url, tenant, reader, tmp_path
    ):
        _config_view(page, web_url, tenant)
        form = _routing_form(page, tenant)
        form.get_by_label("routing_mode", exact=True).fill("adaptive")
        form.get_by_label("min_unique_queries", exact=True).fill("11")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Routing config of {tenant} as version 1."
        )

        transfer = page.get_by_role("region", name=f"Export and import for {tenant}")
        with page.expect_download() as download:
            transfer.get_by_role("group", name="Export configs").get_by_role(
                "button", name="Export configs"
            ).click()
        saved = tmp_path / "export.json"
        download.value.save_as(saved)
        assert download.value.suggested_filename == (
            f"config_export_{tenant.replace(':', '_')}.json"
        )
        exported = json.loads(saved.read_text())
        assert [
            (c["scope"], c["service"], c["config_value"]["routing_mode"])
            for c in exported["configs"]
        ] == [("routing", "gateway_agent", "adaptive")]

        target = f"webcfg{uuid.uuid4().hex[:8]}:main"
        _config_view(page, web_url, target)
        transfer = page.get_by_role("region", name=f"Export and import for {target}")
        importer = transfer.get_by_role("form", name="Import configs")
        importer.get_by_label("Export file").set_input_files(str(saved))
        importer.get_by_role("button", name="Import configs").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Imported 1 configs into {target}."
        )
        copied = reader.get_routing_config(target)
        assert (copied.routing_mode, copied.min_unique_queries) == ("adaptive", 11)


class TestConcurrency:
    def test_a_save_over_another_operators_save_is_refused(
        self, browser, web_url, tenant, reader
    ):
        contexts = [browser.new_context() for _ in range(2)]
        pages = [context.new_page() for context in contexts]
        try:
            for page in pages:
                _config_view(page, web_url, tenant)
            first, second = (_routing_form(page, tenant) for page in pages)
            first.get_by_label("routing_mode", exact=True).fill("direct")
            second.get_by_label("routing_mode", exact=True).fill("adaptive")
            first.get_by_role("button", name="Save").click()
            expect(pages[0].get_by_role("status")).to_have_text(
                f"Saved Routing config of {tenant} as version 1."
            )
            second.get_by_role("button", name="Save").click()
            expect(second.get_by_role("alert")).to_have_text(
                "The config changed since it was read (version 0, now 1); reload "
                "it and apply your edits again."
            )
        finally:
            for context in contexts:
                context.close()
        assert reader.get_routing_config(tenant).routing_mode == "direct"


class TestFaultContract:
    def test_a_down_runtime_reads_as_an_error_not_as_defaults(self, page, built_client):
        dead_runtime = f"http://127.0.0.1:{free_port()}"
        unreachable = (
            f"The Cogniverse runtime at {dead_runtime} did not answer (TypeError)."
        )
        with recording_telemetry_sink() as (sink_url, _):
            with serve_web(
                built_client, dead_runtime, KEY, telemetry_url=sink_url, built=True
            ) as url:
                page.goto(f"{url}/#/ops/config")
                expect(
                    page.get_by_role("heading", name="Configuration", level=1)
                ).to_be_visible()
                stored = page.get_by_role("region", name="Stored configs of the system")
                expect(stored.get_by_role("alert")).to_have_text(unreachable)
                expect(stored.get_by_role("table")).to_have_count(0)
                expect(
                    page.get_by_role("region", name="Config store").get_by_role("alert")
                ).to_have_text(unreachable)
                expect(
                    page.get_by_role("form", name="Edit System config")
                ).to_have_count(0)
                expect(
                    page.get_by_text("Not saved yet; showing the defaults.")
                ).to_have_count(0)
