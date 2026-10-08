"""The web client's Configuration view, driven in Chromium.

The built client's Node server forwards to the runtime's config routes on a
real uvicorn socket over the real Vespa config store. Each save, restore and
import is read back from the store through a ConfigManager of its own.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerType,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.sections import (
    ENVIRONMENTS,
    ROUTING_MODES,
    SEARCH_BACKENDS,
)
from cogniverse_foundation.config.unified_config import RoutingConfigUnified
from cogniverse_foundation.telemetry.config import TelemetryConfig, TelemetryLevel
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import register_tenant, serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


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
        register_tenant(tenant)
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

        form.get_by_label("routing_mode", exact=True).select_option("direct")
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

    def test_reload_discards_unsaved_edits(self, page, web_url, tenant):
        _config_view(page, web_url, tenant)
        form = _routing_form(page, tenant)
        default = RoutingConfigUnified(tenant_id=tenant)
        form.get_by_label("min_unique_queries", exact=True).fill("77")
        form.get_by_label("routing_mode", exact=True).select_option("direct")
        page.get_by_role("region", name=f"Routing config of {tenant}").get_by_role(
            "button", name="Reload"
        ).click()
        form = _routing_form(page, tenant)
        expect(form.get_by_label("min_unique_queries", exact=True)).to_have_value(
            str(default.min_unique_queries)
        )
        expect(form.get_by_label("routing_mode", exact=True)).to_have_value(
            default.routing_mode
        )
        expect(form.get_by_text("Not saved yet; showing the defaults.")).to_be_visible()

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
            form.get_by_label("routing_mode", exact=True).select_option(mode)
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
        form.get_by_label("routing_mode", exact=True).select_option("adaptive")
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


def _system_version(reader) -> int:
    return reader.store.get_config(
        "_system", ConfigScope.SYSTEM, "system", "system_config"
    ).version


def _local_time(page: Page, iso: str) -> str:
    return page.evaluate("(iso) => new Date(iso).toLocaleString()", iso)


class TestSystemAndTelemetryForms:
    def test_the_system_form_saves_urls_models_and_a_listed_environment(
        self, page, web_url, reader, system_restored
    ):
        base = _system_version(reader)
        _config_view(page, web_url)
        form = page.get_by_role("form", name="Edit System config")
        expect(
            form.get_by_label("search_backend", exact=True).locator("option")
        ).to_have_text(list(SEARCH_BACKENDS))
        expect(
            form.get_by_label("environment", exact=True).locator("option")
        ).to_have_text(list(ENVIRONMENTS))
        form.get_by_label("summarizer_agent_url", exact=True).fill(
            "http://summarizer.web:8004"
        )
        form.get_by_label("llm_model", exact=True).fill("Qwen/Qwen2.5-7B-Instruct")
        form.get_by_label("base_url", exact=True).fill("http://llm.web:8101/v1")
        form.get_by_label("telemetry_url", exact=True).fill("http://phoenix.web:6006")
        form.get_by_label("telemetry_collector_endpoint", exact=True).fill(
            "phoenix.web:4317"
        )
        form.get_by_label("environment", exact=True).select_option("staging")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved System config as version {base + 1}."
        )
        stored = reader.get_system_config()
        assert (
            stored.summarizer_agent_url,
            stored.llm_model,
            stored.base_url,
            stored.telemetry_url,
            stored.telemetry_collector_endpoint,
            stored.environment,
            stored.search_backend,
        ) == (
            "http://summarizer.web:8004",
            "Qwen/Qwen2.5-7B-Instruct",
            "http://llm.web:8101/v1",
            "http://phoenix.web:6006",
            "phoenix.web:4317",
            "staging",
            "vespa",
        )

    def test_a_backend_port_outside_1_to_65535_is_refused_before_saving(
        self, page, web_url, reader, system_restored
    ):
        base = _system_version(reader)
        _config_view(page, web_url)
        form = page.get_by_role("form", name="Edit System config")
        port = form.get_by_label("backend_port", exact=True)
        expect(port).to_have_attribute("placeholder", "1–65535")
        for refused in ("0", "70000"):
            port.fill(refused)
            form.get_by_role("button", name="Save").click()
            expect(form.get_by_role("alert")).to_have_text(
                "backend_port must be between 1 and 65535."
            )
        assert _system_version(reader) == base

    def test_a_telemetry_edit_lands_with_its_provider_and_endpoints(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        form = page.get_by_role("form", name=f"Edit Telemetry config of {tenant}")
        expect(form.get_by_text("Not saved yet; showing the defaults.")).to_be_visible()
        provider = form.get_by_label("provider", exact=True)
        expect(provider.locator("option")).to_have_text(["(none)", "phoenix"])
        expect(provider).to_have_value("")
        form.get_by_label("enabled", exact=True).uncheck()
        form.get_by_label("level", exact=True).select_option("verbose")
        form.get_by_label("otlp_enabled", exact=True).uncheck()
        form.get_by_label("otlp_endpoint", exact=True).fill("collector.web:4317")
        provider.select_option("phoenix")
        form.get_by_label("provider_config", exact=True).fill(
            json.dumps({"http_endpoint": "http://phoenix.web:6006"})
        )
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Telemetry config of {tenant} as version 1."
        )
        stored = reader.get_telemetry_config(tenant)
        default = TelemetryConfig()
        assert (
            stored.enabled,
            stored.level,
            stored.otlp_enabled,
            stored.otlp_endpoint,
            stored.provider,
            stored.provider_config,
            stored.service_name,
            stored.max_cached_tenants,
        ) == (
            False,
            TelemetryLevel.VERBOSE,
            False,
            "collector.web:4317",
            "phoenix",
            {"http_endpoint": "http://phoenix.web:6006"},
            default.service_name,
            default.max_cached_tenants,
        )
        form = page.get_by_role("form", name=f"Edit Telemetry config of {tenant}")
        expect(form.get_by_label("provider", exact=True)).to_have_value("phoenix")
        expect(form.get_by_label("level", exact=True)).to_have_value("verbose")

    def test_routing_auto_optimisation_settings_land(
        self, page, web_url, tenant, reader
    ):
        _config_view(page, web_url, tenant)
        form = _routing_form(page, tenant)
        expect(
            form.get_by_label("routing_mode", exact=True).locator("option")
        ).to_have_text(list(ROUTING_MODES))
        form.get_by_label("enable_auto_optimization", exact=True).uncheck()
        form.get_by_label("optimization_interval_seconds", exact=True).fill("900")
        form.get_by_label("min_samples_for_optimization", exact=True).fill("50")
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Routing config of {tenant} as version 1."
        )
        stored = reader.get_routing_config(tenant)
        assert (
            stored.enable_auto_optimization,
            stored.optimization_interval_seconds,
            stored.min_samples_for_optimization,
            stored.routing_mode,
        ) == (False, 900, 50, RoutingConfigUnified(tenant_id=tenant).routing_mode)


class TestAgentConfigEdit:
    def test_an_existing_agent_config_gets_module_params_and_an_optimizer(
        self, page, web_url, tenant, reader
    ):
        reader.set_agent_config(
            tenant,
            "summarizer_agent",
            AgentConfig(
                agent_name="summarizer_agent",
                agent_version="2.0.0",
                agent_description="Summarises",
                agent_url="http://summarizer:8004",
                capabilities=["summarize"],
                skills=[],
                module_config=ModuleConfig(
                    module_type=DSPyModuleType.PREDICT, signature="text -> summary"
                ),
            ),
        )
        _config_view(page, web_url, tenant)
        agents = page.get_by_role("region", name=f"Agents of {tenant}")
        expect(agents.get_by_text("Configured: summarizer_agent.")).to_be_visible()
        chooser = agents.get_by_role("form", name="Choose agent")
        chooser.get_by_label("Agent").fill("summarizer_agent")
        chooser.get_by_role("button", name="Edit agent config").click()

        form = page.get_by_role(
            "form", name=f"Edit Agent summarizer_agent config of {tenant}"
        )
        expect(form.get_by_text("Version 1, saved ")).to_be_visible()
        expect(form.get_by_label("agent_description", exact=True)).to_have_value(
            "Summarises"
        )
        module = form.get_by_role("group", name="module_config")
        expect(module.get_by_label("signature", exact=True)).to_have_value(
            "text -> summary"
        )
        module.get_by_label("custom_params", exact=True).fill(
            json.dumps({"max_sentences": 4})
        )
        optimizer = form.get_by_role("group", name="optimizer_config")
        expect(optimizer.get_by_label("optimizer_type", exact=True)).to_have_count(0)
        optimizer.get_by_label("Set optimizer_config").check()
        expect(optimizer.get_by_label("optimizer_type", exact=True)).to_have_value(
            OptimizerType.BOOTSTRAP_FEW_SHOT.value
        )
        expect(optimizer.get_by_label("num_trials", exact=True)).to_have_value("10")
        optimizer.get_by_label("optimizer_type", exact=True).select_option("mipro_v2")
        optimizer.get_by_label("num_trials", exact=True).fill("5")
        optimizer.get_by_label("teacher_settings", exact=True).fill(
            json.dumps({"temperature": 0.2})
        )
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Agent summarizer_agent config of {tenant} as version 2."
        )
        stored = reader.get_agent_config(tenant, "summarizer_agent")
        assert (
            stored.agent_version,
            stored.module_config.signature,
            stored.module_config.custom_params,
            stored.optimizer_config.optimizer_type,
            stored.optimizer_config.num_trials,
            stored.optimizer_config.max_bootstrapped_demos,
            stored.optimizer_config.teacher_settings,
            stored.optimizer_config.custom_params,
        ) == (
            "2.0.0",
            "text -> summary",
            {"max_sentences": 4},
            OptimizerType.MIPRO_V2,
            5,
            4,
            {"temperature": 0.2},
            {},
        )

        # The agent's editor stays open after the save, reading the stored one.
        expect(form.get_by_text("Version 2, saved ")).to_be_visible()
        optimizer = form.get_by_role("group", name="optimizer_config")
        expect(optimizer.get_by_label("optimizer_type", exact=True)).to_have_value(
            "mipro_v2"
        )
        expect(optimizer.get_by_label("teacher_settings", exact=True)).to_have_value(
            json.dumps({"temperature": 0.2}, indent=2)
        )
        optimizer.get_by_label("Set optimizer_config").uncheck()
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Saved Agent summarizer_agent config of {tenant} as version 3."
        )
        assert (
            reader.get_agent_config(tenant, "summarizer_agent").optimizer_config is None
        )
        expect(form.get_by_text("Version 3, saved ")).to_be_visible()
        expect(optimizer.get_by_label("optimizer_type", exact=True)).to_have_count(0)


class TestHistoryScopes:
    def test_the_system_history_reads_the_system_config_and_shows_both_times(
        self, page, web_url, tenant, reader, runtime_url, system_restored
    ):
        reader.set_config_value(
            tenant,
            ConfigScope.SYSTEM,
            "tenant_instructions",
            "system_prompt",
            {"text": "Answer briefly.", "updated_at": "2026-10-08T00:00:00+00:00"},
        )
        current = reader.get_system_config()
        current.application_name = f"history-{uuid.uuid4().hex[:6]}"
        reader.set_system_config(current)
        where = {"scope": "system", "service": "system", "config_key": "system_config"}
        system_history = httpx.get(
            f"{runtime_url}/admin/config/history", params=where, timeout=60
        ).json()

        _config_view(page, web_url, tenant)
        stored = page.get_by_role("region", name="Stored configs of the system")
        stored.get_by_role(
            "button", name="History of system/system/system_config"
        ).click()
        history = stored.get_by_role(
            "region", name="History of system/system/system_config"
        )
        latest = system_history["versions"][0]
        summaries = history.locator("summary")
        expect(summaries).to_have_count(len(system_history["versions"]))
        expect(summaries.nth(0)).to_have_text(
            f"Version {latest['version']}, created "
            f"{_local_time(page, latest['created_at'])}, updated "
            f"{_local_time(page, latest['updated_at'])} (current)"
        )
        summaries.nth(0).click()
        expect(history.locator("pre").nth(0)).to_contain_text(
            f'"application_name": "{current.application_name}"'
        )

        own = page.get_by_role("region", name=f"Stored configs of {tenant}")
        own.get_by_role(
            "button", name="History of system/tenant_instructions/system_prompt"
        ).click()
        instructions = own.get_by_role(
            "region", name="History of system/tenant_instructions/system_prompt"
        )
        expect(instructions.locator("summary")).to_have_count(1)
        instructions.locator("summary").click()
        expect(instructions.locator("pre")).to_contain_text('"text": "Answer briefly."')


class TestImportPreview:
    def test_an_export_of_every_version_is_previewed_then_imported(
        self, page, web_url, tenant, reader, tmp_path
    ):
        _config_view(page, web_url, tenant)
        for version, mode in enumerate(["direct", "adaptive"], start=1):
            form = _routing_form(page, tenant)
            form.get_by_label("routing_mode", exact=True).select_option(mode)
            form.get_by_role("button", name="Save").click()
            expect(page.get_by_role("status")).to_have_text(
                f"Saved Routing config of {tenant} as version {version}."
            )
        transfer = page.get_by_role("region", name=f"Export and import for {tenant}")
        group = transfer.get_by_role("group", name="Export configs")
        group.get_by_label("Include every version").check()
        with page.expect_download() as download:
            group.get_by_role("button", name="Export configs").click()
        saved = tmp_path / "history.json"
        download.value.save_as(saved)
        exported = json.loads(saved.read_text())
        assert exported["include_history"] is True
        assert [
            (c["scope"], c["version"], c["config_value"]["routing_mode"])
            for c in exported["configs"]
        ] == [("routing", 1, "direct"), ("routing", 2, "adaptive")]

        target = f"webcfg{uuid.uuid4().hex[:8]}:main"
        _config_view(page, web_url, target)
        importer = page.get_by_role(
            "region", name=f"Export and import for {target}"
        ).get_by_role("form", name="Import configs")
        expect(importer.get_by_role("button", name="Import configs")).to_be_disabled()
        importer.get_by_label("Export file").set_input_files(str(saved))
        preview = importer.get_by_role("region", name="Import preview")
        expect(preview.locator("p")).to_have_text(
            f"2 configs exported from {tenant}; importing writes each into {target}."
        )
        expect(preview.locator("tbody td")).to_have_text(
            ["routing", "gateway_agent", "routing_config", "1"]
            + ["routing", "gateway_agent", "routing_config", "2"]
        )
        assert reader.store.list_configs(tenant_id=target) == []
        importer.get_by_role("button", name="Import configs").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Imported 2 configs into {target}."
        )
        assert reader.get_routing_config(target).routing_mode == "adaptive"

    def test_a_file_that_is_not_json_is_refused_on_choosing_it(
        self, page, web_url, tenant, reader, tmp_path
    ):
        bad = tmp_path / "bad.json"
        bad.write_text('{"configs": [')
        _config_view(page, web_url, tenant)
        importer = page.get_by_role(
            "region", name=f"Export and import for {tenant}"
        ).get_by_role("form", name="Import configs")
        importer.get_by_label("Export file").set_input_files(str(bad))
        expect(importer.get_by_role("alert")).to_have_text(
            "bad.json is not valid JSON."
        )
        expect(importer.get_by_role("region", name="Import preview")).to_have_count(0)
        expect(importer.get_by_role("button", name="Import configs")).to_be_disabled()
        assert reader.store.list_configs(tenant_id=tenant) == []


class TestStoreStats:
    def test_the_store_panel_shows_the_backend_and_its_counts(
        self, page, web_url, runtime_url
    ):
        stats = httpx.get(f"{runtime_url}/admin/config/stats", timeout=60).json()
        assert stats["storage_backend"] == "vespa"
        _config_view(page, web_url)
        health = page.get_by_role("region", name="Config store").locator(
            'dl[aria-label="Config store health"]'
        )
        expect(health.locator("dd")).to_have_text(
            ["VespaConfigStore", "healthy: it answers queries"]
        )
        assert httpx.get(f"{runtime_url}/admin/config/health", timeout=60).json() == {
            "store": "VespaConfigStore",
            "healthy": True,
        }
        panel = page.get_by_role("region", name="Config store").locator(
            'dl[aria-label="Config store facts"]'
        )
        expect(panel.locator("dd").first).to_have_text("vespa")
        terms = panel.locator("dt").all_inner_texts()
        values = panel.locator("dd").all_inner_texts()
        assert dict(zip(terms, values, strict=True)) == {
            "Backend": "vespa",
            "Configs": str(stats["total_configs"]),
            "Versions": str(stats["total_versions"]),
            "Tenants": str(stats["total_tenants"]),
            "By scope": ", ".join(
                f"{scope}: {count}"
                for scope, count in sorted(stats["configs_per_scope"].items())
            ),
        }


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
            first.get_by_label("routing_mode", exact=True).select_option("direct")
            second.get_by_label("routing_mode", exact=True).select_option("adaptive")
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
                built_client, dead_runtime, telemetry_url=sink_url, built=True
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
                ).to_have_text([unreachable, unreachable])
                expect(
                    page.locator('dl[aria-label="Config store health"]')
                ).to_have_count(0)
                expect(
                    page.get_by_role("form", name="Edit System config")
                ).to_have_count(0)
                expect(
                    page.get_by_text("Not saved yet; showing the defaults.")
                ).to_have_count(0)
