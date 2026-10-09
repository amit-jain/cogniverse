"""The configuration routes over a real Vespa config store.

Every write is read back from the store through a ConfigManager of its own,
not from the route's answer alone.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.agent_config import AgentConfig
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    DurableExecutionConfig,
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_runtime.routers import admin, config_entries
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.web_client import free_port

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(config_manager, config_change_events):
    app = FastAPI()
    app.include_router(config_entries.router, prefix="/admin")
    admin.set_config_manager(config_manager)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def reader(vespa_instance):
    """A second manager over the same store: what another process reads."""
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
    return f"cfg{uuid.uuid4().hex[:8]}:main"


@pytest.fixture
def system_restored(config_manager):
    """Put the module's system config back after a test writes it."""
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


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _routing(client, tenant):
    response = client.get(
        "/admin/config/sections/routing", params={"tenant_id": tenant}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _put(client, section, value, version, **where):
    return client.put(
        f"/admin/config/sections/{section}",
        json={"value": value, "version": version, **where},
    )


class TestSections:
    def test_each_section_is_its_dataclasss_form(self, client):
        response = client.get("/admin/config/sections")
        assert response.status_code == 200
        sections = {s["name"]: s for s in response.json()["sections"]}
        assert list(sections) == [
            "system",
            "routing",
            "telemetry",
            "agent",
            "durable_execution",
        ]
        properties = {
            name: set(s["schema"]["properties"]) for name, s in sections.items()
        }
        assert properties == {
            "system": _fields(SystemConfig),
            "routing": _fields(RoutingConfigUnified) - {"tenant_id"},
            "telemetry": _fields(TelemetryConfig),
            "agent": _fields(AgentConfig),
            "durable_execution": _fields(DurableExecutionConfig) - {"tenant_id"},
        }
        assert [(n, s["tenant_scoped"], s["service"]) for n, s in sections.items()] == [
            ("system", False, "system"),
            ("routing", True, "gateway_agent"),
            ("telemetry", True, "telemetry"),
            ("agent", True, None),
            ("durable_execution", True, "optimization"),
        ]
        assert sections["system"]["schema"]["properties"]["llm_api_key"]["writeOnly"]
        assert sections["agent"]["schema"]["properties"]["llm_api_key"]["writeOnly"]


class TestReadAndWrite:
    def test_an_unset_section_reads_as_its_defaults_at_version_zero(
        self, client, tenant
    ):
        body = _routing(client, tenant)
        default = dataclasses.asdict(RoutingConfigUnified(tenant_id=tenant))
        default.pop("tenant_id")
        assert body == {
            "section": "routing",
            "tenant_id": tenant,
            "service": "gateway_agent",
            "version": 0,
            "updated_at": None,
            "value": default,
            "secrets": {},
        }

    def test_a_save_lands_in_the_store_and_unsent_fields_keep_their_values(
        self, client, tenant, reader, config_manager
    ):
        floors = {"routing": {"min_examples": 7}}
        first = _put(
            client,
            "routing",
            {"routing_mode": "direct", "optimizer_floors": floors},
            0,
            tenant_id=tenant,
        )
        assert first.status_code == 200, first.text
        assert first.json()["version"] == 1
        # The manager serving the route reads the write at once.
        assert config_manager.get_routing_config(tenant).routing_mode == "direct"

        second = _put(
            client, "routing", {"enable_fast_path": False}, 1, tenant_id=tenant
        )
        assert second.status_code == 200, second.text
        stored = reader.get_routing_config(tenant)
        assert (
            stored.tenant_id,
            stored.routing_mode,
            stored.optimizer_floors,
            stored.enable_fast_path,
            stored.min_samples_for_optimization,
        ) == (
            tenant,
            "direct",
            floors,
            False,
            RoutingConfigUnified(tenant_id=tenant).min_samples_for_optimization,
        )
        assert second.json()["value"]["optimizer_floors"] == floors

    def test_a_save_over_a_replaced_version_is_refused_and_writes_nothing(
        self, client, tenant, reader
    ):
        assert (
            _put(
                client, "routing", {"routing_mode": "direct"}, 0, tenant_id=tenant
            ).status_code
            == 200
        )
        stale = _put(
            client, "routing", {"routing_mode": "adaptive"}, 0, tenant_id=tenant
        )
        assert stale.status_code == 409
        assert stale.json()["detail"] == {
            "error": "config_version_conflict",
            "message": "The config changed since it was read (version 0, now 1); "
            "reload it and apply your edits again.",
            "current_version": 1,
        }
        assert reader.get_routing_config(tenant).routing_mode == "direct"

    def test_an_invalid_value_names_each_problem_and_writes_nothing(
        self, client, tenant, reader
    ):
        response = _put(
            client,
            "telemetry",
            {
                "max_cached_tenants": "many",
                "colour": "blue",
                "batch_config": {"max_queue_size": 5, "speed": 1},
            },
            0,
            tenant_id=tenant,
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "error": "config_value_invalid",
            "message": "The telemetry config was not saved.",
            "errors": ["unknown field colour", "unknown field batch_config.speed"],
        }
        response = _put(
            client, "telemetry", {"max_cached_tenants": "many"}, 0, tenant_id=tenant
        )
        assert response.json()["detail"]["errors"] == [
            "max_cached_tenants: Input should be a valid integer, unable to parse "
            "string as an integer"
        ]
        assert (
            reader.store.get_config(
                tenant, ConfigScope.TELEMETRY, "telemetry", "telemetry_config"
            )
            is None
        )

    def test_a_tenant_section_needs_a_tenant_and_a_system_one_refuses_it(
        self, client, tenant
    ):
        assert client.get("/admin/config/sections/routing").json()["detail"] == (
            "tenant_id is required"
        )
        response = client.get(
            "/admin/config/sections/system", params={"tenant_id": tenant}
        )
        assert (response.status_code, response.json()["detail"]) == (
            400,
            "System configs take no tenant_id",
        )
        response = client.get("/admin/config/sections/nope")
        assert (response.status_code, response.json()["detail"]) == (
            404,
            "No config section 'nope'; sections: ['agent', 'durable_execution', "
            "'routing', 'system', 'telemetry']",
        )


class TestStoreHealth:
    def test_a_store_that_answers_reads_as_healthy(self, client):
        response = client.get("/admin/config/health")
        assert (response.status_code, response.json()) == (
            200,
            {"store": "VespaConfigStore", "healthy": True},
        )


class TestChoicesAndRanges:
    def test_fixed_choice_fields_and_the_port_carry_their_values_in_the_schema(
        self, client
    ):
        sections = {
            s["name"]: s["schema"]["properties"]
            for s in client.get("/admin/config/sections").json()["sections"]
        }
        assert (
            sections["system"]["search_backend"]["enum"],
            sections["system"]["environment"]["enum"],
            sections["routing"]["routing_mode"]["enum"],
            sections["telemetry"]["provider"]["anyOf"],
            sections["system"]["backend_port"]["minimum"],
            sections["system"]["backend_port"]["maximum"],
        ) == (
            ["vespa"],
            ["development", "staging", "production"],
            ["tiered", "direct", "adaptive"],
            [{"type": "string", "enum": ["phoenix"]}, {"type": "null"}],
            1,
            65535,
        )

    def test_a_value_outside_the_choices_or_the_range_stores_nothing(
        self, client, tenant, reader
    ):
        refused = _put(
            client,
            "routing",
            {"routing_mode": "fastest"},
            0,
            tenant_id=tenant,
        )
        assert refused.status_code == 422
        assert refused.json()["detail"]["errors"] == [
            "routing_mode: must be one of tiered, direct, adaptive"
        ]
        refused = _put(
            client, "telemetry", {"provider": "langsmith"}, 0, tenant_id=tenant
        )
        assert refused.json()["detail"]["errors"] == [
            "provider: must be one of phoenix"
        ]
        assert reader.store.list_configs(tenant_id=tenant) == []

        system = client.get("/admin/config/sections/system").json()
        refused = _put(
            client,
            "system",
            {"backend_port": 70000, "environment": "qa", "search_backend": "solr"},
            system["version"],
        )
        assert refused.status_code == 422
        assert refused.json()["detail"]["errors"] == [
            "search_backend: must be one of vespa",
            "environment: must be one of development, staging, production",
            "backend_port: must be between 1 and 65535",
        ]
        assert (
            client.get("/admin/config/sections/system").json()["version"]
            == system["version"]
        )

    def test_a_stored_value_outside_the_choices_is_kept_by_a_save(
        self, client, tenant, reader, config_manager
    ):
        legacy = RoutingConfigUnified(tenant_id=tenant, routing_mode="hybrid")
        config_manager.set_routing_config(legacy)
        version = _routing(client, tenant)["version"]
        saved = _put(
            client,
            "routing",
            {"routing_mode": "hybrid", "min_unique_queries": 6},
            version,
            tenant_id=tenant,
        )
        assert saved.status_code == 200, saved.text
        stored = reader.get_routing_config(tenant)
        assert (stored.routing_mode, stored.min_unique_queries) == ("hybrid", 6)


class TestSecrets:
    def test_a_secret_is_written_never_read_back_kept_when_null_and_cleared_by_empty(
        self, client, reader, system_restored
    ):
        current = client.get("/admin/config/sections/system").json()
        written = _put(client, "system", {"llm_api_key": "s3cret"}, current["version"])
        assert written.status_code == 200, written.text
        body = written.json()
        assert (body["value"]["llm_api_key"], body["secrets"]) == (
            None,
            {"llm_api_key": True},
        )
        assert reader.get_system_config().llm_api_key == "s3cret"

        kept = _put(
            client,
            "system",
            {"llm_api_key": None, "application_name": "renamed"},
            body["version"],
        )
        assert kept.status_code == 200, kept.text
        stored = reader.get_system_config()
        assert (stored.llm_api_key, stored.application_name) == ("s3cret", "renamed")

        history = client.get(
            "/admin/config/history",
            params={
                "scope": "system",
                "service": "system",
                "config_key": "system_config",
            },
        ).json()
        assert [v["value"]["llm_api_key"] for v in history["versions"][:2]] == [
            None,
            None,
        ]

        cleared = _put(client, "system", {"llm_api_key": ""}, kept.json()["version"])
        assert cleared.json()["secrets"] == {"llm_api_key": False}
        assert reader.get_system_config().llm_api_key is None


class TestAgents:
    def test_an_agent_config_is_created_and_read_by_its_service(
        self, client, tenant, reader
    ):
        missing = _put(client, "agent", {}, 0, tenant_id=tenant)
        assert (missing.status_code, missing.json()["detail"]) == (
            400,
            "section agent needs a service name",
        )
        created = _put(
            client,
            "agent",
            {
                "agent_description": "Searches video",
                "llm_model": "qwen3:4b",
                "module_config": {
                    "module_type": "chain_of_thought",
                    "signature": "Q -> A",
                },
                "llm_api_key": "agent-key",
            },
            0,
            tenant_id=tenant,
            service="search_agent",
        )
        assert created.status_code == 200, created.text
        stored = reader.get_agent_config(tenant, "search_agent")
        assert (
            stored.agent_name,
            stored.agent_description,
            stored.llm_model,
            stored.module_config.module_type.value,
            stored.module_config.signature,
            stored.llm_api_key,
        ) == (
            "search_agent",
            "Searches video",
            "qwen3:4b",
            "chain_of_thought",
            "Q -> A",
            "agent-key",
        )
        read = client.get(
            "/admin/config/sections/agent",
            params={"tenant_id": tenant, "service": "search_agent"},
        ).json()
        assert (read["version"], read["value"]["llm_api_key"], read["secrets"]) == (
            1,
            None,
            {"llm_api_key": True},
        )


class TestEntriesHistoryRollback:
    def test_entries_history_and_rollback(self, client, tenant, reader):
        for version, mode in enumerate(["direct", "adaptive"]):
            assert (
                _put(
                    client, "routing", {"routing_mode": mode}, version, tenant_id=tenant
                ).status_code
                == 200
            )
        assert (
            _put(
                client, "durable_execution", {"enabled": True}, 0, tenant_id=tenant
            ).status_code
            == 200
        )

        entries = client.get(
            "/admin/config/entries", params={"tenant_id": tenant}
        ).json()
        assert entries["tenant_id"] == tenant
        assert [
            (e["scope"], e["service"], e["config_key"], e["version"], e["section"])
            for e in entries["entries"]
        ] == [
            (
                "durable",
                "optimization",
                "durable_execution_config",
                1,
                "durable_execution",
            ),
            ("routing", "gateway_agent", "routing_config", 2, "routing"),
        ]

        where = {
            "tenant_id": tenant,
            "scope": "routing",
            "service": "gateway_agent",
            "config_key": "routing_config",
        }
        history = client.get("/admin/config/history", params=where).json()
        assert [
            (v["version"], v["value"]["routing_mode"]) for v in history["versions"]
        ] == [
            (2, "adaptive"),
            (1, "direct"),
        ]
        assert history["section"] == "routing"

        stale = client.post(
            "/admin/config/rollback",
            json={**where, "version": 1, "expected_version": 1},
        )
        assert stale.status_code == 400
        restored = client.post(
            "/admin/config/rollback",
            json={**where, "version": 1, "expected_version": 2},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["version"] == 3
        assert reader.get_routing_config(tenant).routing_mode == "direct"

        late = client.post(
            "/admin/config/rollback",
            json={**where, "version": 1, "expected_version": 2},
        )
        assert late.status_code == 409
        assert late.json()["detail"]["current_version"] == 3

        missing = client.get(
            "/admin/config/history", params={**where, "service": "nobody"}
        )
        assert missing.status_code == 404

    def test_the_system_history_is_read_from_the_system_configs(
        self, client, tenant, system_restored
    ):
        """System configs live under the system tenant; a history read with
        no tenant is theirs, never a tenant's system-scope rows."""
        for name in ("history-one", "history-two"):
            current = client.get("/admin/config/sections/system").json()
            saved = _put(
                client, "system", {"application_name": name}, current["version"]
            )
            assert saved.status_code == 200, saved.text
        where = {"scope": "system", "service": "system", "config_key": "system_config"}
        history = client.get("/admin/config/history", params=where).json()
        latest = client.get("/admin/config/sections/system").json()["version"]
        assert history["tenant_id"] == "_system"
        assert [
            (v["version"], v["value"]["application_name"])
            for v in history["versions"][:2]
        ] == [(latest, "history-two"), (latest - 1, "history-one")]
        tenant_history = client.get(
            "/admin/config/history", params={**where, "tenant_id": tenant}
        )
        assert (tenant_history.status_code, tenant_history.json()["detail"]) == (
            404,
            f"No system config system/system_config for {tenant}",
        )

    def test_an_export_imports_whole_into_another_tenant(self, client, tenant, reader):
        assert (
            _put(
                client,
                "routing",
                {"routing_mode": "adaptive", "min_unique_queries": 9},
                0,
                tenant_id=tenant,
            ).status_code
            == 200
        )
        exported = client.get("/admin/config/export", params={"tenant_id": tenant})
        assert exported.status_code == 200
        target = f"cfg{uuid.uuid4().hex[:8]}:main"
        imported = client.post(
            "/admin/config/import",
            json={"tenant_id": target, "configs": exported.json()},
        )
        assert imported.status_code == 200, imported.text
        assert imported.json() == {"tenant_id": target, "imported": 1}
        copied = reader.get_routing_config(target)
        assert (copied.tenant_id, copied.routing_mode, copied.min_unique_queries) == (
            tenant,
            "adaptive",
            9,
        )


class TestConcurrency:
    def test_concurrent_saves_of_one_version_land_exactly_once(
        self, client, tenant, reader
    ):
        modes = ["direct", "adaptive", "tiered", "direct", "adaptive", "tiered"]
        barrier = threading.Barrier(len(modes))

        def save(index: int):
            barrier.wait()
            return index, _put(
                client,
                "routing",
                {"routing_mode": modes[index], "min_unique_queries": 100 + index},
                0,
                tenant_id=tenant,
            )

        with ThreadPoolExecutor(max_workers=len(modes)) as pool:
            results = list(pool.map(save, range(len(modes))))

        codes = sorted(response.status_code for _, response in results)
        assert codes == [200] + [409] * (len(modes) - 1)
        winner = next(
            index for index, response in results if response.status_code == 200
        )
        stored = reader.get_routing_config(tenant)
        assert (stored.routing_mode, stored.min_unique_queries) == (
            modes[winner],
            100 + winner,
        )
        assert (
            reader.store.get_config(
                tenant, ConfigScope.ROUTING, "gateway_agent", "routing_config"
            ).version
            == 1
        )


class TestFaultContract:
    def test_a_down_store_answers_503_not_defaults(self, tenant, config_change_events):
        dead = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://localhost", backend_port=free_port()
            )
        )
        app = FastAPI()
        app.include_router(config_entries.router, prefix="/admin")
        app.dependency_overrides[admin.get_config_manager_dependency] = lambda: dead
        with TestClient(app) as down:
            read = down.get(
                "/admin/config/sections/routing", params={"tenant_id": tenant}
            )
            assert read.status_code == 503
            assert read.json()["detail"] == {
                "error": "config_store_unavailable",
                "message": "The config store did not answer while reading the "
                "routing config; retry.",
                "failure": "ConfigStoreUnavailableError",
                "tenant_id": tenant,
                "service": "gateway_agent",
            }
            write = _put(
                down, "routing", {"routing_mode": "direct"}, 0, tenant_id=tenant
            )
            assert write.status_code == 503
            assert write.json()["detail"]["message"] == (
                "The config store did not answer while saving the routing config; retry."
            )
            listed = down.get("/admin/config/entries", params={"tenant_id": tenant})
            assert (listed.status_code, listed.json()["detail"]["error"]) == (
                503,
                "config_store_unavailable",
            )
            health = down.get("/admin/config/health")
            assert (health.status_code, health.json()) == (
                200,
                {"store": "VespaConfigStore", "healthy": False},
            )
