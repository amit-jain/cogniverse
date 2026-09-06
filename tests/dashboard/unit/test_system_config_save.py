"""Round-trip test for the dashboard System Config save path.

The Save handler rebuilt SystemConfig() from only the ~11 form fields, and
set_system_config persists to_dict() wholesale, so every omitted field
(inference_service_urls, redis_url, minio_endpoint, agent_registry_url, ...) was
reset to its dataclass default — silently breaking ingestion / messaging /
inference for the whole deployment. This exercises the real save function
through a real ConfigManager + ConfigStore round-trip (set -> get).
"""

from __future__ import annotations

from cogniverse_dashboard.tabs.config_management import save_system_config_edits
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from tests.utils.memory_store import InMemoryConfigStore


def test_save_preserves_unedited_infra_fields():
    manager = ConfigManager(store=InMemoryConfigStore())

    # Seed a populated global config with non-default infra fields the form
    # never exposes.
    manager.set_system_config(
        SystemConfig(
            backend_url="http://vespa.svc",
            backend_port=8080,
            redis_url="redis://redis.svc:6379/0",
            minio_endpoint="minio.svc:9000",
            agent_registry_url="http://runtime.svc:8000",
            inference_service_urls={"denseon": "http://denseon.svc:8002"},
            llm_model="old-model",
            environment="staging",
        )
    )

    current = manager.get_system_config()
    assert current.redis_url == "redis://redis.svc:6379/0"  # sanity on seed

    # Apply the exact edits the form would submit (only its fields).
    save_system_config_edits(
        manager,
        current,
        summarizer_agent_url="http://summ.svc",
        search_backend="vespa",
        backend_url="http://vespa.svc",
        backend_port=8080,
        llm_model="new-model",
        base_url="http://llm.svc/v1",
        llm_api_key=None,
        telemetry_url="http://phoenix.svc:6006",
        telemetry_collector_endpoint="http://phoenix.svc:4317",
        environment="production",
    )

    reloaded = manager.get_system_config()

    # Edited fields took effect.
    assert reloaded.llm_model == "new-model"
    assert reloaded.environment == "production"
    assert reloaded.summarizer_agent_url == "http://summ.svc"

    # Unedited infra fields survived (these were wiped by the old code).
    assert reloaded.redis_url == "redis://redis.svc:6379/0"
    assert reloaded.minio_endpoint == "minio.svc:9000"
    assert reloaded.agent_registry_url == "http://runtime.svc:8000"
    assert reloaded.inference_service_urls == {"denseon": "http://denseon.svc:8002"}


def test_application_name_survives_config_store_round_trip():
    """application_name is the Vespa application-package name. to_dict wrote it
    but from_dict dropped it, so a custom value silently reverted to
    'cogniverse' on the next cold load (a fresh process/replica)."""
    store = InMemoryConfigStore()
    ConfigManager(store=store).set_system_config(
        SystemConfig(application_name="acme-cogniverse")
    )

    # A fresh manager reads cold from the store via from_dict (the writing
    # manager's instance cache would otherwise mask the serialization gap).
    reloaded = ConfigManager(store=store).get_system_config()
    assert reloaded.application_name == "acme-cogniverse"


def test_application_name_defaults_when_absent():
    assert SystemConfig.from_dict({}).application_name == "cogniverse"
