"""Unit tests for the runtime's semantic-router env-var wiring.

``_semantic_router_config_from_env`` is what makes a DEPLOYED runtime boot
with semantic routing on: the chart sets SEMANTIC_ROUTER_ENABLED +
SEMANTIC_ROUTER_URL (+ optional JSON tier/task maps) and this builds the
``SemanticRouterConfig`` that gets persisted into SystemConfig. These pin the
exact config produced for each env shape, including the graceful-degradation
paths (off when no URL, empty map on malformed JSON).
"""

import pytest

from cogniverse_foundation.config.unified_config import SemanticRouterConfig
from cogniverse_runtime.main import _semantic_router_config_from_env

_SR_ENV = (
    "SEMANTIC_ROUTER_ENABLED",
    "SEMANTIC_ROUTER_URL",
    "SEMANTIC_ROUTER_TENANT_TIERS",
    "SEMANTIC_ROUTER_AGENT_TASKS",
)


@pytest.fixture(autouse=True)
def _clear_sr_env(monkeypatch):
    for name in _SR_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSemanticRouterConfigFromEnv:
    def test_unset_returns_none(self):
        assert _semantic_router_config_from_env() is None

    def test_enabled_without_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "true")
        assert _semantic_router_config_from_env() is None

    def test_url_without_enable_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        assert _semantic_router_config_from_env() is None

    def test_explicit_false_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "false")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        assert _semantic_router_config_from_env() is None

    def test_enabled_with_url_builds_config(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "true")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")

        cfg = _semantic_router_config_from_env()

        assert isinstance(cfg, SemanticRouterConfig)
        assert cfg.enabled is True
        assert cfg.semantic_router_url == "http://cogniverse-gateway:8801/v1"
        assert cfg.tenant_tiers == {}
        assert cfg.agent_tasks == {}

    def test_json_maps_parsed(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "1")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv(
            "SEMANTIC_ROUTER_TENANT_TIERS", '{"acme:prod": "pro", "beta:dev": "free"}'
        )
        monkeypatch.setenv(
            "SEMANTIC_ROUTER_AGENT_TASKS", '{"orchestrator_agent": "orchestrator_plan"}'
        )

        cfg = _semantic_router_config_from_env()

        assert cfg.tenant_tiers == {"acme:prod": "pro", "beta:dev": "free"}
        assert cfg.agent_tasks == {"orchestrator_agent": "orchestrator_plan"}

    def test_malformed_json_map_degrades_to_empty(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "yes")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv("SEMANTIC_ROUTER_TENANT_TIERS", "not-json{")
        monkeypatch.setenv("SEMANTIC_ROUTER_AGENT_TASKS", "[1, 2, 3]")  # not an object

        cfg = _semantic_router_config_from_env()

        # Routing still turns on; the malformed/ill-typed maps degrade to {}.
        assert cfg.enabled is True
        assert cfg.tenant_tiers == {}
        assert cfg.agent_tasks == {}
