"""Telemetry providers must be cached per (tenant, project).

A tenant can register distinct endpoints per project; the registry's default
tenant-only cache key made a second project silently reuse the first's cached
provider (and its endpoints). The provider cache key must distinguish projects.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import TelemetryRegistry


def test_cache_key_distinguishes_projects():
    a = TelemetryRegistry._cache_key("phoenix", {"project_name": "search"}, "acme:acme")
    b = TelemetryRegistry._cache_key(
        "phoenix", {"project_name": "routing"}, "acme:acme"
    )
    none = TelemetryRegistry._cache_key("phoenix", {}, "acme:acme")

    assert a != b, "two projects of one tenant must not share a cached provider"
    assert a != none
    # No project → tenant-only key (unchanged default behaviour).
    assert none == "phoenix_acme:acme"


def test_get_provider_threads_project_name_into_config():
    manager = object.__new__(TelemetryManager)
    manager.config = SimpleNamespace(
        otlp_endpoint="localhost:4317",
        otlp_use_tls=False,
        provider_config={},
        provider=None,
    )
    manager._project_configs = {}

    captured = {}
    fake_registry = MagicMock()

    def _get(name, tenant_id, config):
        captured.update(config)
        return MagicMock()

    fake_registry.get.side_effect = _get

    with patch(
        "cogniverse_foundation.telemetry.registry.get_telemetry_registry",
        return_value=fake_registry,
    ):
        manager.get_provider(tenant_id="acme:acme", project_name="routing")

    assert captured.get("project_name") == "routing"
