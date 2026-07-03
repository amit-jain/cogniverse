"""Unit tests for ``ConfigUtils`` LM-key resolution.

Agents build their per-tenant DSPy LM from ``llm_model`` / ``llm_base_url`` /
``llm_api_key``. These resolve from config.json's complete ``llm_config.primary``
(model + in-cluster api_base + no-auth key) so litellm targets the in-cluster
endpoint; they fall back to the piecemeal system config only when primary is
unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cogniverse_foundation.config.unified_config import (
    LLMConfig,
    LLMEndpointConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils


def _utils(system_config: SystemConfig) -> ConfigUtils:
    cu = ConfigUtils("acme:acme", config_manager=MagicMock())
    cu._system_config = system_config  # pin lazy system config (no backend)
    return cu


def test_llm_fields_resolve_from_primary():
    primary = LLMEndpointConfig(
        model="openai/gemma3:4b",
        api_base="http://cogniverse-llm:11434/v1",
        api_key="placeholder-no-auth-needed",
    )
    cu = _utils(SystemConfig(llm_model="bare:model", base_url=""))
    with patch.object(
        cu, "get_llm_config", return_value=LLMConfig(primary=primary, teacher=primary)
    ):
        assert cu.get("llm_model") == "openai/gemma3:4b"
        assert cu.get("llm_base_url") == "http://cogniverse-llm:11434/v1"
        assert cu.get("llm_api_key") == "placeholder-no-auth-needed"


def test_llm_base_url_falls_back_to_system_config_when_primary_missing():
    cu = _utils(SystemConfig(base_url="http://fallback:8000/v1"))
    # No config.json llm_config → get_llm_config raises → fall back.
    with patch.object(cu, "get_llm_config", side_effect=ValueError("missing")):
        assert cu.get("llm_base_url") == "http://fallback:8000/v1"


def test_empty_primary_api_base_falls_back():
    primary = LLMEndpointConfig(model="openai/gemma3:4b", api_base="")
    cu = _utils(SystemConfig(base_url="http://fallback:8000/v1"))
    with patch.object(
        cu, "get_llm_config", return_value=LLMConfig(primary=primary, teacher=primary)
    ):
        # Empty primary api_base must not shadow a usable system base_url.
        assert cu.get("llm_base_url") == "http://fallback:8000/v1"


class TestScopedConfigCache:
    """ConfigManager must serve per-tenant scoped configs from its TTL cache
    — the ensure cascade previously paid one store round-trip (a YQL query
    against Vespa) per config group per request."""

    def _manager(self, ttl: float = 5.0):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import BackendConfig

        store = MagicMock()
        entry = MagicMock()
        entry.config_value = BackendConfig(
            tenant_id="acme:acme",
            profiles={},
        ).to_dict()
        store.get_config.return_value = entry
        return (
            ConfigManager(store=store, scoped_config_cache_ttl_s=ttl),
            store,
        )

    def test_repeat_reads_hit_the_store_once(self):
        manager, store = self._manager()
        for _ in range(4):
            manager.get_backend_config(tenant_id="acme:acme")
        assert store.get_config.call_count == 1

    def test_setter_invalidates_immediately(self):
        from cogniverse_foundation.config.unified_config import BackendConfig

        manager, store = self._manager()
        manager.get_backend_config(tenant_id="acme:acme")
        manager.set_backend_config(
            BackendConfig(tenant_id="acme:acme"), tenant_id="acme:acme"
        )
        manager.get_backend_config(tenant_id="acme:acme")
        assert store.get_config.call_count == 2

    def test_ttl_expiry_reconsults_the_store(self):
        manager, store = self._manager(ttl=0.0)
        manager.get_backend_config(tenant_id="acme:acme")
        manager.get_backend_config(tenant_id="acme:acme")
        assert store.get_config.call_count == 2

    def test_returned_config_mutation_does_not_leak_into_cache(self):
        from cogniverse_foundation.config.unified_config import (
            BackendProfileConfig,
        )

        manager, store = self._manager()
        first = manager.get_backend_config(tenant_id="acme:acme")
        first.profiles["injected"] = BackendProfileConfig.from_dict(
            "injected", {"embedding_model": "x", "schema_name": "s"}
        )
        second = manager.get_backend_config(tenant_id="acme:acme")
        assert "injected" not in second.profiles

    def test_absent_config_is_cached_as_default(self):
        from cogniverse_foundation.config.manager import ConfigManager

        store = MagicMock()
        store.get_config.return_value = None
        manager = ConfigManager(store=store, scoped_config_cache_ttl_s=5.0)
        for _ in range(3):
            cfg = manager.get_routing_config(tenant_id="acme:acme")
            assert cfg.tenant_id == "acme:acme"
        assert store.get_config.call_count == 1


class TestJsonConfigCache:
    """config.json is parsed once per (path, mtime) and shared across
    ConfigUtils instances — get_config builds a fresh instance per call."""

    def test_second_instance_reuses_parsed_json(self, tmp_path, monkeypatch):
        import json as json_mod

        from cogniverse_foundation.config import utils as utils_mod

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"backend": {"port": 8080}, "marker": 1}')
        monkeypatch.setenv("COGNIVERSE_CONFIG", str(cfg_file))
        utils_mod._JSON_CONFIG_CACHE.clear()

        loads = []
        real_load = json_mod.load

        def counting_load(fh):
            loads.append(fh.name)
            return real_load(fh)

        monkeypatch.setattr(utils_mod.json, "load", counting_load)

        a = ConfigUtils("acme:acme", config_manager=MagicMock())
        a._load_json_config()
        b = ConfigUtils("acme:acme", config_manager=MagicMock())
        b._load_json_config()

        assert loads == [str(cfg_file)], "file must be parsed exactly once"
        assert a._json_config == b._json_config
        # Instances get isolated copies — mutating one can't corrupt the other.
        a._json_config["marker"] = 999
        assert b._json_config["marker"] == 1

        # Editing the file (new mtime) must invalidate.
        import os as os_mod

        cfg_file.write_text('{"backend": {"port": 9090}, "marker": 2}')
        os_mod.utime(cfg_file, (1e9, 2e9))
        c = ConfigUtils("acme:acme", config_manager=MagicMock())
        c._load_json_config()
        assert c._json_config["marker"] == 2

        utils_mod._JSON_CONFIG_CACHE.clear()
