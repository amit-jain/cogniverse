"""Unit tests for opt-in LLM router routing.

These exercise only code we own: the tier/task header resolver, the
``apply_semantic_routing`` transform, and config serialization. They do NOT
stand up a router — a real semantic-router/Envoy round-trip is a separate
Docker-backed integration suite. Asserting against a stubbed boundary here
would only re-prove internal wiring, so these stay honestly unit-level and
pin exact values on the objects the code produces.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import (
    apply_semantic_routing,
    create_routed_lm,
    resolve_semantic_router_config,
    resolve_semantic_router_headers,
    routed_lm_context_for,
)
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils

DIRECT = "http://vllm-student:8101/v1"
SR_URL = "http://semantic-router-envoy:8801/v1"


def _enabled_config(**overrides) -> SemanticRouterConfig:
    base = dict(
        enabled=True,
        semantic_router_url=SR_URL,
        tenant_tiers={"acme:prod": "pro"},
        default_tier="free",
        agent_tasks={"query_enhancement_agent": "enhance"},
        default_task="general",
    )
    base.update(overrides)
    return SemanticRouterConfig(**base)


class TestResolveSemanticRouterHeaders:
    def test_returns_none_when_disabled(self):
        cfg = SemanticRouterConfig(enabled=False)
        assert resolve_semantic_router_headers(cfg, "acme:prod", "search_agent") is None

    def test_known_tenant_and_agent_map_to_exact_headers(self):
        cfg = _enabled_config()
        assert resolve_semantic_router_headers(
            cfg, "acme:prod", "query_enhancement_agent"
        ) == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_unknown_tenant_falls_back_to_default_tier(self):
        cfg = _enabled_config()
        headers = resolve_semantic_router_headers(
            cfg, "unregistered:tenant", "search_agent"
        )
        assert headers["x-authz-user-groups"] == "free"

    def test_unknown_agent_falls_back_to_default_task(self):
        cfg = _enabled_config()
        headers = resolve_semantic_router_headers(cfg, "acme:prod", "brand_new_agent")
        assert headers["x-vsr-task"] == "general"

    def test_custom_header_names_are_honored(self):
        cfg = _enabled_config(tier_header="x-tenant-tier", task_header="x-task")
        assert resolve_semantic_router_headers(
            cfg, "acme:prod", "query_enhancement_agent"
        ) == {
            "x-tenant-tier": "pro",
            "x-task": "enhance",
        }


class TestApplySemanticRouting:
    def test_disabled_returns_the_same_object_untouched(self):
        cfg = SemanticRouterConfig(enabled=False)
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="search_agent",
        )
        assert routed is endpoint
        assert routed.api_base == DIRECT
        assert routed.extra_headers is None

    def test_enabled_rewrites_api_base_and_attaches_exact_headers(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert routed.api_base == SR_URL
        assert routed.extra_headers == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_merges_onto_preexisting_extra_headers(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-trace-id": "abc123"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert routed.extra_headers == {
            "x-trace-id": "abc123",
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_semantic_router_headers_win_on_key_collision(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-vsr-task": "stale"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert routed.extra_headers["x-vsr-task"] == "enhance"

    def test_does_not_mutate_the_input_endpoint(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert endpoint.api_base == DIRECT
        assert endpoint.extra_headers is None

    def test_enabled_without_semantic_router_url_raises(self):
        cfg = _enabled_config(semantic_router_url="")
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        with pytest.raises(ValueError, match="semantic_router_url is empty"):
            apply_semantic_routing(
                endpoint=endpoint,
                config=cfg,
                tenant_id="acme:prod",
                agent_name="search_agent",
            )

    def test_apply_then_factory_wires_semantic_router_onto_dspy_lm(self):
        # Ties the transform to the factory: the constructed dspy.LM must carry
        # exactly the semantic router api_base and the resolved routing headers.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="unregistered:tenant",
            agent_name="query_enhancement_agent",
        )
        lm = create_dspy_lm(routed)
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-groups": "free",
            "x-vsr-task": "enhance",
        }


class TestSemanticRouterConfigSerialization:
    def test_config_round_trips_exactly(self):
        cfg = _enabled_config()
        rt = SemanticRouterConfig.from_dict(cfg.to_dict())
        assert (rt.enabled, rt.semantic_router_url) == (True, SR_URL)
        assert rt.tenant_tiers == {"acme:prod": "pro"}
        assert rt.default_tier == "free"
        assert rt.agent_tasks == {"query_enhancement_agent": "enhance"}
        assert rt.default_task == "general"
        assert (rt.tier_header, rt.task_header) == ("x-authz-user-groups", "x-vsr-task")

    def test_system_config_default_leaves_semantic_router_disabled(self):
        assert SystemConfig().semantic_router.enabled is False

    def test_system_config_round_trips_semantic_router(self):
        syscfg = SystemConfig(semantic_router=_enabled_config())
        rt = SystemConfig.from_dict(syscfg.to_dict())
        assert rt.semantic_router.enabled is True
        assert rt.semantic_router.semantic_router_url == SR_URL
        assert rt.semantic_router.tenant_tiers == {"acme:prod": "pro"}
        assert rt.semantic_router.default_tier == "free"


class TestConfigUtilsSemanticRouting:
    """ConfigUtils exposes SystemConfig.semantic_router to the LM-build path."""

    def _config_utils(self, system_config) -> ConfigUtils:
        manager = MagicMock()
        manager.get_system_config.return_value = system_config
        return ConfigUtils("acme:prod", config_manager=manager)

    def test_returns_the_system_config_block(self):
        router = _enabled_config()
        cu = self._config_utils(SystemConfig(semantic_router=router))
        result = cu.get_semantic_router()
        assert result is router
        assert result.enabled is True
        assert result.semantic_router_url == SR_URL

    def test_default_system_config_is_disabled(self):
        cu = self._config_utils(SystemConfig())
        assert cu.get_semantic_router().enabled is False

    def test_missing_field_falls_back_to_disabled_default(self):
        cu = self._config_utils(object())  # no semantic_router attribute
        result = cu.get_semantic_router()
        assert isinstance(result, SemanticRouterConfig)
        assert result.enabled is False


class TestResolveSemanticRouterConfig:
    def test_absent_accessor_returns_disabled_default(self):
        result = resolve_semantic_router_config(object())
        assert isinstance(result, SemanticRouterConfig)
        assert result.enabled is False

    def test_valid_accessor_returns_the_config(self):
        router = _enabled_config()
        accessor = MagicMock()
        accessor.get_semantic_router.return_value = router
        assert resolve_semantic_router_config(accessor) is router

    def test_mocked_accessor_value_is_rejected(self):
        # A bare MagicMock's get_semantic_router() returns a MagicMock whose
        # .enabled is truthy; the isinstance guard must reject it.
        result = resolve_semantic_router_config(MagicMock())
        assert isinstance(result, SemanticRouterConfig)
        assert result.enabled is False

    def test_raising_accessor_returns_disabled_default(self):
        accessor = MagicMock()
        accessor.get_semantic_router.side_effect = RuntimeError("boom")
        assert resolve_semantic_router_config(accessor).enabled is False


class TestCreateRoutedLM:
    def test_enabled_builds_lm_on_semantic_router_with_headers(self):
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint,
            config=_enabled_config(),
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_disabled_builds_lm_on_direct_endpoint(self):
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint,
            config=SemanticRouterConfig(enabled=False),
            tenant_id="acme:prod",
            agent_name="search_agent",
        )
        assert lm.kwargs["api_base"] == DIRECT
        assert "extra_headers" not in lm.kwargs


class TestRoutedLMContextFor:
    """Per-request routing context for tenant-agnostic agents."""

    def _patch_get_config(self, monkeypatch, cfg):
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
        )

    def test_no_endpoint_disabled_is_nullcontext(self, monkeypatch):
        # No endpoint supplied (orchestrator case) + disabled => ambient LM.
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        self._patch_get_config(monkeypatch, cfg)
        ctx = routed_lm_context_for(MagicMock(), "acme:prod", "orchestrator_agent")
        assert isinstance(ctx, nullcontext)

    def test_enabled_routes_through_semantic_router(self, monkeypatch):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = _enabled_config()
        cfg.get_llm_config.return_value.resolve.return_value = LLMEndpointConfig(
            model="openai/s", api_base=DIRECT
        )
        self._patch_get_config(monkeypatch, cfg)
        with routed_lm_context_for(MagicMock(), "acme:prod", "query_enhancement_agent"):
            lm = dspy.settings.lm
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_error_with_endpoint_builds_from_that_endpoint(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("config store down")

        monkeypatch.setattr("cogniverse_foundation.config.utils.get_config", boom)
        endpoint = LLMEndpointConfig(model="openai/local", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(), "acme:prod", "summarizer_agent", endpoint=endpoint
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/local"
        assert lm.kwargs["api_base"] == DIRECT

    def test_enabled_routes_the_given_endpoint(self, monkeypatch):
        # endpoint param: route the agent's own endpoint (preserving its model)
        # rather than re-resolving from config.
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = _enabled_config()
        self._patch_get_config(monkeypatch, cfg)
        endpoint = LLMEndpointConfig(model="openai/tuned", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(), "acme:prod", "knowledge_summarization_agent", endpoint=endpoint
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/tuned"  # model preserved
        assert lm.kwargs["api_base"] == SR_URL  # routed
        cfg.get_llm_config.assert_not_called()  # endpoint used, no re-resolve

    def test_disabled_with_endpoint_builds_from_that_endpoint(self, monkeypatch):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        self._patch_get_config(monkeypatch, cfg)
        endpoint = LLMEndpointConfig(model="openai/tuned", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(),
            "acme:prod",
            "multi_document_synthesis_agent",
            endpoint=endpoint,
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/tuned"
        assert lm.kwargs["api_base"] == DIRECT  # not routed
        assert "extra_headers" not in lm.kwargs
