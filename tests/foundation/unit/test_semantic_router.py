"""Unit tests for opt-in LLM semantic routing.

These exercise only code we own: the tenant-tier header resolver, the
``apply_semantic_routing`` transform, and config serialization. They do NOT
stand up a router — a real semantic-router/Envoy round-trip is a separate
Docker-backed integration suite. Asserting against a stubbed boundary here
would only re-prove internal wiring, so these stay honestly unit-level and
pin exact values on the objects the code produces.

Routing keys on the tenant tier only (the router classifies request content
itself); there is no per-agent task header. And a broken config store raises
rather than silently routing direct.
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
    )
    base.update(overrides)
    return SemanticRouterConfig(**base)


class TestResolveSemanticRouterHeaders:
    def test_returns_none_when_disabled(self):
        cfg = SemanticRouterConfig(enabled=False)
        assert resolve_semantic_router_headers(cfg, "acme:prod") is None

    def test_known_tenant_maps_to_exact_tier_header(self):
        cfg = _enabled_config()
        assert resolve_semantic_router_headers(cfg, "acme:prod") == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_unknown_tenant_falls_back_to_default_tier(self):
        cfg = _enabled_config()
        assert resolve_semantic_router_headers(cfg, "unregistered:tenant") == {
            "x-authz-user-id": "unregistered:tenant",
            "x-authz-user-groups": "free",
        }

    def test_custom_tier_header_name_is_honored(self):
        cfg = _enabled_config(tier_header="x-tenant-tier")
        assert resolve_semantic_router_headers(cfg, "acme:prod") == {
            "x-authz-user-id": "acme:prod",
            "x-tenant-tier": "pro",
        }


class TestApplySemanticRouting:
    def test_disabled_returns_the_same_object_untouched(self):
        cfg = SemanticRouterConfig(enabled=False)
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="acme:prod"
        )
        assert routed is endpoint
        assert routed.api_base == DIRECT
        assert routed.extra_headers is None

    def test_enabled_rewrites_api_base_and_attaches_tier_header(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/some-provider-model", api_base=DIRECT
        )
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="acme:prod"
        )
        assert routed.api_base == SR_URL
        assert routed.extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_enabled_replaces_model_with_router_auto_alias(self):
        # The router resolves models by its own catalog names / auto alias and
        # 400s on raw provider model ids — the routed request must not carry
        # the endpoint's model.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it", api_base=DIRECT
        )
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="acme:prod"
        )
        assert routed.model == "openai/auto"
        assert endpoint.model == "openai/google/gemma-4-e4b-it"

    def test_merges_onto_preexisting_extra_headers(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-trace-id": "abc123"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="acme:prod"
        )
        assert routed.extra_headers == {
            "x-trace-id": "abc123",
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_tier_header_wins_on_key_collision(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-authz-user-groups": "stale"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="acme:prod"
        )
        assert routed.extra_headers["x-authz-user-groups"] == "pro"

    def test_does_not_mutate_the_input_endpoint(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        apply_semantic_routing(endpoint=endpoint, config=cfg, tenant_id="acme:prod")
        assert endpoint.api_base == DIRECT
        assert endpoint.extra_headers is None

    def test_enabled_without_semantic_router_url_raises(self):
        cfg = _enabled_config(semantic_router_url="")
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        with pytest.raises(ValueError, match="semantic_router_url is"):
            apply_semantic_routing(endpoint=endpoint, config=cfg, tenant_id="acme:prod")

    def test_apply_then_factory_wires_semantic_router_onto_dspy_lm(self):
        # Ties the transform to the factory: the constructed dspy.LM must carry
        # exactly the semantic router api_base and the resolved tier header.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint, config=cfg, tenant_id="unregistered:tenant"
        )
        lm = create_dspy_lm(routed)
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "unregistered:tenant",
            "x-authz-user-groups": "free",
        }


class TestSemanticRouterConfigSerialization:
    def test_config_round_trips_exactly(self):
        cfg = _enabled_config()
        rt = SemanticRouterConfig.from_dict(cfg.to_dict())
        assert (rt.enabled, rt.semantic_router_url) == (True, SR_URL)
        assert rt.tenant_tiers == {"acme:prod": "pro"}
        assert rt.default_tier == "free"
        assert rt.tier_header == "x-authz-user-groups"
        assert rt.routed_model == "openai/auto"

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

    def test_raising_accessor_propagates(self):
        # No silent fallback: a broken config store must surface, not disable.
        accessor = MagicMock()
        accessor.get_semantic_router.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            resolve_semantic_router_config(accessor)


class TestCreateRoutedLM:
    def test_enabled_builds_lm_on_semantic_router_with_tier_header(self):
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint, config=_enabled_config(), tenant_id="acme:prod"
        )
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_disabled_builds_lm_on_direct_endpoint(self):
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint,
            config=SemanticRouterConfig(enabled=False),
            tenant_id="acme:prod",
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
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_config_error_propagates(self, monkeypatch):
        # No silent fallback: a broken config store surfaces, even with an
        # endpoint in hand — it does NOT quietly build a direct LM.
        def boom(**kw):
            raise RuntimeError("config store down")

        monkeypatch.setattr("cogniverse_foundation.config.utils.get_config", boom)
        endpoint = LLMEndpointConfig(model="openai/local", api_base=DIRECT)
        with pytest.raises(RuntimeError, match="config store down"):
            routed_lm_context_for(
                MagicMock(), "acme:prod", "summarizer_agent", endpoint=endpoint
            )

    def test_enabled_routes_the_given_endpoint(self, monkeypatch):
        # endpoint param: route the agent's own endpoint rather than
        # re-resolving from config. The routed request carries the router's
        # auto alias — the router picks the concrete model.
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = _enabled_config()
        self._patch_get_config(monkeypatch, cfg)
        endpoint = LLMEndpointConfig(model="openai/tuned", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(), "acme:prod", "knowledge_summarization_agent", endpoint=endpoint
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/auto"  # router's auto alias, not the raw id
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
