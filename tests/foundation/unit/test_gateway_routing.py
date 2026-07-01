"""Unit tests for opt-in LLM gateway routing.

These exercise only code we own: the tier/task header resolver, the
``apply_gateway_routing`` transform, and config serialization. They do NOT
stand up a gateway — a real semantic-router/Envoy round-trip is a separate
Docker-backed integration suite. Asserting against a stubbed boundary here
would only re-prove internal wiring, so these stay honestly unit-level and
pin exact values on the objects the code produces.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_foundation.config.gateway_routing import (
    apply_gateway_routing,
    resolve_gateway_headers,
)
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import (
    GatewayRoutingConfig,
    LLMEndpointConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils

DIRECT = "http://vllm-student:8101/v1"
GATEWAY = "http://semantic-router-envoy:8801/v1"


def _enabled_config(**overrides) -> GatewayRoutingConfig:
    base = dict(
        enabled=True,
        gateway_base_url=GATEWAY,
        tenant_tiers={"acme:prod": "pro"},
        default_tier="free",
        agent_tasks={"query_enhancement_agent": "enhance"},
        default_task="general",
    )
    base.update(overrides)
    return GatewayRoutingConfig(**base)


class TestResolveGatewayHeaders:
    def test_returns_none_when_disabled(self):
        cfg = GatewayRoutingConfig(enabled=False)
        assert resolve_gateway_headers(cfg, "acme:prod", "search_agent") is None

    def test_known_tenant_and_agent_map_to_exact_headers(self):
        cfg = _enabled_config()
        assert resolve_gateway_headers(cfg, "acme:prod", "query_enhancement_agent") == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "enhance",
        }

    def test_unknown_tenant_falls_back_to_default_tier(self):
        cfg = _enabled_config()
        headers = resolve_gateway_headers(cfg, "unregistered:tenant", "search_agent")
        assert headers["x-authz-user-groups"] == "free"

    def test_unknown_agent_falls_back_to_default_task(self):
        cfg = _enabled_config()
        headers = resolve_gateway_headers(cfg, "acme:prod", "brand_new_agent")
        assert headers["x-vsr-task"] == "general"

    def test_custom_header_names_are_honored(self):
        cfg = _enabled_config(tier_header="x-tenant-tier", task_header="x-task")
        assert resolve_gateway_headers(cfg, "acme:prod", "query_enhancement_agent") == {
            "x-tenant-tier": "pro",
            "x-task": "enhance",
        }


class TestApplyGatewayRouting:
    def test_disabled_returns_the_same_object_untouched(self):
        cfg = GatewayRoutingConfig(enabled=False)
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        routed = apply_gateway_routing(
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
        routed = apply_gateway_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert routed.api_base == GATEWAY
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
        routed = apply_gateway_routing(
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

    def test_gateway_headers_win_on_key_collision(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-vsr-task": "stale"},
        )
        routed = apply_gateway_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert routed.extra_headers["x-vsr-task"] == "enhance"

    def test_does_not_mutate_the_input_endpoint(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        apply_gateway_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            agent_name="query_enhancement_agent",
        )
        assert endpoint.api_base == DIRECT
        assert endpoint.extra_headers is None

    def test_enabled_without_gateway_base_url_raises(self):
        cfg = _enabled_config(gateway_base_url="")
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        with pytest.raises(ValueError, match="gateway_base_url is empty"):
            apply_gateway_routing(
                endpoint=endpoint,
                config=cfg,
                tenant_id="acme:prod",
                agent_name="search_agent",
            )

    def test_apply_then_factory_wires_gateway_onto_dspy_lm(self):
        # Ties the transform to the factory: the constructed dspy.LM must carry
        # exactly the gateway api_base and the resolved routing headers.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        routed = apply_gateway_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="unregistered:tenant",
            agent_name="query_enhancement_agent",
        )
        lm = create_dspy_lm(routed)
        assert lm.kwargs["api_base"] == GATEWAY
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-groups": "free",
            "x-vsr-task": "enhance",
        }


class TestGatewayRoutingConfigSerialization:
    def test_config_round_trips_exactly(self):
        cfg = _enabled_config()
        rt = GatewayRoutingConfig.from_dict(cfg.to_dict())
        assert (rt.enabled, rt.gateway_base_url) == (True, GATEWAY)
        assert rt.tenant_tiers == {"acme:prod": "pro"}
        assert rt.default_tier == "free"
        assert rt.agent_tasks == {"query_enhancement_agent": "enhance"}
        assert rt.default_task == "general"
        assert (rt.tier_header, rt.task_header) == ("x-authz-user-groups", "x-vsr-task")

    def test_system_config_default_leaves_gateway_disabled(self):
        assert SystemConfig().gateway_routing.enabled is False

    def test_system_config_round_trips_gateway_routing(self):
        syscfg = SystemConfig(gateway_routing=_enabled_config())
        rt = SystemConfig.from_dict(syscfg.to_dict())
        assert rt.gateway_routing.enabled is True
        assert rt.gateway_routing.gateway_base_url == GATEWAY
        assert rt.gateway_routing.tenant_tiers == {"acme:prod": "pro"}
        assert rt.gateway_routing.default_tier == "free"


class TestConfigUtilsGatewayRouting:
    """ConfigUtils exposes SystemConfig.gateway_routing to the LM-build path."""

    def _config_utils(self, system_config) -> ConfigUtils:
        manager = MagicMock()
        manager.get_system_config.return_value = system_config
        return ConfigUtils("acme:prod", config_manager=manager)

    def test_returns_the_system_config_block(self):
        gateway = _enabled_config()
        cu = self._config_utils(SystemConfig(gateway_routing=gateway))
        result = cu.get_gateway_routing()
        assert result is gateway
        assert result.enabled is True
        assert result.gateway_base_url == GATEWAY

    def test_default_system_config_is_disabled(self):
        cu = self._config_utils(SystemConfig())
        assert cu.get_gateway_routing().enabled is False

    def test_missing_field_falls_back_to_disabled_default(self):
        cu = self._config_utils(object())  # no gateway_routing attribute
        result = cu.get_gateway_routing()
        assert isinstance(result, GatewayRoutingConfig)
        assert result.enabled is False
