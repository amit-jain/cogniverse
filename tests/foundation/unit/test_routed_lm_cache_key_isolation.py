"""Routed requests include both canonical tenant identity and routing headers."""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.lm_response_cache import request_cache_key
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

pytestmark = [pytest.mark.unit]

_ENDPOINT = LLMEndpointConfig(
    model="openai/google/gemma-4-e4b-it",
    api_base="https://llm.example/v1",
)
_ROUTER = SemanticRouterConfig(
    enabled=True,
    semantic_router_url="http://router.example/v1",
)
_TIERS = {"acme:prod": "pro", "globex:prod": "free"}
_MESSAGES = [{"role": "user", "content": "rewrite: people exercising"}]


def _cache_key(tenant_id: str, tier: str | None = None) -> str:
    """The owned response cache key for this tenant and routed request."""
    routed = apply_semantic_routing(
        endpoint=_ENDPOINT,
        config=_ROUTER,
        tenant_id=tenant_id,
        tier=tier or _TIERS.get(tenant_id, "default"),
    )
    request = {
        "model": routed.model,
        "messages": _MESSAGES,
        "api_base": routed.api_base,
        "extra_headers": routed.extra_headers,
    }
    return request_cache_key(tenant_id, request)


class TestTheCacheKeySeparatesTenants:
    def test_two_tenants_asking_the_same_thing_get_different_keys(self):
        """Identical prompt, different tenant: no shared entry."""
        assert _cache_key("acme:prod") != _cache_key("globex:prod")

    def test_the_same_tenant_asking_the_same_thing_reuses_one_key(self):
        """The cache still has to work; separation is not just salting."""
        assert _cache_key("acme:prod") == _cache_key("acme:prod")

    def test_two_tenants_on_the_same_tier_still_get_different_keys(self):
        """Identity, not only tier, is in the key: two free-tier tenants must
        not share a rewrite either."""
        assert _cache_key("a:prod", "free") != _cache_key("b:prod", "free")

    def test_the_tier_is_part_of_the_key(self):
        """Tier selects the model, so one tenant's tier change must not be
        answered out of the entry written under its old tier."""
        assert _cache_key("acme:prod", "pro") != _cache_key("acme:prod", "free")

    def test_the_headers_that_carry_the_tenant_are_actually_on_the_endpoint(self):
        """Both routing headers accompany the endpoint request."""
        routed = apply_semantic_routing(
            endpoint=_ENDPOINT, config=_ROUTER, tenant_id="acme:prod", tier="pro"
        )
        assert routed.extra_headers == {
            _ROUTER.user_id_header: "acme:prod",
            _ROUTER.tier_header: "pro",
        }
