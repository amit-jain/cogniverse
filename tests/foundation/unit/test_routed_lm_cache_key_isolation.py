"""A cached rewrite must never cross a tenant boundary.

The serving path caches LM calls in DSPy's process-global cache, whose key is a
hash of the request minus ``api_key``/``api_base``/``base_url``. Two tenants
issuing the same query therefore share a cache entry unless something
tenant-specific is inside that hashed request. Semantic routing puts it there:
the authz identity and tier headers ride on ``extra_headers``, which is part of
the hashed request. These pin that, and pin that the tier is in the key too -
tier selects the model, so two tiers sharing an entry would serve one tier's
model to the other.
"""

from __future__ import annotations

import pytest
from dspy.clients.cache import Cache

from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

pytestmark = [pytest.mark.unit]

# The arguments DSPy excludes from the cache key; everything else is hashed.
_IGNORED = ["api_key", "api_base", "base_url"]

_ENDPOINT = LLMEndpointConfig(
    model="openai/google/gemma-4-e4b-it",
    api_base="https://llm.example/v1",
)
_ROUTER = SemanticRouterConfig(
    enabled=True,
    semantic_router_url="http://router.example/v1",
    tenant_tiers={"acme:prod": "pro", "globex:prod": "free"},
    default_tier="default",
)
_MESSAGES = [{"role": "user", "content": "rewrite: people exercising"}]


def _cache_key(tenant_id: str, router: SemanticRouterConfig = _ROUTER) -> str:
    """The key DSPy would store this tenant's rewrite under."""
    routed = apply_semantic_routing(
        endpoint=_ENDPOINT, config=router, tenant_id=tenant_id
    )
    request = {
        "model": routed.model,
        "messages": _MESSAGES,
        "api_base": routed.api_base,
        "extra_headers": routed.extra_headers,
    }
    return Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir="",
    ).cache_key(request, ignored_args_for_cache_key=_IGNORED)


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
        router = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://router.example/v1",
            tenant_tiers={"a:prod": "free", "b:prod": "free"},
            default_tier="default",
        )
        assert _cache_key("a:prod", router) != _cache_key("b:prod", router)

    def test_the_tier_is_part_of_the_key(self):
        """Tier selects the model, so one tenant's tier change must not be
        answered out of the entry written under its old tier."""
        promoted = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://router.example/v1",
            tenant_tiers={"acme:prod": "free"},
            default_tier="default",
        )
        assert _cache_key("acme:prod") != _cache_key("acme:prod", promoted)

    def test_the_headers_that_carry_the_tenant_are_actually_on_the_endpoint(self):
        """The separation above only holds because these two headers ride on
        extra_headers, which DSPy hashes. Pin the exact pair."""
        routed = apply_semantic_routing(
            endpoint=_ENDPOINT, config=_ROUTER, tenant_id="acme:prod"
        )
        assert routed.extra_headers == {
            _ROUTER.user_id_header: "acme:prod",
            _ROUTER.tier_header: "pro",
        }
