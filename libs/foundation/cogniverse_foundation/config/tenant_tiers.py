"""Per-tenant router tier: the stored attribute and the reader the seam uses.

A tenant's tier is one ``RouterTier`` value held in the configuration store
under ``ConfigScope.ROUTING`` / ``TENANT_TIER_SERVICE`` / ``TENANT_TIER_KEY``,
addressed by canonical tenant id. A tenant with no row is
``DEFAULT_ROUTER_TIER``: absence is the default, so a tenant is routable
without a row ever being written for it.

``TenantRouterTiers`` is the read path a request takes. It holds the answer per
canonical tenant in a ``RefreshingCache``: the request thread reads the store
only for a tenant it holds nothing for, or holds a tier
``TENANT_TIER_MAX_STALENESS_S`` old. Every write through ``set_tenant_tier``
drops the written tenant from every reader in this process, so an operator's
change is visible to the next request here; ``TENANT_TIER_MAX_STALENESS_S``
bounds how long another replica keeps serving the tier it read before that
write.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import ClassVar

from cogniverse_foundation.caching.refreshing_cache import RefreshingCache
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
    RouterTier,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope

logger = logging.getLogger(__name__)

TENANT_TIER_SERVICE = "semantic_router"
TENANT_TIER_KEY = "tenant_tier"
TENANT_TIER_VALUE_FIELD = "tier"
# Age at which a held tier is re-read on a background thread while it keeps
# answering.
TENANT_TIER_REFRESH_S = 15.0
# Oldest held tier that answers: how long another replica's write can go unseen
# here.
TENANT_TIER_MAX_STALENESS_S = 30.0
# Most tenants one reader holds.
TENANT_TIER_MAX_TENANTS = 512


def validate_router_tier(tier: str) -> RouterTier:
    """Return ``tier`` when the router binds a group for it, else raise.

    The message names the whole vocabulary because a tier outside it matches
    no routing decision and falls through to the default model.
    """
    if tier not in ROUTER_TIERS:
        raise ValueError(
            f"Unknown router tier {tier!r}. Valid tiers: {sorted(ROUTER_TIERS)}"
        )
    return tier  # type: ignore[return-value]


def read_tenant_tier(config_manager, tenant_id: str) -> RouterTier:
    """The tenant's stored tier, or ``DEFAULT_ROUTER_TIER`` when it has none.

    Reads the store directly, uncached. A store failure propagates: a tier
    read that failed is not a tenant on the default tier.
    """
    canonical = canonical_tenant_id(tenant_id)
    entry = config_manager.store.get_config(
        tenant_id=canonical,
        scope=ConfigScope.ROUTING,
        service=TENANT_TIER_SERVICE,
        config_key=TENANT_TIER_KEY,
    )
    if entry is None:
        return DEFAULT_ROUTER_TIER
    stored = entry.config_value.get(TENANT_TIER_VALUE_FIELD)
    if stored not in ROUTER_TIERS:
        raise ValueError(
            f"Tenant {canonical!r} has stored tier {stored!r}, which the router "
            f"binds no group for. Valid tiers: {sorted(ROUTER_TIERS)}"
        )
    return stored  # type: ignore[return-value]


def set_tenant_tier(config_manager, tenant_id: str, tier: str) -> RouterTier:
    """Store ``tier`` for the tenant and drop it from every reader here.

    Returns the stored tier. Raises ``ValueError`` before touching the store
    when ``tier`` is outside ``ROUTER_TIERS``.
    """
    validated = validate_router_tier(tier)
    canonical = canonical_tenant_id(tenant_id)
    try:
        config_manager.store.set_config(
            tenant_id=canonical,
            scope=ConfigScope.ROUTING,
            service=TENANT_TIER_SERVICE,
            config_key=TENANT_TIER_KEY,
            config_value={TENANT_TIER_VALUE_FIELD: validated},
        )
    finally:
        invalidate_tenant_tier(canonical)
    return validated


class TenantRouterTiers:
    """A tenant's router tier, held per canonical tenant.

    ``reader(tenant_id)`` answers a ``RouterTier``. A held tier answers with no
    store read until ``refresh_after_s`` after the read that produced it began;
    until ``max_staleness_s`` it still answers while one background read
    replaces it, so the caller never waits on that read. A tenant with nothing
    held, or a tier ``max_staleness_s`` old, is read on the caller's thread;
    concurrent callers share that read, and a failure raises to each of them
    and caches nothing, so an outage is never recorded as a tier. A failed
    background read is logged and the held tier answers until
    ``max_staleness_s``.
    """

    _live: ClassVar["weakref.WeakSet[TenantRouterTiers]"] = weakref.WeakSet()
    _live_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config_manager,
        refresh_after_s: float = TENANT_TIER_REFRESH_S,
        max_staleness_s: float = TENANT_TIER_MAX_STALENESS_S,
    ) -> None:
        self._config_manager = config_manager
        self._tiers: RefreshingCache[str, RouterTier] = RefreshingCache(
            name="router-tier",
            refresh_after_s=refresh_after_s,
            max_staleness_s=max_staleness_s,
            max_entries=TENANT_TIER_MAX_TENANTS,
        )
        with TenantRouterTiers._live_lock:
            TenantRouterTiers._live.add(self)

    @property
    def config_manager(self):
        return self._config_manager

    @property
    def refresh_after_s(self) -> float:
        return self._tiers.refresh_after_s

    @property
    def max_staleness_s(self) -> float:
        return self._tiers.max_staleness_s

    def __call__(self, tenant_id: str) -> RouterTier:
        canonical = canonical_tenant_id(tenant_id)
        return self._tiers.get(
            canonical, lambda: read_tenant_tier(self._config_manager, canonical)
        )

    def invalidate(self, tenant_id: str) -> None:
        """Drop the tenant's entry and detach any read in flight for it."""
        canonical = canonical_tenant_id(tenant_id)
        self._tiers.invalidate(lambda key: key == canonical)


def invalidate_tenant_tier(tenant_id: str) -> None:
    """Drop the tenant's tier from every reader in this process."""
    with TenantRouterTiers._live_lock:
        readers = list(TenantRouterTiers._live)
    for reader in readers:
        reader.invalidate(tenant_id)


_readers: "weakref.WeakKeyDictionary[object, TenantRouterTiers]" = (
    weakref.WeakKeyDictionary()
)
_readers_lock = threading.Lock()


def tenant_tier_reader(config_manager) -> TenantRouterTiers:
    """The reader bound to ``config_manager``, built once per manager.

    One reader per manager is what makes the cache worth having: a reader
    built per request would read the store on every call.
    """
    with _readers_lock:
        reader = _readers.get(config_manager)
        if reader is None:
            reader = _readers[config_manager] = TenantRouterTiers(config_manager)
        return reader


def resolve_tenant_tier(config_accessor, tenant_id: str) -> RouterTier:
    """The tier to send for ``tenant_id``, resolved through the request's config.

    ``config_accessor`` is the per-tenant config the caller already holds (a
    ``ConfigUtils``); its ``config_manager`` owns the store the tier lives in.

    A store failure resolves to ``DEFAULT_ROUTER_TIER`` and logs a WARNING
    naming the tenant and the error: the request is then served on the default
    model, where raising would fail a request the tier only influences. An
    accessor exposing no ``config_manager`` raises instead -- that is a caller
    which cannot read tiers at all, not an outage, and degrading it would put
    every tenant on the default tier permanently and silently.
    """
    config_manager = getattr(config_accessor, "config_manager", None)
    if config_manager is None:
        raise TypeError(
            "resolve_tenant_tier needs a config accessor exposing "
            f"config_manager; got {type(config_accessor).__name__}"
        )
    try:
        return tenant_tier_reader(config_manager)(tenant_id)
    except Exception as exc:
        logger.warning(
            "Router tier read failed for tenant %s (%s: %s); routing as %s",
            tenant_id,
            type(exc).__name__,
            exc,
            DEFAULT_ROUTER_TIER,
        )
        return DEFAULT_ROUTER_TIER
