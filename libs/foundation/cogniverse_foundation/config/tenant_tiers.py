"""Per-tenant router tier: the stored attribute and the reader the seam uses.

A tenant's tier is one ``RouterTier`` value held in the configuration store
under ``ConfigScope.ROUTING`` / ``TENANT_TIER_SERVICE`` / ``TENANT_TIER_KEY``,
addressed by canonical tenant id. A tenant with no row is
``DEFAULT_ROUTER_TIER``: absence is the default, so a tenant is routable
without a row ever being written for it.

``TenantRouterTiers`` is the read path a request takes. It caches the answer
per canonical tenant for ``ttl_s`` and shares one store read between concurrent
first-touches. Every write through ``set_tenant_tier`` drops the written tenant
from every reader in this process, so an operator's change is visible to the
next request here; ``ttl_s`` bounds only how long another replica keeps serving
the tier it read before that write.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from concurrent.futures import Future
from typing import ClassVar, Dict

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
TENANT_TIER_TTL_S = 30.0


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
    """A tenant's router tier, cached per canonical tenant for ``ttl_s``.

    ``reader(tenant_id)`` answers a ``RouterTier``. A cached tenant answers
    with no store read until ``ttl_s`` after the read that produced it began.
    Concurrent first-touches for one tenant share a single store read; a failed
    read raises to every caller and caches nothing, so an outage is never
    recorded as a tier.
    """

    _live: ClassVar["weakref.WeakSet[TenantRouterTiers]"] = weakref.WeakSet()
    _live_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config_manager, ttl_s: float = TENANT_TIER_TTL_S) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be positive, got {ttl_s}")
        self._config_manager = config_manager
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._entries: Dict[str, tuple[RouterTier, float]] = {}
        self._reads: Dict[str, Future] = {}
        with TenantRouterTiers._live_lock:
            TenantRouterTiers._live.add(self)

    @property
    def config_manager(self):
        return self._config_manager

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def __call__(self, tenant_id: str) -> RouterTier:
        canonical = canonical_tenant_id(tenant_id)
        with self._lock:
            cached = self._entries.get(canonical)
            if cached is not None and time.monotonic() < cached[1]:
                return cached[0]
        return self._read(canonical)

    def _read(self, canonical: str) -> RouterTier:
        with self._lock:
            started = time.monotonic()
            read = self._reads.get(canonical)
            if read is not None:
                owner = False
            else:
                owner = True
                read = self._reads[canonical] = Future()
        if not owner:
            return read.result()
        try:
            tier = read_tenant_tier(self._config_manager, canonical)
        except BaseException as exc:
            with self._lock:
                if self._reads.get(canonical) is read:
                    del self._reads[canonical]
            read.set_exception(exc)
            raise
        with self._lock:
            # An invalidation during the read detached it: the answer may
            # predate that write, so it is returned but never cached.
            if self._reads.get(canonical) is read:
                del self._reads[canonical]
                for expired in [
                    key for key, (_, until) in self._entries.items() if until <= started
                ]:
                    del self._entries[expired]
                self._entries[canonical] = (tier, started + self._ttl_s)
        read.set_result(tier)
        return tier

    def invalidate(self, tenant_id: str) -> None:
        """Drop the tenant's entry and detach any read in flight for it."""
        canonical = canonical_tenant_id(tenant_id)
        with self._lock:
            self._entries.pop(canonical, None)
            self._reads.pop(canonical, None)


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
