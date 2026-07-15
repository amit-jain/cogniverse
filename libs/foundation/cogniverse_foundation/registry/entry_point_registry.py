"""Generic plugin registry over ``importlib.metadata`` entry points.

Subclasses declare:

- ``_entry_point_group`` — the entry-point group string under which
  installed packages register implementations (e.g.
  ``"cogniverse.telemetry.providers"``).
- ``_label`` — singular human-readable noun used in log / error
  messages (e.g. ``"telemetry provider"``).

Subclasses may also set ``_tenant_scoped = True`` to opt into the
``initialize()``-style lifecycle: instances are constructed with
``klass()`` then handed the merged ``{tenant_id, **config}`` dict via
``instance.initialize(...)``, and the per-instance cache is keyed by
``(name, tenant_id)``.  The default (``_tenant_scoped = False``) uses
direct ``klass(**config)`` construction and a cache key derived from
``config['backend_url']`` + ``config['backend_port']`` — the shape the
existing workflow- and adapter-store plugins already used.

Conflict detection is always on: if two installed packages register the
same entry-point name, :meth:`EntryPointRegistry.discover` raises
``ValueError`` rather than silently picking one. This matches the
behavior the evaluation and telemetry registries already had — the
workflow- and adapter-store registries silently ignored conflicts,
which would hide a genuine misconfiguration.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from typing import Any, ClassVar, Dict, Generic, List, Optional, Type, TypeVar

from cogniverse_foundation.caching import TenantLRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_TENANT_CACHE_CAPACITY = 16


def _tenant_cache_capacity() -> int:
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "COGNIVERSE_TENANT_CACHE_CAPACITY",
                    _DEFAULT_TENANT_CACHE_CAPACITY,
                )
            ),
        )
    except (TypeError, ValueError):
        return _DEFAULT_TENANT_CACHE_CAPACITY


def _on_instance_evicted(key: str, instance: Any) -> None:
    """Call ``close()`` or ``shutdown()`` on an evicted plugin instance.

    Phoenix / OTLP / Vespa-store plugins hold sockets, HTTP clients, and
    pyvespa app handles. Releasing them on LRU eviction prevents the
    transport pool from leaking one socket per test tenant for the
    lifetime of the process.
    """
    for method in ("close", "shutdown"):
        closer = getattr(instance, method, None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                logger.debug("Plugin %s.%s() failed: %s", key, method, exc)
            return


class EntryPointRegistry(Generic[T]):
    """Base class for entry-point plugin registries.

    See module docstring for the contract subclasses must satisfy.
    """

    _entry_point_group: ClassVar[str]
    _label: ClassVar[str]
    _tenant_scoped: ClassVar[bool] = False

    _instance: ClassVar[Optional["EntryPointRegistry"]] = None
    _registry_classes: ClassVar[Dict[str, Type[Any]]] = {}
    _instances: ClassVar[TenantLRUCache[Any]] = TenantLRUCache(
        capacity=_DEFAULT_TENANT_CACHE_CAPACITY,
        on_evict=_on_instance_evicted,
    )
    _entry_points_loaded: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "_entry_point_group", None):
            raise TypeError(
                f"{cls.__name__} must declare class var "
                "``_entry_point_group`` (e.g. "
                '``_entry_point_group = "cogniverse.<group>.<sub>"``).'
            )
        if not getattr(cls, "_label", None):
            raise TypeError(
                f"{cls.__name__} must declare class var ``_label`` "
                '(singular noun used in diagnostics, e.g. "telemetry provider").'
            )
        cls._registry_classes = {}
        cls._instances = TenantLRUCache(
            capacity=_tenant_cache_capacity(),
            on_evict=_on_instance_evicted,
        )
        cls._entry_points_loaded = False
        cls._instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def discover(cls) -> None:
        """Scan installed packages for entry-point registrations.

        Raises ``ValueError`` if two installed packages register a
        plugin under the same name — that's a genuine misconfiguration
        the operator needs to resolve by uninstalling one package.
        """
        if cls._entry_points_loaded:
            return

        logger.info("Discovering %ss via entry points...", cls._label)

        try:
            entry_points = importlib.metadata.entry_points(group=cls._entry_point_group)
        except TypeError:
            entry_points = importlib.metadata.entry_points().get(
                cls._entry_point_group, []
            )

        for entry_point in entry_points:
            name = entry_point.name
            if name in cls._registry_classes:
                existing_module = cls._registry_classes[name].__module__
                new_module = entry_point.value.split(":")[0]
                if existing_module != new_module:
                    raise ValueError(
                        f"Conflict: {cls._label} '{name}' registered by "
                        f"multiple packages:\n"
                        f"  - {existing_module}\n"
                        f"  - {new_module}\n"
                        "Only one can be installed. Uninstall one package."
                    )
                logger.debug(
                    "%s '%s' already registered; skipping duplicate entry point",
                    cls._label.title(),
                    name,
                )
                continue

            try:
                klass = entry_point.load()
                cls._registry_classes[name] = klass
                logger.info(
                    "Discovered %s: %s (%s)",
                    cls._label,
                    name,
                    entry_point.value,
                )
            except Exception as exc:
                logger.error("Failed to load %s '%s': %s", cls._label, name, exc)

        cls._entry_points_loaded = True

        if not cls._registry_classes:
            logger.warning("No %ss discovered.", cls._label)
        else:
            logger.info(
                "%ss available: %s",
                cls._label.title(),
                list(cls._registry_classes.keys()),
            )

    @classmethod
    def register(cls, name: str, klass: Type[T]) -> None:
        """Manually register a plugin class (bypasses entry-point discovery)."""
        cls._registry_classes[name] = klass
        logger.info("Registered %s: %s", cls._label, name)

    @classmethod
    def list_available(cls) -> List[str]:
        """Names of all discovered + manually-registered plugins."""
        if not cls._entry_points_loaded:
            cls.discover()
        return list(cls._registry_classes.keys())

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Whether a plugin with this name is registered."""
        if not cls._entry_points_loaded:
            cls.discover()
        return name in cls._registry_classes

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all cached plugin instances (triggers close/shutdown)."""
        cls._instances.clear()
        logger.info("Cleared %s cache", cls._label)

    @classmethod
    def reset(cls) -> None:
        """Restore the registry to its pristine state — for tests only."""
        cls._registry_classes.clear()
        cls._instances.clear()
        cls._entry_points_loaded = False
        cls._instance = None

    @classmethod
    def get(
        cls,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> T:
        """Return a (possibly cached) plugin instance.

        ``name=None`` selects the first discovered plugin — the
        "fallback" mode used when only one provider is installed.
        ``tenant_id`` is required when ``_tenant_scoped`` is True.
        """
        if cls._tenant_scoped:
            from cogniverse_foundation.common.tenant_utils import require_tenant_id

            source = f"get_{cls._label}".replace(" ", "_")
            tenant_id = require_tenant_id(tenant_id, source=source)

        if not cls._entry_points_loaded:
            cls.discover()

        if name is None:
            if not cls._registry_classes:
                raise ValueError(
                    f"No {cls._label}s installed. Install a provider package."
                )
            name = next(iter(cls._registry_classes))
            logger.info("No %s specified, using first available: %s", cls._label, name)

        if name not in cls._registry_classes:
            available = list(cls._registry_classes.keys())
            raise ValueError(
                f"{cls._label.title()} '{name}' not found. "
                f"Available: {available or 'none'}"
            )

        cache_key = cls._cache_key(name, config or {}, tenant_id)
        cached = cls._instances.get(cache_key)
        if cached is not None:
            logger.debug("Returning cached %s: %s", cls._label, cache_key)
            return cached

        klass = cls._registry_classes[name]
        instance = cls._create_instance(klass, config or {}, tenant_id)
        # Atomically resolve concurrent cold-starts to one shared instance. A
        # plain set() would displace-and-close an instance another thread is
        # already holding (use-after-close on the store's HTTP/Vespa session);
        # set_if_absent keeps the winner and hands us the loser to release.
        winner = cls._instances.set_if_absent(cache_key, instance)
        if winner is not instance:
            _on_instance_evicted(cache_key, instance)
            return winner
        logger.info("Created %s: %s", cls._label, cache_key)
        return instance

    @classmethod
    def _cache_key(
        cls,
        name: str,
        config: Dict[str, Any],
        tenant_id: Optional[str],
    ) -> str:
        """Cache key for one plugin instance.

        Tenant-scoped registries key by ``(name, tenant_id)`` — one
        instance per tenant for isolation. Config-scoped registries key
        by ``(name, backend_url, backend_port)`` so two callers asking
        for the same backend connection reuse one underlying client.
        """
        if cls._tenant_scoped:
            return f"{name}_{tenant_id}"
        return (
            f"{name}_{config.get('backend_url', '')}_{config.get('backend_port', '')}"
        )

    @classmethod
    def _create_instance(
        cls,
        klass: Type[T],
        config: Dict[str, Any],
        tenant_id: Optional[str],
    ) -> T:
        """Construct one plugin instance from its registered class.

        Tenant-scoped plugins follow the lifecycle pattern: ``klass()``
        then ``.initialize(merged_config)``. Config-scoped plugins are
        constructed directly with ``klass(**config)`` so config keys
        flow through as constructor kwargs.
        """
        if cls._tenant_scoped:
            instance = klass()
            full_config: Dict[str, Any] = {"tenant_id": tenant_id}
            full_config.update(config)
            instance.initialize(full_config)
            return instance
        return klass(**config) if config else klass()
