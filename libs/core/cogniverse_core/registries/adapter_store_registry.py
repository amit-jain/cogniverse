"""
AdapterStore Registry - Auto-discovery and registration of adapter store implementations.

This module provides a registry pattern for adapter stores to self-register,
enabling a pluggable architecture where new stores can be added without
modifying core code.
"""

import importlib.metadata
import logging
from typing import Any, Dict, Optional, Type

from cogniverse_sdk.interfaces.adapter_store import AdapterStore

logger = logging.getLogger(__name__)


class AdapterStoreRegistry:
    """
    Central registry for adapter store auto-discovery and management.

    Stores self-register via entry points, allowing the app layer
    to discover and use them without direct imports.

    Entry point group: cogniverse.adapter.stores
    Example entry point:
        [project.entry-points."cogniverse.adapter.stores"]
        vespa = "cogniverse_vespa.registry.adapter_store:VespaAdapterStore"
    """

    _instance = None
    _stores: Dict[str, Type[AdapterStore]] = {}
    _store_instances: Dict[str, AdapterStore] = {}
    _entry_points_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AdapterStoreRegistry":
        """Get the singleton AdapterStoreRegistry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def discover_stores(cls) -> None:
        """
        Auto-discover adapter stores via entry points.

        Scans installed packages for entry points in group
        'cogniverse.adapter.stores'. Called lazily on first
        get_adapter_store() call.
        """
        if cls._entry_points_loaded:
            return

        logger.info("Discovering adapter stores via entry points...")

        try:
            # Python 3.10+ API
            entry_points = importlib.metadata.entry_points(
                group="cogniverse.adapter.stores"
            )
        except TypeError:
            # Python 3.9 fallback
            entry_points = importlib.metadata.entry_points().get(
                "cogniverse.adapter.stores", []
            )

        for entry_point in entry_points:
            name = entry_point.name

            if name in cls._stores:
                logger.debug(
                    f"Adapter store '{name}' already registered, skipping duplicate"
                )
                continue

            try:
                store_class = entry_point.load()
                cls._stores[name] = store_class
                logger.info(
                    f"Discovered adapter store: {name} ({entry_point.value})"
                )
            except Exception as e:
                logger.error(f"Failed to load adapter store '{name}': {e}")

        cls._entry_points_loaded = True

        if not cls._stores:
            logger.warning(
                "No adapter stores discovered. "
                "Install cogniverse-vespa or another store package."
            )
        else:
            logger.info(f"Adapter stores available: {list(cls._stores.keys())}")

    @classmethod
    def register(cls, name: str, store_class: Type[AdapterStore]) -> None:
        """
        Manually register an adapter store.

        Args:
            name: Store name (e.g., "vespa", "sqlite")
            store_class: Store class implementing AdapterStore
        """
        cls._stores[name] = store_class
        logger.info(f"Registered adapter store: {name}")

    @classmethod
    def get_adapter_store(
        cls,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AdapterStore:
        """
        Get an adapter store instance.

        Args:
            name: Store name (e.g., "vespa"). If None, uses first available.
            config: Configuration for store initialization (url, port, etc.)

        Returns:
            Initialized AdapterStore instance

        Raises:
            ValueError: If no stores available or specified store not found
        """
        # Lazy discovery on first call
        if not cls._entry_points_loaded:
            cls.discover_stores()

        # Fallback: use first available store if name not specified
        if name is None:
            if not cls._stores:
                raise ValueError(
                    "No adapter stores installed. "
                    "Install cogniverse-vespa or another store package."
                )
            name = list(cls._stores.keys())[0]
            logger.info(f"No store specified, using first available: {name}")

        # Check if store exists
        if name not in cls._stores:
            available = list(cls._stores.keys())
            raise ValueError(
                f"Adapter store '{name}' not found. "
                f"Available stores: {available or 'none'}"
            )

        # Create cache key from config
        config = config or {}
        cache_key = f"{name}_{config.get('vespa_url', '')}_{config.get('vespa_port', '')}"

        # Check cache
        if cache_key in cls._store_instances:
            logger.debug(f"Returning cached adapter store: {cache_key}")
            return cls._store_instances[cache_key]

        # Create new instance
        store_class = cls._stores[name]

        # Initialize with config
        if config:
            instance = store_class(**config)
        else:
            instance = store_class()

        # Cache instance
        cls._store_instances[cache_key] = instance
        logger.info(f"Created adapter store: {name}")

        return instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        cls._stores.clear()
        cls._store_instances.clear()
        cls._entry_points_loaded = False
        cls._instance = None
