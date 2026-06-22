"""
Registry for cache backend plugins
"""

import dataclasses
from typing import Dict, Type

from .base import CacheBackend


class CacheBackendRegistry:
    """Registry for cache backend plugins"""

    _backends: Dict[str, Type[CacheBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[CacheBackend]):
        """Register a new cache backend"""
        cls._backends[name] = backend_class

    @classmethod
    def create(cls, config: dict) -> CacheBackend:
        """Create backend instance from config.

        Each backend advertises its config dataclass via ``CONFIG_CLASS``;
        the dict is filtered to that dataclass's fields so layered config
        with shared/extra keys (``default_ttl``, sibling-backend keys) does
        not raise ``TypeError``.
        """
        backend_type = config.get("backend_type")
        backend_class = cls._backends.get(backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend type: {backend_type}")

        config_class = getattr(backend_class, "CONFIG_CLASS", None)
        if config_class is None:
            raise ValueError(
                f"Backend '{backend_type}' does not declare a CONFIG_CLASS"
            )

        field_names = {f.name for f in dataclasses.fields(config_class)}
        kwargs = {k: v for k, v in config.items() if k in field_names}
        return backend_class(config_class(**kwargs))

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend types"""
        return list(cls._backends.keys())
