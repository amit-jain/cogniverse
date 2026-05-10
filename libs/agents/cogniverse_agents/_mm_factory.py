"""Helpers for the per-tenant Mem0MemoryManager factory pattern shared by
the knowledge agents."""

from __future__ import annotations

from typing import Any, Callable, Optional


def default_mm_factory() -> Callable[[str], Any]:
    """Production factory: ``lambda tid: Mem0MemoryManager(tenant_id=tid)``."""
    from cogniverse_core.memory.manager import Mem0MemoryManager

    return lambda tid: Mem0MemoryManager(tenant_id=tid)


def make_mm_factory(
    factory: Optional[Callable[[str], Any]],
) -> Callable[[str], Any]:
    """Eager-init the factory at construction; production callers pass None."""
    return factory if factory is not None else default_mm_factory()


def require_tenant_id(input_obj: Any, deps_obj: Any, agent_name: str) -> str:
    """Resolve tenant_id from input or deps; raise ValueError when missing."""
    tenant_id = getattr(input_obj, "tenant_id", None) or getattr(
        deps_obj, "tenant_id", None
    )
    if not tenant_id:
        raise ValueError(f"{agent_name}: no tenant_id on input or deps")
    return tenant_id
