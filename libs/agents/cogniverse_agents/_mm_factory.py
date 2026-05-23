"""Helpers for the per-tenant Mem0MemoryManager factory pattern shared
by the knowledge agents (federated_query, temporal_reasoning,
cross_tenant_comparison, knowledge_summarization, audit_explanation).

Two helpers live here:

* :func:`make_mm_factory` — resolves the agent's optional
  ``memory_manager_factory`` constructor argument: returns it as-is
  when non-None, falls back to a Mem0MemoryManager-per-tenant lambda
  otherwise. One function (previously two — collapsed
  ``default_mm_factory`` into the same call).
* :func:`tenant_id_from_input_or_deps` — pulls ``tenant_id`` off
  either the agent's input or its deps. Named distinctly from
  :func:`cogniverse_core.common.tenant_utils.require_tenant_id` (a
  different shape — that one validates a single value) so the two
  don't collide at the import boundary.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def make_mm_factory(
    factory: Optional[Callable[[str], Any]] = None,
) -> Callable[[str], Any]:
    """Resolve the memory-manager factory at agent construction time.

    Production callers pass ``None`` and accept the default — a lambda
    that returns ``Mem0MemoryManager(tenant_id=tid)`` per call. Tests
    pass an explicit factory (typically a MagicMock-returning lambda)
    to bypass real Mem0 / Vespa setup.
    """
    if factory is not None:
        return factory
    from cogniverse_core.memory.manager import Mem0MemoryManager

    return lambda tid: Mem0MemoryManager(tenant_id=tid)


def tenant_id_from_input_or_deps(input_obj: Any, deps_obj: Any, agent_name: str) -> str:
    """Resolve ``tenant_id`` from the agent's input first, then its
    deps. Raises ``ValueError`` when neither has it. Named distinctly
    from ``cogniverse_core.common.tenant_utils.require_tenant_id`` to
    avoid an import-path foot-gun (that one validates a single string
    value; this one looks the attribute up across two objects).
    """
    tenant_id = getattr(input_obj, "tenant_id", None) or getattr(
        deps_obj, "tenant_id", None
    )
    if not tenant_id:
        raise ValueError(f"{agent_name}: no tenant_id on input or deps")
    return tenant_id
