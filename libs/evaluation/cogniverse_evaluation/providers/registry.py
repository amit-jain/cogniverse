"""Evaluation provider registry — entry-point auto-discovery + default-provider state.

A thin subclass of :class:`cogniverse_foundation.registry.EntryPointRegistry`.
The base handles discovery, manual registration, conflict detection,
tenant-scoped caching, and lifecycle-style initialization
(``klass()`` + ``.initialize(config)``). This module adds the
evaluation-specific concept of a "default provider" — a singleton an
evaluator can fall back on when no explicit name is passed — plus the
backward-compatible module-level helpers (``get_evaluation_provider``,
``register_evaluation_provider``, ``reset_evaluation_provider``,
``set_evaluation_provider``, ``get_evaluation_registry``).

Implementations register via the ``cogniverse.evaluation.providers``
entry-point group::

    [project.entry-points."cogniverse.evaluation.providers"]
    phoenix = "cogniverse_telemetry_phoenix.evaluation:PhoenixEvaluationProvider"
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Optional, Type

from cogniverse_evaluation.providers.base import EvaluationProvider
from cogniverse_foundation.registry import EntryPointRegistry

logger = logging.getLogger(__name__)


class EvaluationRegistry(EntryPointRegistry[EvaluationProvider]):
    """Plugin registry for ``EvaluationProvider`` implementations.

    Providers are tenant-scoped: each ``get()`` call returns a per-tenant
    cached instance, constructed via ``klass()`` then handed the merged
    ``{tenant_id, **config}`` dict via ``.initialize(...)``.
    """

    _entry_point_group = "cogniverse.evaluation.providers"
    _label = "evaluation provider"
    _tenant_scoped = True

    _default_provider: ClassVar[Optional[EvaluationProvider]] = None

    @classmethod
    def set_default_provider(cls, provider: EvaluationProvider) -> None:
        """Pin a pre-initialized provider as the fallback for ``get_default_provider()``."""
        cls._default_provider = provider
        logger.info("Set default evaluation provider: %s", type(provider).__name__)

    @classmethod
    def get_default_provider(cls) -> EvaluationProvider:
        """Return the pinned default provider, lazily auto-initializing one if not set.

        Auto-init uses :class:`cogniverse_core.common.tenant_utils.SYSTEM_TENANT_ID`
        — the only tenant id that makes sense for a process-wide default.
        """
        if cls._default_provider is None:
            if not cls._entry_points_loaded:
                cls.discover()

            if cls._registry_classes:
                from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

                logger.info("Auto-initializing default evaluation provider")
                cls._default_provider = cls.get(tenant_id=SYSTEM_TENANT_ID)
            else:
                raise ValueError(
                    "No default evaluation provider set. "
                    "Call set_evaluation_provider() first or install a provider package."
                )

        return cls._default_provider

    @classmethod
    def clear_cache(cls) -> None:
        """Evict cached instances AND drop the pinned default."""
        super().clear_cache()
        cls._default_provider = None


# Module-level singleton + convenience helpers.
# These are the public API the rest of cogniverse_evaluation imports
# (see ``evaluators/base.py``, ``core/solvers.py``, etc.).

_evaluation_registry = EvaluationRegistry()


def get_evaluation_registry() -> EvaluationRegistry:
    """Get the global ``EvaluationRegistry`` instance."""
    return _evaluation_registry


def get_evaluation_provider(
    tenant_id: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> EvaluationProvider:
    """Convenience: resolve an evaluation provider, defaulting tenant to system.

    When ``tenant_id`` is omitted, lookup runs under the cluster-wide
    ``SYSTEM_TENANT_ID``. This keeps framework-level callers (that only
    read static provider metadata such as the evaluator base class) from
    having to fabricate a tenant id; callers doing per-tenant work must
    still pass their tenant explicitly.
    """
    if tenant_id is None:
        from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

        tenant_id = SYSTEM_TENANT_ID
    return _evaluation_registry.get(name=name, tenant_id=tenant_id, config=config)


def set_evaluation_provider(provider: EvaluationProvider) -> None:
    """Convenience: pin a default evaluation provider."""
    _evaluation_registry.set_default_provider(provider)


def register_evaluation_provider(
    name: str, provider_class: Type[EvaluationProvider]
) -> None:
    """Manually register an evaluation provider (for testing)."""
    _evaluation_registry.register(name, provider_class)


def reset_evaluation_provider() -> None:
    """Reset evaluation provider cache and default provider."""
    _evaluation_registry.clear_cache()
    logger.info("Reset evaluation provider")
