"""Per-tenant signature variants.

Today every tenant compiles against the *same* DSPy signature for an agent
(prompts and demos vary, but the input/output shape is fixed). Some
tenants have unusual data — e.g. a legal-knowledge tenant whose queries
always carry a ``jurisdiction`` field — and benefit from a variant
signature that exposes that field.

This module ships **Option B** from the plan: a small *named-variant
registry* per agent. Each variant has a stable id (``"default"``,
``"with_jurisdiction"``, ``"with_temporal_qualifiers"``, …) and a
description; tenant config picks one. The artefact manager keys prompts /
demos / experiments by ``(tenant_id, agent_type, variant_id)`` so each
variant has its own compiled artefacts.

We deliberately do NOT take Option A (free-form per-tenant signatures);
the registry is the single source of truth for which variants exist, so
upgrades and tests stay tractable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_VARIANT_ID = "default"


@dataclass(frozen=True)
class SignatureVariant:
    """A named signature variant available for one agent.

    Variants are intentionally schemaless from the registry's perspective —
    the agent itself owns the DSPy signature class. The registry only
    tracks which variant ids are valid for an agent; the actual class
    selection happens at agent construction time via a lookup table the
    agent maintains.
    """

    agent_type: str
    variant_id: str
    description: str = ""


class SignatureVariantRegistry:
    """In-memory registry of per-agent signature variants.

    Each agent has at minimum a ``"default"`` variant (auto-registered on
    first lookup). Operators / tenant admins register additional variants
    at runtime via ``register``. Tenant selection is read from
    :class:`TenantConfig.metadata['signature_variants'][agent_type]` —
    see ``selected_for_tenant``.
    """

    def __init__(self) -> None:
        # agent_type -> {variant_id -> SignatureVariant}
        self._variants: Dict[str, Dict[str, SignatureVariant]] = {}

    def register(
        self,
        agent_type: str,
        variant_id: str,
        description: str = "",
        *,
        replace: bool = False,
    ) -> SignatureVariant:
        """Register a new variant for an agent.

        Idempotent for re-registration of identical (agent_type, variant_id);
        raises when ``replace=False`` and the variant exists with a
        different description.
        """
        if not variant_id:
            raise ValueError("variant_id must be a non-empty string")
        per_agent = self._variants.setdefault(agent_type, {})
        existing = per_agent.get(variant_id)
        new = SignatureVariant(
            agent_type=agent_type,
            variant_id=variant_id,
            description=description,
        )
        if existing is not None and not replace:
            if existing == new:
                return existing
            raise ValueError(
                f"variant id={variant_id!r} for agent={agent_type!r} already "
                "registered with a different definition; pass replace=True"
            )
        per_agent[variant_id] = new
        return new

    def list_for_agent(self, agent_type: str) -> List[SignatureVariant]:
        """List variants registered for an agent (always includes default)."""
        per_agent = self._variants.setdefault(agent_type, {})
        per_agent.setdefault(
            DEFAULT_VARIANT_ID,
            SignatureVariant(
                agent_type=agent_type,
                variant_id=DEFAULT_VARIANT_ID,
                description="Cogniverse-shipped baseline signature",
            ),
        )
        return sorted(per_agent.values(), key=lambda v: v.variant_id)

    def is_registered(self, agent_type: str, variant_id: str) -> bool:
        return variant_id in self._variants.get(agent_type, {})

    def selected_for_tenant(
        self,
        tenant_config: Optional[object],
        agent_type: str,
    ) -> str:
        """Resolve the variant id this tenant has selected for ``agent_type``.

        Reads ``TenantConfig.metadata['signature_variants'][agent_type]``;
        falls back to ``"default"`` when no override is present, when the
        config is missing, or when the requested variant is not registered
        (with a warning — operators want to know about typos).
        """
        meta = getattr(tenant_config, "metadata", None) if tenant_config else None
        if not isinstance(meta, dict):
            return DEFAULT_VARIANT_ID
        per_agent = meta.get("signature_variants") or {}
        if not isinstance(per_agent, dict):
            return DEFAULT_VARIANT_ID
        explicit = per_agent.get(agent_type)
        # No override (or the default itself) → the baseline default, silently.
        # Only an EXPLICIT non-default variant that isn't registered is an
        # operator typo worth warning about.
        if not explicit or explicit == DEFAULT_VARIANT_ID:
            return DEFAULT_VARIANT_ID
        if not self.is_registered(agent_type, explicit):
            logger.warning(
                "Tenant selected variant %r for agent %r but it is not "
                "registered; falling back to %r",
                explicit,
                agent_type,
                DEFAULT_VARIANT_ID,
            )
            return DEFAULT_VARIANT_ID
        return explicit


def variant_qualified_agent_key(agent_type: str, variant_id: str) -> str:
    """Stable, dataset-name-safe key for ``(agent_type, variant_id)``.

    Default variant returns the bare agent_type so existing artefacts (which
    were saved before the variant story existed) keep working — they map
    naturally onto the default variant. Non-default variants get the
    ``::variant=`` suffix.
    """
    if not variant_id or variant_id == DEFAULT_VARIANT_ID:
        return agent_type
    safe = variant_id.replace(":", "_").replace("/", "_")
    return f"{agent_type}::variant={safe}"
