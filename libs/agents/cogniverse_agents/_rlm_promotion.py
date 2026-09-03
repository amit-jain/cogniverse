"""RLM-promotion helper for ``orchestrator_agent``.

Promotion knobs are module state set once at process start via
``configure_rlm_promotion`` (the runtime entrypoint resolves
``COGNIVERSE_ORCH_RLM_PROMOTION`` / ``COGNIVERSE_ORCH_RLM_PROMOTION_FRACTION``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_RLM_PROMOTION_DEFAULT_FRACTION = 0.75
_RLM_PROMOTION_DEFAULT_THRESHOLD = 50_000

_promotion_enabled: bool = True
_promotion_fraction: float = _RLM_PROMOTION_DEFAULT_FRACTION


def configure_rlm_promotion(*, enabled: bool, fraction: float) -> None:
    """Set the promotion knobs for this process."""
    global _promotion_enabled, _promotion_fraction
    _promotion_enabled = enabled
    _promotion_fraction = fraction


_RLM_PROMOTABLE_AGENTS = frozenset(
    {
        "search_agent",
        "deep_research_agent",
        "detailed_report_agent",
        "coding_agent",
    }
)


def _projected_payload_chars(agent_input: Dict[str, Any]) -> int:
    """Cheap upper bound on prompt size — sums stringified value lengths.

    Over-counts (ignores tokenisation) but the threshold is a coarse switch
    not a hard budget; false-positive promotion just exercises RLM
    unnecessarily without breaking results.
    """
    total = 0
    for v in agent_input.values():
        try:
            total += len(str(v))
        except Exception:
            continue
    return total


def maybe_promote_to_rlm(agent_name: str, agent_input: Dict[str, Any]) -> None:
    """Stamp ``agent_input["rlm"]`` to enable RLM when payload is large.

    Idempotent: if the caller already supplied an ``rlm`` field (any value,
    including ``None`` for explicit opt-out), this is a no-op. Disabled
    entirely via ``configure_rlm_promotion(enabled=False, ...)``.
    """
    if not _promotion_enabled:
        return

    canonical = agent_name if agent_name.endswith("_agent") else f"{agent_name}_agent"
    if canonical not in _RLM_PROMOTABLE_AGENTS:
        return

    if "rlm" in agent_input:
        return

    cutoff = int(_RLM_PROMOTION_DEFAULT_THRESHOLD * _promotion_fraction)

    projected = _projected_payload_chars(agent_input)
    if projected < cutoff:
        return

    agent_input["rlm"] = {
        "enabled": True,
        "auto_detect": True,
        "context_threshold": _RLM_PROMOTION_DEFAULT_THRESHOLD,
    }
    logger.info(
        "Orchestrator promoted %s to RLM (projected_chars=%d, cutoff=%d)",
        canonical,
        projected,
        cutoff,
    )
