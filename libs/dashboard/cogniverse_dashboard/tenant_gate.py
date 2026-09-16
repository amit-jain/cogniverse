"""Tenant gate decision for the dashboard.

Kept out of ``app.py`` so it is importable (and testable) without executing
``app.py``'s top-level Streamlit UI body.

The gate blocks the whole dashboard, so it distinguishes a runtime that
answered and does not know the tenant from a runtime that could not be
reached. Only the former is a verdict about the tenant; the latter is a
statement about the probe, and treating it as a verdict collapses a working
session on one slow request.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenantProbe:
    """Outcome of asking the runtime about a tenant.

    ``reachable`` is whether the runtime answered at all; ``registered`` is
    only meaningful when it did.
    """

    reachable: bool
    registered: bool
    detail: str = ""


@dataclass(frozen=True)
class GateDecision:
    allow: bool
    error: str = ""
    warning: str = ""


def decide_tenant_gate(
    probe: TenantProbe, tenant_id: str, *, previously_validated: bool
) -> GateDecision:
    """Whether to render the dashboard for ``tenant_id``.

    A tenant the runtime has confirmed before stays usable across a probe
    failure, because the alternative is tearing down a working session for a
    timeout that says nothing about the tenant.
    """
    if probe.reachable and probe.registered:
        return GateDecision(allow=True)

    if probe.reachable and not probe.registered:
        return GateDecision(
            allow=False,
            error=(
                f"Tenant **{tenant_id}** cannot be used: {probe.detail}. "
                "Register the tenant first via `POST /admin/tenants` or pick a "
                "registered tenant in the sidebar."
            ),
        )

    if previously_validated:
        return GateDecision(
            allow=True,
            warning=(
                f"Could not re-check tenant **{tenant_id}** ({probe.detail}). "
                "Continuing with the last successful validation."
            ),
        )

    return GateDecision(
        allow=False,
        error=(
            f"Tenant **{tenant_id}** cannot be validated: {probe.detail}. "
            "The runtime has not confirmed this tenant, so the dashboard "
            "cannot scope its data safely."
        ),
    )


# Session-state entries that hold one tenant's data: search results and the
# span ids they carry, chat and conversation turns, annotation queues,
# telemetry frames and generated datasets.
TENANT_SCOPED_SESSION_KEYS: tuple[str, ...] = (
    "current_search_results",
    "conversation_history",
    "chat_messages",
    "search_annotations",
    "search_spans",
    "orch_spans",
    "processing_results",
    "golden_dataset",
    "golden_dataset_size",
    "annotation_requests",
    "annotation_count",
    "auto_annotations",
    "approval_agent",
    "approval_agent_tenant_id",
    "approval_storage",
    "approved_items",
    "pending_items",
    "rejected_items",
    "last_generated_batch",
    "synthetic_data_result",
    "embedding_atlas_file",
    "last_optimize_run",
    "optimization_requests",
    "_root_cause_analysis",
)

# The subset the app body appends to without a presence check, so the reset
# leaves an empty list rather than a missing key.
_LIST_VALUED_SESSION_KEYS = frozenset(
    {
        "conversation_history",
        "chat_messages",
        "search_annotations",
        "processing_results",
    }
)


def reset_tenant_scoped_state(session_state) -> None:
    """Drop every tenant-scoped entry and start a new telemetry session.

    Called whenever the sidebar's active tenant changes. Without it the
    previous tenant's search results, chat turns and span ids stay in
    session state: the new tenant's page renders them, and Save Annotation
    writes the previous tenant's span id into the new tenant's project.
    """
    import uuid

    for key in TENANT_SCOPED_SESSION_KEYS:
        if key in _LIST_VALUED_SESSION_KEYS:
            session_state[key] = []
        else:
            session_state.pop(key, None)
    session_state["session_id"] = str(uuid.uuid4())


def require_result_tenant(result: dict, tenant_id: str) -> dict:
    """Return ``result`` when it was produced for ``tenant_id``.

    Search results carry the tenant they were fetched for. Rendering one
    under a different tenant, or attaching its span id to a different
    tenant's Phoenix project, is a cross-tenant write, so it raises instead.
    """
    owner = result.get("tenant_id")
    if owner != tenant_id:
        raise ValueError(f"Search result belongs to {owner}, not {tenant_id}")
    return result
