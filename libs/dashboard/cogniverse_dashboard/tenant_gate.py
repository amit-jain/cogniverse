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
