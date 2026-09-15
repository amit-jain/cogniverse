def canonicalize_tenant_input(raw: str) -> str:
    """Canonical ``org:tenant`` form for a sidebar tenant entry.

    Every span/config/memory namespace is keyed by the canonical form; a
    simple-form entry ("acme") passes the registration gate (the runtime
    canonicalizes server-side) but would make every tab read the empty
    raw-form namespace. Malformed input is returned unchanged — the app
    shell's registration gate rejects it with a visible error.
    """
    if not raw:
        return raw
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

    try:
        return canonical_tenant_id(raw)
    except ValueError:
        return raw


def tenant_project_name(telemetry_manager, tenant_id: str) -> str:
    """Phoenix project holding the runtime's spans for ``tenant_id``.

    GatewayAgent, ProfileSelectionAgent and OrchestratorAgent open their
    spans with ``TelemetryManager.span(..., tenant_id=...)`` and no project
    override, so the spans land in the manager's bare tenant project.
    Readers derive the name from the same config object and the same
    canonical tenant form, or they query a project nothing writes to.
    """
    return telemetry_manager.config.get_project_name(
        canonicalize_tenant_input(tenant_id)
    )
