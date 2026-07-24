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
