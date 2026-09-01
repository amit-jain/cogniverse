"""The tenant gate blocks the whole dashboard, so it must separate a verdict
about the tenant from a failure of the probe that asks about it.

A slow or unreachable runtime says nothing about whether a tenant is valid.
Treating it as "tenant cannot be used" collapsed a working session -- the
dashboard st.stop()s before rendering any tab, so the entire UI disappears --
whenever one probe exceeded its budget, which happens when the runtime is
busy serving the query the user just issued.
"""

from __future__ import annotations

from cogniverse_dashboard.tenant_gate import (
    TenantProbe,
    decide_tenant_gate,
)

TENANT = "flywheel_org:production"


def test_registered_tenant_renders_without_notice():
    decision = decide_tenant_gate(
        TenantProbe(reachable=True, registered=True),
        TENANT,
        previously_validated=False,
    )

    assert (decision.allow, decision.error, decision.warning) == (True, "", "")


def test_runtime_says_not_registered_blocks_even_if_validated_before():
    """A 200-answering runtime that returns 404 is a verdict, not a hiccup:
    the tenant was deleted or renamed, and stale success must not override it."""
    decision = decide_tenant_gate(
        TenantProbe(
            reachable=True,
            registered=False,
            detail="tenant 'flywheel_org:production' is not registered",
        ),
        TENANT,
        previously_validated=True,
    )

    assert decision.allow is False
    assert decision.warning == ""
    assert decision.error == (
        "Tenant **flywheel_org:production** cannot be used: tenant "
        "'flywheel_org:production' is not registered. Register the tenant "
        "first via `POST /admin/tenants` or pick a registered tenant in the "
        "sidebar."
    )


def test_unreachable_runtime_keeps_a_previously_validated_tenant_usable():
    """The regression this gate exists to prevent: a probe timeout used to
    tear down every tab mid-session."""
    decision = decide_tenant_gate(
        TenantProbe(
            reachable=False,
            registered=False,
            detail="runtime unreachable at http://runtime:8000: timed out",
        ),
        TENANT,
        previously_validated=True,
    )

    assert decision.allow is True
    assert decision.error == ""
    assert decision.warning == (
        "Could not re-check tenant **flywheel_org:production** (runtime "
        "unreachable at http://runtime:8000: timed out). Continuing with the "
        "last successful validation."
    )


def test_unreachable_runtime_blocks_a_tenant_never_confirmed():
    """Without a prior confirmation there is nothing to fall back on, so the
    gate must refuse rather than scope data to an unverified tenant."""
    decision = decide_tenant_gate(
        TenantProbe(
            reachable=False,
            registered=False,
            detail="runtime unreachable at http://runtime:8000: timed out",
        ),
        TENANT,
        previously_validated=False,
    )

    assert decision.allow is False
    assert decision.warning == ""
    assert decision.error == (
        "Tenant **flywheel_org:production** cannot be validated: runtime "
        "unreachable at http://runtime:8000: timed out. The runtime has not "
        "confirmed this tenant, so the dashboard cannot scope its data safely."
    )


def test_non_404_http_error_is_treated_as_a_probe_failure_not_a_verdict():
    """A 500 or 503 from the runtime describes the runtime, not the tenant, so
    it must behave like unreachability rather than like a 404."""
    unreachable = TenantProbe(
        reachable=False,
        registered=False,
        detail="HTTP 503 from /admin/tenants/flywheel_org:production",
    )

    assert (
        decide_tenant_gate(unreachable, TENANT, previously_validated=True).allow is True
    )
    assert (
        decide_tenant_gate(unreachable, TENANT, previously_validated=False).allow
        is False
    )
