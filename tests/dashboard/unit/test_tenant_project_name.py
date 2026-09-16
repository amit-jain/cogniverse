"""``tenant_project_name`` names the project the span producers write into.

``TelemetryManager`` formats ``TelemetryConfig.tenant_project_template`` with
the canonical tenant id, so a reader that skips either half names a project
nothing writes to.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_dashboard.utils import tenant_project_name
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import TelemetryConfig

# A template no shipped config uses, so a reader that formats
# ``cogniverse-{tenant_id}`` itself names a different project than the producer.
_TEMPLATE = "spans-{tenant_id}-v2"
_TENANT = "acme:prod"


def _manager(template: str | None = None) -> SimpleNamespace:
    config = (
        TelemetryConfig()
        if template is None
        else TelemetryConfig(tenant_project_template=template)
    )
    return SimpleNamespace(config=config)


def _producer_project(tenant: str = _TENANT, template: str = _TEMPLATE) -> str:
    """The project ``TelemetryManager.span`` writes into for ``tenant``."""
    return TelemetryConfig(tenant_project_template=template).get_project_name(
        canonical_tenant_id(tenant)
    )


# --- the tenant form the helper derives from -------------------------------


@pytest.mark.parametrize(
    "raw, project",
    [
        ("acme", "cogniverse-acme:acme"),
        ("ACME", "cogniverse-ACME:ACME"),
        ("acme:prod", "cogniverse-acme:prod"),
        ("ACME:Prod", "cogniverse-ACME:Prod"),
        ("__system__", "cogniverse-__system__"),
        # Malformed input has no canonical form; the registration gate rejects
        # it, so the helper passes it through instead of raising mid-render.
        ("a:b:c", "cogniverse-a:b:c"),
    ],
)
def test_project_name_for_each_tenant_form(raw: str, project: str) -> None:
    assert tenant_project_name(_manager(), raw) == project


def test_simple_form_tenant_names_the_canonical_project() -> None:
    manager = _manager()
    # The producer canonicalizes at the request boundary, so the reader must
    # too: the raw form names a project nothing writes to.
    assert tenant_project_name(manager, "acme") == manager.config.get_project_name(
        canonical_tenant_id("acme")
    )
    assert manager.config.get_project_name("acme") == "cogniverse-acme"
    assert tenant_project_name(manager, "acme") == "cogniverse-acme:acme"


def test_project_name_formats_the_configured_template() -> None:
    assert tenant_project_name(_manager(_TEMPLATE), "acme") == "spans-acme:acme-v2"
    assert tenant_project_name(_manager(_TEMPLATE), _TENANT) == "spans-acme:prod-v2"
