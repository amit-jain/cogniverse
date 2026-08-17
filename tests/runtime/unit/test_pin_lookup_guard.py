"""The lifecycle scheduler's pin lookup never reads an outage as "no pins".

``build_pin_lookup`` produces the callable ``lifespan`` hands to
``LifecycleScheduler``. If a pin-store failure returned an empty set, the
scheduler would treat genuinely pinned memories as unpinned and prune them —
data loss. The callable must RAISE on failure (tick_once then skips that
tenant's cleanup), return the exact pinned-id set on success, and honor
per-tenant quota overrides via ``PinQuotas.for_tenant``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_runtime.main import build_pin_lookup

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _manager(tenant_id="acme:acme"):
    mm = MagicMock()
    mm.tenant_id = tenant_id
    return mm


def test_outage_raises_instead_of_returning_no_pins():
    registry = MagicMock()
    quota_loader = MagicMock(side_effect=ConnectionError("pin store unreachable"))
    pin_lookup = build_pin_lookup(registry, quota_loader)

    failing_service = MagicMock()
    failing_service.list_pins.side_effect = ConnectionError("pin store unreachable")

    with (
        patch(
            "cogniverse_core.memory.pinning.PinService",
            return_value=failing_service,
        ),
    ):
        with pytest.raises(ConnectionError):
            pin_lookup(_manager())

    quota_loader.assert_called_once_with("acme:acme")


def test_success_returns_exact_pinned_id_set_with_tenant_quotas():
    registry = MagicMock()
    quota_loader = MagicMock(
        return_value={"user": 2, "tenant_admin": 4, "org_admin": -1}
    )
    pin_lookup = build_pin_lookup(registry, quota_loader)

    records = [
        SimpleNamespace(target_memory_id="m1"),
        SimpleNamespace(target_memory_id="m2"),
    ]
    service = MagicMock()
    service.list_pins.return_value = records
    tenant_quotas = MagicMock(name="tenant_quotas")

    with (
        patch(
            "cogniverse_core.memory.pinning.PinService", return_value=service
        ) as svc_cls,
        patch("cogniverse_core.memory.pinning.PinQuotas") as quotas,
    ):
        quotas.for_tenant.return_value = tenant_quotas
        mm = _manager("acme:acme")
        assert pin_lookup(mm) == {"m1", "m2"}

    quota_loader.assert_called_once_with("acme:acme")
    quotas.for_tenant.assert_called_once_with(
        "acme:acme",
        admin_overrides={"user": 2, "tenant_admin": 4, "org_admin": -1},
    )
    assert svc_cls.call_args.kwargs["quotas"] is tenant_quotas
    assert svc_cls.call_args.args == (mm, registry)
    service.list_pins.assert_called_once_with("acme:acme")


def test_manager_without_tenant_yields_empty_set():
    quota_loader = MagicMock()
    pin_lookup = build_pin_lookup(MagicMock(), quota_loader)
    anonymous = SimpleNamespace(tenant_id="")
    assert pin_lookup(anonymous) == set()
    quota_loader.assert_not_called()
