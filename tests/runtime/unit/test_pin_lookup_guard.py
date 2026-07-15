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
    pin_lookup = build_pin_lookup(registry)

    failing_service = MagicMock()
    failing_service.list_pins.side_effect = ConnectionError("pin store unreachable")

    with (
        patch(
            "cogniverse_core.memory.pinning.PinService",
            return_value=failing_service,
        ),
        patch("cogniverse_core.memory.pinning.PinQuotas") as quotas,
    ):
        quotas.for_tenant.return_value = MagicMock()
        with pytest.raises(ConnectionError):
            pin_lookup(_manager())


def test_success_returns_exact_pinned_id_set_with_tenant_quotas():
    registry = MagicMock()
    pin_lookup = build_pin_lookup(registry)

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

    quotas.for_tenant.assert_called_once_with("acme:acme")
    assert svc_cls.call_args.kwargs["quotas"] is tenant_quotas
    assert svc_cls.call_args.args == (mm, registry)
    service.list_pins.assert_called_once_with("acme:acme")


def test_manager_without_tenant_yields_empty_set():
    pin_lookup = build_pin_lookup(MagicMock())
    anonymous = SimpleNamespace(tenant_id="")
    assert pin_lookup(anonymous) == set()
