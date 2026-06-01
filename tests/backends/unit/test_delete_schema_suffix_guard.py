"""delete_schema's cross-tenant guard must use the canonical tenant suffix.

The target name is built from the canonicalized tenant (acme -> acme:acme ->
suffix _acme_acme), but the guard compared against the raw tenant_id's suffix
(_acme), which is only a substring — so a wrong-tenant target ending in _acme
slipped past the defensive check.
"""

from __future__ import annotations

import pytest

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager


def _bare_manager() -> VespaSchemaManager:
    mgr = object.__new__(VespaSchemaManager)
    mgr._schema_registry = object()  # truthy — past the registry guard
    mgr._PROTECTED_SCHEMAS = frozenset()
    return mgr


def test_cross_tenant_target_rejected_by_canonical_suffix():
    mgr = _bare_manager()
    # A target for a DIFFERENT tenant that still ends in the raw "_acme".
    mgr.get_tenant_schema_name = lambda tenant_id, base: "video_other_acme"

    with pytest.raises(ValueError, match="does not carry the expected"):
        mgr.delete_schema("acme", "video")
