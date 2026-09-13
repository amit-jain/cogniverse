"""The provisioning script's tier step against the real config store.

A tenant provisioned by the Argo template gets its router tier from the same
seam the admin route writes through, so what the script stores is exactly
what the runtime's tier reader answers.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

from cogniverse_foundation.config.tenant_tiers import read_tenant_tier
from cogniverse_foundation.config.unified_config import DEFAULT_ROUTER_TIER

pytestmark = pytest.mark.integration

_SCRIPT = Path(__file__).parents[3] / "scripts" / "provision_tenant.py"


def _load():
    spec = importlib.util.spec_from_file_location("provision_tenant", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_tier_step_stores_what_the_reader_answers(config_manager, monkeypatch):
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: config_manager,
    )
    tenant = f"prov{uuid.uuid4().hex[:8]}:production"
    untouched = f"prov{uuid.uuid4().hex[:8]}:production"
    pt = _load()

    assert pt.init_tier(tenant, "pro") == "pro"
    assert read_tenant_tier(config_manager, tenant) == "pro"

    assert pt.init_tier(tenant, "free") == "free"
    assert read_tenant_tier(config_manager, tenant) == "free"

    assert read_tenant_tier(config_manager, untouched) == DEFAULT_ROUTER_TIER
