#!/usr/bin/env python3
"""Provision a tenant's backend resources without a live runtime.

Used by the tenant-provisioning WorkflowTemplate as a cold-bootstrap step that
talks directly to the data-plane backends (Vespa/Phoenix), so it does not need
the runtime API. Schema deployment is handled separately by
``deploy_json_schema.py``; this script covers the memory, telemetry and
router-tier steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def init_memory(tenant_id: str) -> None:
    """Create the tenant's Mem0 memory schema via the runtime initializer."""
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.memory_init import lazy_init_memory

    config_manager = create_default_config_manager()
    mgr = Mem0MemoryManager(tenant_id)
    lazy_init_memory(mgr, tenant_id, config_manager, auto_create_schema=True)


def init_telemetry(tenant_id: str) -> None:
    """Emit a probe span so the tenant's Phoenix project is created."""
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    tm = get_telemetry_manager()
    with tm.span("provision.probe", tenant_id=tenant_id, component="search_service"):
        pass


def init_tier(tenant_id: str, tier: str) -> str:
    """Store the tenant's semantic-router tier, the seam the admin route writes.

    The tier is validated before any store is built, so a value outside the
    router's vocabulary fails with the vocabulary and never touches a backend.
    Returns the stored tier.
    """
    from cogniverse_foundation.config.tenant_tiers import (
        set_tenant_tier,
        validate_router_tier,
    )
    from cogniverse_foundation.config.utils import create_default_config_manager

    validate_router_tier(tier)
    return set_tenant_tier(create_default_config_manager(), tenant_id, tier)


_STEPS = {"memory": init_memory, "telemetry": init_telemetry, "tier": init_tier}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant identifier")
    parser.add_argument(
        "--step", required=True, choices=sorted(_STEPS), help="Provisioning step"
    )
    parser.add_argument("--tier", help="Router tier to store; required by --step tier")
    args = parser.parse_args()
    if args.step == "tier":
        if not args.tier:
            parser.error("--step tier requires --tier")
        try:
            init_tier(args.tenant_id, args.tier)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        _STEPS[args.step](args.tenant_id)
    print(f"Provisioned {args.step} for tenant {args.tenant_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
