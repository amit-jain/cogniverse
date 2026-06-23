#!/usr/bin/env python3
"""Provision a tenant's backend resources without a live runtime.

Used by the tenant-provisioning WorkflowTemplate as a cold-bootstrap step that
talks directly to the data-plane backends (Vespa/Phoenix), so it does not need
the runtime API. Schema deployment is handled separately by
``deploy_json_schema.py``; this script covers the memory and telemetry steps.
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
    if not lazy_init_memory(mgr, tenant_id, config_manager, auto_create_schema=True):
        raise RuntimeError(f"Memory initialization failed for tenant '{tenant_id}'")


def init_telemetry(tenant_id: str) -> None:
    """Emit a probe span so the tenant's Phoenix project is created."""
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    tm = get_telemetry_manager()
    with tm.span("provision.probe", tenant_id=tenant_id, component="search_service"):
        pass


_STEPS = {"memory": init_memory, "telemetry": init_telemetry}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant identifier")
    parser.add_argument(
        "--step", required=True, choices=sorted(_STEPS), help="Provisioning step"
    )
    args = parser.parse_args()
    _STEPS[args.step](args.tenant_id)
    print(f"Provisioned {args.step} for tenant {args.tenant_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
