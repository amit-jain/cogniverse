#!/usr/bin/env python3
"""Checkout-side entry point for the tenant cold-bootstrap steps.

The implementation lives in ``cogniverse_runtime.provision_tenant`` so the
tenant-provisioning WorkflowTemplate can run it from the installed runtime
image, which ships no ``scripts/`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_runtime.provision_tenant import (  # noqa: E402
    deploy_schemas,
    init_memory,
    init_telemetry,
    init_tier,
    main,
    verify_schemas,
)

__all__ = [
    "deploy_schemas",
    "init_memory",
    "init_telemetry",
    "init_tier",
    "main",
    "verify_schemas",
]

if __name__ == "__main__":
    sys.exit(main())
