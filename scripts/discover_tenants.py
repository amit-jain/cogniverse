#!/usr/bin/env python3
"""
Tenant Discovery for Auto-Optimization

Discovers all tenants with routing configurations and outputs them
for workflow parallelization.

Outputs JSON array of tenant IDs to stdout for Argo workflow consumption.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_sdk.interfaces.config_store import ConfigScope

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def discover_tenants() -> list[str]:
    """
    Discover all tenants with routing configurations

    Returns:
        List of tenant IDs
    """
    try:
        config_manager = create_default_config_manager()

        # Get all routing configs across tenants
        configs = config_manager.store.list_all_configs(
            scope=ConfigScope.ROUTING,
        )

        # Extract unique tenant IDs
        tenant_ids = sorted(set(
            config.tenant_id for config in configs
            if config.tenant_id  # Filter out None
        ))

        logger.info(f"Discovered {len(tenant_ids)} tenants: {tenant_ids}")

        return tenant_ids

    except Exception as e:
        logger.error(f"Failed to discover tenants: {e}")
        raise


def main():
    """
    Main entry point

    Outputs JSON array to stdout for Argo workflow consumption
    """
    logger.info("🔍 Discovering tenants for auto-optimization...")

    tenant_ids = discover_tenants()

    if not tenant_ids:
        logger.error(
            "No tenants found with routing configs. Auto-optimization requires "
            "at least one tenant to be registered and configured."
        )
        return 1

    # Output JSON array to stdout for Argo
    output = json.dumps(tenant_ids)
    print(output)

    logger.info(f"✅ Tenant discovery complete: {len(tenant_ids)} tenants")

    return 0


if __name__ == "__main__":
    sys.exit(main())
