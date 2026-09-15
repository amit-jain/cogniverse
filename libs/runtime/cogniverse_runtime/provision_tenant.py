"""Provision a tenant's backend resources without a live runtime.

The tenant-provisioning WorkflowTemplate runs this as a cold-bootstrap step
inside the runtime image, so every step is reachable from the installed
package alone: ``python -m cogniverse_runtime.provision_tenant``.

Schemas go through ``SchemaRegistry.deploy_schema`` — the same seam
``POST /admin/profiles/{name}/deploy`` uses — so the tenant's schema is
added to the live application package alongside every other tenant's,
under its tenant-scoped name.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA_DIR = Path("configs/schemas")


def _backend_config() -> Dict[str, Any]:
    """Vespa endpoint for this step.

    ``BACKEND_URL``/``BACKEND_PORT`` are the data endpoint every other
    cogniverse process reads; ``VESPA_CONFIG_PORT`` is the config server
    the schema deploy posts to, which is not derivable from the data port
    on a cluster that remaps either.
    """
    config: Dict[str, Any] = {}
    if os.environ.get("BACKEND_URL"):
        config["url"] = os.environ["BACKEND_URL"]
    if os.environ.get("BACKEND_PORT"):
        config["port"] = int(os.environ["BACKEND_PORT"])
    if os.environ.get("VESPA_CONFIG_PORT"):
        config["config_port"] = int(os.environ["VESPA_CONFIG_PORT"])
    return config


def _resolve(tenant_id: str, profiles: List[str]) -> Tuple[Any, Any, List[str]]:
    """Return (config_manager, backend, base schema names) for ``profiles``."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    schema_names = []
    for profile_name in profiles:
        profile = config_manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )
        if profile is None:
            raise RuntimeError(
                f"Provisioning failed for tenant {tenant_id}: profile "
                f"{profile_name!r} is not configured"
            )
        schema_names.append(profile.schema_name)

    backend = BackendRegistry.get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config=_backend_config(),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMA_DIR),
    )
    return config_manager, backend, schema_names


def deploy_schemas(tenant_id: str, profiles: List[str]) -> List[str]:
    """Deploy each profile's tenant schema and return the full names."""
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

    tenant = canonical_tenant_id(tenant_id)
    _, backend, schema_names = _resolve(tenant, profiles)
    deployed = []
    for base_schema_name in schema_names:
        try:
            deployed.append(
                backend.schema_registry.deploy_schema(
                    tenant_id=tenant, base_schema_name=base_schema_name
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Provisioning schemas failed for tenant {tenant}: "
                f"{base_schema_name}: {exc}"
            ) from exc
    return deployed


def verify_schemas(tenant_id: str, profiles: List[str]) -> List[str]:
    """Confirm each profile's tenant schema is registered and queryable."""
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

    tenant = canonical_tenant_id(tenant_id)
    _, backend, schema_names = _resolve(tenant, profiles)
    verified = []
    for base_schema_name in schema_names:
        if not backend.schema_exists(schema_name=base_schema_name, tenant_id=tenant):
            raise RuntimeError(
                f"Provisioning verify failed for tenant {tenant}: "
                f"{base_schema_name} is not deployed"
            )
        verified.append(backend.get_tenant_schema_name(tenant, base_schema_name))
    return verified


def init_memory(tenant_id: str) -> None:
    """Create the tenant's Mem0 memory schema via the runtime initializer."""
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.memory_init import lazy_init_memory

    config_manager = create_default_config_manager()
    mgr = Mem0MemoryManager(tenant_id)
    lazy_init_memory(mgr, tenant_id, config_manager, auto_create_schema=True)


def init_telemetry(tenant_id: str) -> None:
    """Export a probe span so the tenant's Phoenix project exists.

    The export is required: a collector that never received the probe has
    not created the project, and a step that exits 0 anyway would report a
    tenant as observable when none of its spans have anywhere to land.
    """
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults

    tenant = canonical_tenant_id(tenant_id)
    # The collector the step was pointed at, the same override the runtime
    # entrypoint applies. Without it the probe goes to the default local
    # collector and the tenant's project is created nowhere.
    manager = get_telemetry_manager(
        otlp_endpoint=resolve_library_env_defaults()["telemetry_otlp_endpoint"]
    )

    async def _emit() -> None:
        async with manager.required_span("provision.probe", tenant_id=tenant):
            pass

    try:
        asyncio.run(_emit())
    except Exception as exc:
        raise RuntimeError(
            f"Provisioning telemetry failed for tenant {tenant}: {exc}"
        ) from exc


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


_PROFILE_STEPS = {"schemas": deploy_schemas, "verify": verify_schemas}
_TENANT_STEPS = {"memory": init_memory, "telemetry": init_telemetry}
_STEPS = sorted({*_PROFILE_STEPS, *_TENANT_STEPS, "tier"})


def main() -> int:
    # Same wiring every other runtime entrypoint applies: the step's
    # TELEMETRY_OTLP_ENDPOINT, MinIO credentials and embed URLs reach the
    # libraries only through this call. Without it the telemetry probe goes
    # to the default localhost collector and the tenant's project is never
    # created on the one the workflow named.
    from cogniverse_runtime.entrypoint_env import (
        configure_runtime_library_defaults,
        resolve_library_env_defaults,
    )

    configure_runtime_library_defaults(resolve_library_env_defaults())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant identifier")
    parser.add_argument(
        "--step", required=True, choices=_STEPS, help="Provisioning step"
    )
    parser.add_argument("--tier", help="Router tier to store; required by --step tier")
    parser.add_argument(
        "--profiles",
        help="Comma-separated backend profiles; required by --step schemas/verify",
    )
    args = parser.parse_args()

    if args.step in _PROFILE_STEPS:
        if not args.profiles:
            parser.error(f"--step {args.step} requires --profiles")
        profiles = [name for name in args.profiles.split(",") if name.strip()]
        if not profiles:
            parser.error(f"--step {args.step} requires --profiles")
        try:
            names = _PROFILE_STEPS[args.step](args.tenant_id, profiles)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Provisioned {args.step} for tenant {args.tenant_id}: {', '.join(names)}"
        )
        return 0

    if args.step == "tier":
        if not args.tier:
            parser.error("--step tier requires --tier")
        try:
            init_tier(args.tenant_id, args.tier)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        try:
            _TENANT_STEPS[args.step](args.tenant_id)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    print(f"Provisioned {args.step} for tenant {args.tenant_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
