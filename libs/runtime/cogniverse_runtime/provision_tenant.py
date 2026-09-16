"""Provision a tenant's backend resources without a live runtime.

The tenant-provisioning WorkflowTemplate runs this as a cold-bootstrap step
inside the runtime image, so every step is reachable from the installed
package alone: ``python -m cogniverse_runtime.provision_tenant``.

Profiles resolve through the tenant's merged backend catalog — the shipped
``configs/config.json`` profiles with the tenant's stored overrides on top —
so a tenant registered a minute ago provisions on the cluster catalog.
Schemas go through ``SchemaRegistry.deploy_schemas`` as one application
package, so a tenant's profiles land together or not at all, under their
tenant-scoped names beside every other tenant's.
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


def _config_manager(tenant_id: str):
    """The tenant store this step reads, reported with the tenant on failure.

    The endpoint arrives as ``BACKEND_URL``/``BACKEND_PORT``; a step the
    workflow launched without them cannot reach any configuration at all,
    which is a provisioning failure rather than a crash.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager

    try:
        return create_default_config_manager()
    except ValueError as exc:
        raise RuntimeError(
            f"Provisioning failed for tenant {tenant_id}: {exc}"
        ) from exc


def resolve_schema_names(
    config_manager, tenant_id: str, profiles: List[str]
) -> List[str]:
    """Map each profile to its schema name through the tenant's catalog.

    The catalog is the cluster's shipped profiles merged with the tenant's
    stored overrides, so provisioning does not require the tenant to own a
    backend row before its first schema exists.
    """
    from cogniverse_foundation.config.utils import get_config

    try:
        catalog = (
            get_config(tenant_id, config_manager).get("backend", {}).get("profiles", {})
        )
    except Exception as exc:
        raise RuntimeError(
            f"Provisioning failed for tenant {tenant_id}: "
            f"the profile catalog is unreadable: {exc}"
        ) from exc
    schema_names = []
    for profile_name in profiles:
        profile = catalog.get(profile_name)
        if profile is None:
            raise RuntimeError(
                f"Provisioning failed for tenant {tenant_id}: profile "
                f"{profile_name!r} is not configured. "
                f"Configured profiles: {sorted(catalog)}"
            )
        schema_names.append(profile["schema_name"])
    return schema_names


def _resolve(tenant_id: str, profiles: List[str]) -> Tuple[Any, Any, List[str]]:
    """Return (config_manager, backend, base schema names) for ``profiles``."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    config_manager = _config_manager(tenant_id)
    schema_names = resolve_schema_names(config_manager, tenant_id, profiles)
    backend = BackendRegistry.get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config=_backend_config(),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMA_DIR),
    )
    return config_manager, backend, schema_names


def deploy_schemas(tenant_id: str, profiles: List[str]) -> List[str]:
    """Deploy the tenant's profile schemas as one package; return full names."""
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

    tenant = canonical_tenant_id(tenant_id)
    _, backend, schema_names = _resolve(tenant, profiles)
    try:
        return backend.schema_registry.deploy_schemas(
            tenant_id=tenant, base_schema_names=schema_names
        )
    except Exception as exc:
        raise RuntimeError(
            f"Provisioning schemas failed for tenant {tenant}: "
            f"{', '.join(schema_names)}: {exc}"
        ) from exc


def verify_schemas(tenant_id: str, profiles: List[str]) -> List[str]:
    """Confirm each profile's tenant schema is registered and queryable.

    Registration is a registry row; queryable is a real YQL request against
    the tenant schema, so a row whose schema never reached the live
    application package fails the step instead of passing it.
    """
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
        try:
            backend.query_metadata_documents(
                schema=base_schema_name,
                yql=f"select * from {base_schema_name} where true limit 0",
                tenant_id=tenant,
                hits=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Provisioning verify failed for tenant {tenant}: "
                f"{base_schema_name} is not queryable: {exc}"
            ) from exc
        verified.append(backend.get_tenant_schema_name(tenant, base_schema_name))
    return verified


def init_memory(tenant_id: str) -> None:
    """Create the tenant's Mem0 memory schema via the runtime initializer."""
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_runtime.memory_init import lazy_init_memory

    config_manager = _config_manager(tenant_id)
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
        _config_manager(tenant),
        otlp_endpoint=resolve_library_env_defaults()["telemetry_otlp_endpoint"],
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

    validate_router_tier(tier)
    return set_tenant_tier(_config_manager(tenant_id), tenant_id, tier)


_PROFILE_STEPS = {"schemas": deploy_schemas, "verify": verify_schemas}
_TENANT_STEPS = {"memory": init_memory, "telemetry": init_telemetry}
_STEPS = sorted({*_PROFILE_STEPS, *_TENANT_STEPS, "tier"})


def main() -> int:
    # Same wiring every other runtime entrypoint applies: the step's
    # TELEMETRY_OTLP_ENDPOINT, MinIO credentials and embed URLs reach the
    # libraries only through this call. Without it the telemetry probe goes
    # to the default localhost collector and the tenant's project is never
    # created on the one the workflow named.
    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
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

    profiles: List[str] = []
    if args.step in _PROFILE_STEPS:
        profiles = [name for name in (args.profiles or "").split(",") if name.strip()]
        if not profiles:
            parser.error(f"--step {args.step} requires --profiles")
    if args.step == "tier" and not args.tier:
        parser.error("--step tier requires --tier")

    try:
        # Every step keys on the canonical tenant id, so the process resolves
        # it once and each step reports the id its rows are stored under.
        tenant = canonical_tenant_id(args.tenant_id)
        if args.step in _PROFILE_STEPS:
            names = _PROFILE_STEPS[args.step](tenant, profiles)
            print(f"Provisioned {args.step} for tenant {tenant}: {', '.join(names)}")
            return 0
        if args.step == "tier":
            init_tier(tenant, args.tier)
        else:
            _TENANT_STEPS[args.step](tenant)
        print(f"Provisioned {args.step} for tenant {tenant}")
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
