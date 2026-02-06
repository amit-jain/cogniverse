#!/usr/bin/env python3
"""
Deploy Vespa Schemas for Video Search

This script supports two modes:
1. Base schema deployment (default): Deploys all base schema templates from configs/schemas/
2. Tenant schema deployment (--tenant-id): Deploys tenant-specific schemas using SchemaRegistry

Usage:
    # Deploy base schemas (development/initial setup)
    python scripts/deploy_all_schemas.py

    # Deploy schemas for a specific tenant
    python scripts/deploy_all_schemas.py --tenant-id acme:production

    # Deploy specific schemas for a tenant
    python scripts/deploy_all_schemas.py --tenant-id acme:production \
        --base-schemas video_colpali_smol500_mv_frame,video_videoprism_base_mv_chunk_30s

    # Force redeploy tenant schemas (even if already deployed)
    python scripts/deploy_all_schemas.py --tenant-id acme:production --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def deploy_base_schemas(logger: logging.Logger) -> int:
    """Deploy base schema templates (development/initial setup mode)."""
    from datetime import datetime, timedelta

    from vespa.package import ApplicationPackage, Validation

    from cogniverse_foundation.config.utils import create_default_config_manager, get_config
    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)

    # Initialize the schema manager with backend config
    schema_manager = VespaSchemaManager(
        backend_endpoint=config.get("backend_url"),
        backend_port=config.get("backend_port")
    )

    # Get all schema files
    schemas_dir = Path(__file__).parent.parent / "configs" / "schemas"
    schema_files = list(schemas_dir.glob("*.json"))

    logger.info(f"🚀 Found {len(schema_files)} schemas to deploy")

    # Create application package with all schemas
    app_package = ApplicationPackage(name="videosearch")

    # Parse each schema and add to package
    for schema_file in schema_files:
        logger.info(f"📄 Loading schema from {schema_file.name}")
        try:
            parser = JsonSchemaParser()
            schema = parser.load_schema_from_json_file(str(schema_file))
            app_package.add_schema(schema)
            logger.info(f"✅ Added schema: {schema.name}")
        except Exception as e:
            logger.error(f"❌ Failed to parse {schema_file.name}: {str(e)}")
            continue

    # Deploy all schemas at once
    logger.info("📦 Deploying all base schemas to Vespa...")

    # Add validation overrides to allow schema changes
    until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    validation = Validation(
        validation_id="schema-removal",
        until=until_date
    )
    if app_package.validations is None:
        app_package.validations = []
    app_package.validations.append(validation)

    try:
        # Use the schema manager's deploy method
        schema_manager._deploy_package(app_package)
        logger.info("✅ All base schemas deployed successfully!")

        # Extract and save ranking strategies after schema deployment
        logger.info("📊 Extracting ranking strategies from all schemas...")
        from cogniverse_vespa.ranking_strategy_extractor import (
            extract_all_ranking_strategies,
            save_ranking_strategies,
        )

        strategies = extract_all_ranking_strategies(schemas_dir)
        save_ranking_strategies(strategies, schemas_dir / "ranking_strategies.json")
        logger.info(f"✅ Extracted {sum(len(s) for s in strategies.values())} ranking strategies from {len(strategies)} schemas")

    except Exception as e:
        logger.error(f"❌ Failed to deploy schemas: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("🎉 Base schema deployment complete!")
    return 0


def deploy_tenant_schemas(
    tenant_id: str,
    base_schemas: list[str] | None,
    force: bool,
    logger: logging.Logger
) -> int:
    """Deploy tenant-specific schemas using SchemaRegistry."""
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager, get_config
    from cogniverse_vespa.backend import VespaBackend

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)

    # Initialize backend
    backend = VespaBackend(
        vespa_url=config.get("backend_url"),
        vespa_port=config.get("backend_port")
    )

    # Initialize schema loader
    schemas_dir = Path(__file__).parent.parent / "configs" / "schemas"
    schema_loader = FilesystemSchemaLoader(schemas_dir)

    # Initialize schema registry
    registry = SchemaRegistry()
    registry.configure(backend=backend, schema_loader=schema_loader)

    # Determine which schemas to deploy
    if base_schemas:
        schemas_to_deploy = base_schemas
    else:
        # Get all video schemas (exclude metadata schemas)
        all_schemas = schema_loader.list_schemas()
        schemas_to_deploy = [
            s for s in all_schemas
            if s.startswith("video_") and not s.endswith("_metadata")
        ]

    if not schemas_to_deploy:
        logger.warning("⚠️ No schemas found to deploy")
        return 1

    logger.info(f"🚀 Deploying {len(schemas_to_deploy)} schemas for tenant: {tenant_id}")

    deployed = []
    failed = []

    for base_schema in schemas_to_deploy:
        try:
            logger.info(f"📄 Deploying schema: {base_schema} for tenant: {tenant_id}")
            tenant_schema_name = registry.deploy_schema(
                tenant_id=tenant_id,
                base_schema_name=base_schema,
                force=force
            )
            logger.info(f"✅ Deployed: {tenant_schema_name}")
            deployed.append(tenant_schema_name)
        except Exception as e:
            logger.error(f"❌ Failed to deploy {base_schema}: {str(e)}")
            failed.append(base_schema)

    # Summary
    logger.info("=" * 60)
    logger.info(f"🎉 Tenant schema deployment complete for: {tenant_id}")
    logger.info(f"   ✅ Deployed: {len(deployed)} schemas")
    if failed:
        logger.info(f"   ❌ Failed: {len(failed)} schemas: {', '.join(failed)}")
    logger.info("=" * 60)

    return 0 if not failed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Vespa schemas for video search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Deploy base schemas (development/initial setup)
    python scripts/deploy_all_schemas.py

    # Deploy schemas for a specific tenant
    python scripts/deploy_all_schemas.py --tenant-id acme:production

    # Deploy specific schemas for a tenant
    python scripts/deploy_all_schemas.py --tenant-id acme:production \\
        --base-schemas video_colpali_smol500_mv_frame,video_videoprism_base_mv_chunk_30s

    # Force redeploy tenant schemas
    python scripts/deploy_all_schemas.py --tenant-id acme:production --force
        """
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        help="Tenant ID (e.g., 'acme:production'). If provided, deploys tenant-specific schemas."
    )
    parser.add_argument(
        "--base-schemas",
        type=str,
        help="Comma-separated list of base schemas to deploy (default: all video schemas)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redeploy even if schemas already exist (only with --tenant-id)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        if args.tenant_id:
            # Tenant-specific deployment
            base_schemas = None
            if args.base_schemas:
                base_schemas = [s.strip() for s in args.base_schemas.split(",")]

            return deploy_tenant_schemas(
                tenant_id=args.tenant_id,
                base_schemas=base_schemas,
                force=args.force,
                logger=logger
            )
        else:
            # Base schema deployment (default)
            if args.force:
                logger.warning("⚠️ --force flag is ignored without --tenant-id")
            if args.base_schemas:
                logger.warning("⚠️ --base-schemas flag is ignored without --tenant-id (deploys all base schemas)")

            return deploy_base_schemas(logger)

    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
