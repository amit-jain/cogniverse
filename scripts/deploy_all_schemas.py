#!/usr/bin/env python3
"""
Deploy All Vespa Schemas for Video Search

This script deploys all schemas from the configs/schemas directory to Vespa.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backends.vespa.json_schema_parser import JsonSchemaParser
from src.backends.vespa.vespa_schema_manager import VespaSchemaManager
from src.common.config_utils import get_config
from vespa.package import ApplicationPackage


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get configuration
        from cogniverse_core.config.utils import create_default_config_manager, get_config
        config_manager = create_default_config_manager()
        config = get_config(tenant_id="default", config_manager=config_manager)
        
        # Initialize the schema manager
        schema_manager = VespaSchemaManager()
        
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
        logger.info("📦 Deploying all schemas to Vespa...")
        
        # Add validation overrides to allow schema changes
        from datetime import datetime, timedelta

        from vespa.package import Validation
        
        # Allow schema removal for deployment
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
            logger.info("✅ All schemas deployed successfully!")
            
            # Extract and save ranking strategies after schema deployment
            logger.info("📊 Extracting ranking strategies from all schemas...")
            from src.backends.vespa.ranking_strategy_extractor import (
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
        
        logger.info("🎉 Schema deployment complete!")
        logger.info("🔍 You can now run video ingestion with: python scripts/run_ingestion.py --video_dir data/videos --backend vespa")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
