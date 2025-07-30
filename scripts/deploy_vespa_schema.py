#!/usr/bin/env python3
"""
Deploy Vespa Schema for Video Search

This script deploys the video_frame schema to Vespa using the VespaSchemaManager.
It should be run after starting Vespa but before running ingestion.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.vespa.vespa_schema_manager import VespaSchemaManager
from src.tools.config import get_config

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deploy Vespa schema')
    parser.add_argument('--schema', type=str, help='Schema name to deploy (e.g., video_frame, video_colqwen)')
    args = parser.parse_args()
    
    try:
        # Get configuration
        config = get_config()
        vespa_schema = args.schema if args.schema else config.get("vespa_schema", "video_frame")
        
        # Initialize the schema manager
        schema_manager = VespaSchemaManager()
        
        # Deploy the JSON video frame schema
        logger.info(f"🚀 Deploying {vespa_schema} schema from JSON to Vespa...")
        schema_path = Path(__file__).parent.parent / "configs" / "schemas" / f"{vespa_schema}_schema.json"
        
        if not schema_path.exists():
            logger.error(f"❌ Schema file not found: {schema_path}")
            logger.info(f"Available schemas in configs/schemas/:")
            schemas_dir = Path(__file__).parent.parent / "configs" / "schemas"
            if schemas_dir.exists():
                for schema_file in schemas_dir.glob("*.json"):
                    logger.info(f"  - {schema_file.name}")
            return 1
            
        schema_manager.upload_schema_from_json_file(str(schema_path))
        
        logger.info("✅ Schema deployed successfully!")
        logger.info("🔍 You can now run video ingestion with: python scripts/run_ingestion.py --video_dir data/videos/video_chatgpt_eval/sample_videos --backend vespa")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to deploy schema: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())