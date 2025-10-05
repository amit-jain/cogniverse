#!/usr/bin/env python3
"""
Generic script to deploy any JSON schema to Vespa

Usage:
    python scripts/deploy_json_schema.py configs/schemas/agent_memories_schema.json
    python scripts/deploy_json_schema.py configs/schemas/config_metadata_schema.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from vespa.package import ApplicationPackage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.vespa.json_schema_parser import JsonSchemaParser


def deploy_json_schema(schema_file: str, vespa_host: str = "localhost",
                      config_port: int = 19071, data_port: int = 8080) -> bool:
    """Deploy a JSON schema file to Vespa"""

    schema_path = Path(schema_file)
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return False

    print(f"📄 Loading schema from {schema_path}")

    # Load and parse JSON schema
    with open(schema_path, 'r') as f:
        schema_config = json.load(f)

    schema_name = schema_config.get('name', 'unknown')
    print(f"📦 Processing schema: {schema_name}")

    # Parse schema using JsonSchemaParser
    parser = JsonSchemaParser()
    schema = parser.parse_schema(schema_config)

    # Create application package
    app_package = ApplicationPackage(name=schema_name.replace('_', ''))
    app_package.add_schema(schema)

    # Deploy to Vespa
    deploy_url = f"http://{vespa_host}:{config_port}/application/v2/tenant/default/prepareandactivate"
    print(f"🚀 Deploying to {deploy_url}...")

    try:
        # Generate the ZIP package
        app_zip = app_package.to_zip()

        # Deploy via HTTP
        response = requests.post(
            deploy_url,
            headers={"Content-Type": "application/zip"},
            data=app_zip,
            timeout=60,
            verify=False,
        )

        if response.status_code == 200:
            print(f"✅ Schema '{schema_name}' deployed successfully!")
            return True
        else:
            print(f"❌ Deployment failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Deployment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_deployment(schema_name: str, vespa_host: str = "localhost",
                     data_port: int = 8080) -> bool:
    """Verify a schema is deployed by checking application status"""

    print(f"\n🔍 Verifying '{schema_name}' deployment...")

    try:
        # Check application status
        response = requests.get(
            f"http://{vespa_host}:{data_port}/ApplicationStatus",
            timeout=5
        )

        if response.status_code == 200:
            print(f"✅ Vespa is running and responding")
            return True
        else:
            print(f"⚠️  Vespa returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def main():
    """Main deployment function"""

    parser = argparse.ArgumentParser(
        description="Deploy JSON schema files to Vespa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy agent memories schema
  python scripts/deploy_json_schema.py configs/schemas/agent_memories_schema.json

  # Deploy config metadata schema
  python scripts/deploy_json_schema.py configs/schemas/config_metadata_schema.json

  # Deploy to remote Vespa instance
  python scripts/deploy_json_schema.py configs/schemas/agent_memories_schema.json \\
    --config-host vespa.example.com --config-port 19071
        """
    )

    parser.add_argument(
        "schema_file",
        help="Path to JSON schema file to deploy"
    )

    parser.add_argument(
        "--config-host",
        default=os.getenv("VESPA_CONFIG_HOST", "localhost"),
        help="Vespa config server host (default: localhost)"
    )

    parser.add_argument(
        "--config-port",
        type=int,
        default=int(os.getenv("VESPA_CONFIG_PORT", "19071")),
        help="Vespa config server port (default: 19071)"
    )

    parser.add_argument(
        "--data-host",
        default=os.getenv("VESPA_HOST", "localhost"),
        help="Vespa data endpoint host (default: localhost)"
    )

    parser.add_argument(
        "--data-port",
        type=int,
        default=int(os.getenv("VESPA_PORT", "8080")),
        help="Vespa data endpoint port (default: 8080)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Vespa JSON Schema Deployment")
    print("=" * 60)
    print(f"\nSchema file: {args.schema_file}")
    print(f"Config server: {args.config_host}:{args.config_port}")
    print(f"Data endpoint: {args.data_host}:{args.data_port}\n")

    # Deploy schema
    if deploy_json_schema(
        args.schema_file,
        args.config_host,
        args.config_port,
        args.data_port
    ):
        # Wait for deployment to propagate
        print("\n⏳ Waiting for deployment to propagate...")
        time.sleep(5)

        # Extract schema name for verification
        with open(args.schema_file, 'r') as f:
            schema_config = json.load(f)
            schema_name = schema_config.get('name', 'unknown')

        # Verify deployment
        verify_deployment(schema_name, args.data_host, args.data_port)

        print("\n" + "=" * 60)
        print("Deployment complete!")
        print("=" * 60)
    else:
        print("\n❌ Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()