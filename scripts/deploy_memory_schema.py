"""
Deploy agent_memories schema to Vespa

This script deploys the agent_memories schema to a running Vespa instance.
"""

import os
import sys
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vespa.package import ApplicationPackage

from src.backends.vespa.vespa_schema_manager import VespaSchemaManager


def create_application_package(vespa_host="localhost", data_port=8080, config_port=19071):
    """Create Vespa application package with agent_memories schema"""

    # Schema path
    schema_path = Path(__file__).parent.parent / "vespa_schemas" / "agent_memories.sd"

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)

    print(f"📄 Loading schema from {schema_path}")

    # Use VespaSchemaManager to parse the .sd file
    schema_manager = VespaSchemaManager(
        vespa_endpoint=f"http://{vespa_host}:{data_port}",
        vespa_port=config_port
    )
    sd_content = schema_manager.read_sd_file(str(schema_path))
    schema = schema_manager.parse_sd_schema(sd_content)

    # Create application package
    app_package = ApplicationPackage(name="agentmemories")
    app_package.add_schema(schema)

    return app_package


def deploy_schema(vespa_host="localhost", vespa_port=19071, data_port=8080):
    """Deploy schema to Vespa config server"""

    print("📦 Creating application package...")
    app_package = create_application_package(vespa_host, data_port, vespa_port)

    # Deploy to Vespa
    deploy_url = f"http://{vespa_host}:{vespa_port}/application/v2/tenant/default/prepareandactivate"

    print(f"🚀 Deploying to {deploy_url}...")

    try:
        # Generate the ZIP package using pyvespa
        app_zip = app_package.to_zip()

        # Deploy via HTTP with correct Content-Type
        response = requests.post(
            deploy_url,
            headers={"Content-Type": "application/zip"},
            data=app_zip,
            timeout=60,
            verify=False,
        )

        if response.status_code == 200:
            print("✅ Schema deployed successfully!")
            print(f"Response: {response.text}")
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


def verify_schema(vespa_host="localhost", vespa_port=8080):
    """Verify schema is deployed"""

    print("\n🔍 Verifying schema deployment...")

    try:
        # Try to feed a test document to verify schema exists
        test_doc = {
            "fields": {
                "id": "test_doc",
                "text": "test content",
                "user_id": "test_user",
                "agent_id": "test_agent",
                "embedding": [0.0] * 768,
                "metadata_": "{}",
                "created_at": 1234567890
            }
        }

        response = requests.post(
            f"http://{vespa_host}:{vespa_port}/document/v1/agent_memories/agent_memories/docid/test_doc",
            json=test_doc,
            timeout=5,
        )

        if response.status_code in [200, 201]:
            print("✅ Schema verified - test document indexed successfully")

            # Clean up test document
            requests.delete(
                f"http://{vespa_host}:{vespa_port}/document/v1/agent_memories/agent_memories/docid/test_doc",
                timeout=5,
            )
            return True
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def main():
    """Main deployment function"""

    print("=" * 60)
    print("Vespa Agent Memories Schema Deployment")
    print("=" * 60)

    # Get config from environment or use defaults
    config_host = os.getenv("VESPA_CONFIG_HOST", "localhost")
    config_port = int(os.getenv("VESPA_CONFIG_PORT", "19071"))
    data_host = os.getenv("VESPA_HOST", "localhost")
    data_port = int(os.getenv("VESPA_PORT", "8080"))

    print(f"\nConfig server: {config_host}:{config_port}")
    print(f"Data endpoint: {data_host}:{data_port}\n")

    # Deploy schema
    if deploy_schema(config_host, config_port):
        # Wait a bit for deployment to propagate
        import time
        print("\n⏳ Waiting for deployment to propagate...")
        time.sleep(5)

        # Verify deployment
        verify_schema(data_host, data_port)
    else:
        print("\n❌ Deployment failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
