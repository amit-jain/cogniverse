"""
Pytest configuration for routing integration tests.

Provides Ollama + Vespa Docker fixtures for tests that need real services.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    BackendConfig,
    DSPyModuleConfig,
    LLMEndpointConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


def is_ollama_available():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(autouse=True)
def dspy_lm():
    """Configure DSPy with real Ollama LM for integration tests"""
    if not is_ollama_available():
        pytest.skip("Ollama not available - required for integration tests")

    config = LLMEndpointConfig(
        model="ollama/gemma3:4b",
        api_base="http://localhost:11434",
    )
    lm = create_dspy_lm(config)
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)


@pytest.fixture(scope="module")
def vespa_instance():
    """Start isolated Vespa Docker for routing integration tests.

    Module-scoped to share across all tests. Deploys metadata + colpali
    schemas so SearchService can query against it. Sets BACKEND_URL and
    BACKEND_PORT env vars so create_default_config_manager() connects to
    this container instead of localhost:8080.
    """
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from tests.utils.vespa_docker import VespaDockerManager

    manager = VespaDockerManager()

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()

    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    try:
        container_info = manager.start_container(
            module_name="routing_integration_tests",
            use_module_ports=True,
        )
        manager.wait_for_config_ready(container_info, timeout=180)

        logger.info("Waiting 15s for Vespa internal services to initialize...")
        time.sleep(15)

        from vespa.package import ApplicationPackage

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        schema_file = SCHEMAS_DIR / "video_colpali_smol500_mv_frame_schema.json"
        with open(schema_file) as f:
            schema_json = json.load(f)
        schema_json["name"] = "video_colpali_smol500_mv_frame_default"
        schema_json["document"]["name"] = "video_colpali_smol500_mv_frame_default"
        parser = JsonSchemaParser()
        data_schema = parser.parse_schema(schema_json)

        all_schemas = metadata_schemas + [data_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)

        manager.wait_for_application_ready(container_info, timeout=120)

        # Point create_default_config_manager() at the test container
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(container_info["http_port"])

        # Seed SystemConfig into VespaConfigStore so get_config() works
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import (
            BackendProfileConfig,
            SystemConfig,
        )
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=container_info["http_port"],
        )
        cm = ConfigManager(store=store)
        cm.set_system_config(
            SystemConfig(
                backend_url="http://localhost",
                backend_port=container_info["http_port"],
            )
        )
        cm.add_backend_profile(
            BackendProfileConfig(
                profile_name="video_colpali_smol500_mv_frame",
                type="video",
                schema_name="video_colpali_smol500_mv_frame",
                embedding_model="vidore/colsmol-500m",
            ),
        )

        logger.info("Routing Vespa ready for integration tests")
        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        # Restore original env vars
        if original_url is not None:
            os.environ["BACKEND_URL"] = original_url
        elif "BACKEND_URL" in os.environ:
            del os.environ["BACKEND_URL"]

        if original_port is not None:
            os.environ["BACKEND_PORT"] = original_port
        elif "BACKEND_PORT" in os.environ:
            del os.environ["BACKEND_PORT"]

        manager.stop_container()

        try:
            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
        except Exception as cleanup_err:
            logger.warning(f"BackendRegistry cleanup failed: {cleanup_err}")


@pytest.fixture
def test_generator_config():
    """Create test generator configuration with all required optimizer configs"""
    return SyntheticGeneratorConfig(
        optimizer_configs={
            "modality": OptimizerGenerationConfig(
                optimizer_type="modality",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
                        module_type="Predict",
                    )
                },
                agent_mappings=[
                    AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
                    AgentMappingRule(
                        modality="DOCUMENT", agent_name="document_search_agent"
                    ),
                ],
            ),
            "routing": OptimizerGenerationConfig(
                optimizer_type="routing",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                        module_type="Predict",
                    )
                },
            ),
        }
    )


@pytest.fixture
def test_backend_config():
    """Create test backend configuration"""
    return BackendConfig(profiles={})
