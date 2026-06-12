"""
Pytest configuration for routing integration tests.

Provides LM + Vespa fixtures for tests that need real services.
"""

import logging
import os
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

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401, E402

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.fixture(autouse=True)
def dspy_lm():
    """Configure DSPy with the configured test LM for integration tests."""
    from tests.fixtures.llm import (
        is_test_lm_available,
        resolve_api_key,
        resolve_base_url,
        resolve_prefixed_model,
    )

    # Provision the self-managed LM sidecar first — it exports
    # COGNIVERSE_CONFIG so the resolvers below see the hermetic endpoint.
    from tests.utils.hermetic_llm import ensure_llm

    if ensure_llm() is None or not is_test_lm_available():
        pytest.skip(f"Test LM not provisionable (resolved {resolve_base_url()})")

    config = LLMEndpointConfig(
        model=resolve_prefixed_model(),
        api_base=resolve_base_url(),
        api_key=resolve_api_key(),
    )
    lm = create_dspy_lm(config)
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)


@pytest.fixture(scope="module")
def vespa_instance(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape routing/integration tests
    expect, backed by the project-wide ``shared_vespa``.

    Deploys ``video_colpali_smol500_mv_frame`` for tenant ``test_unit``
    via SchemaRegistry (merge-safe, doesn't touch other tenants'
    schemas). Sets ``BACKEND_URL`` / ``BACKEND_PORT`` env vars so any
    ``create_default_config_manager()`` call inside tests resolves to
    the shared container. Registers a profile under ``test:unit`` so
    ``get_config(tenant_id="test:unit")`` can look it up.
    """
    from cogniverse_core.registries.backend_registry import BackendRegistry

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None

    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    # Pre-deploy the data schema via SchemaRegistry (merge-safe).
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_vespa,
        tenant_id="test:unit",
        base_schema_name="video_colpali_smol500_mv_frame",
    )

    # Point create_default_config_manager() at the shared container.
    os.environ["BACKEND_URL"] = "http://localhost"
    os.environ["BACKEND_PORT"] = str(shared_vespa["http_port"])

    # Seed SystemConfig + register the test:unit profile.
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendProfileConfig,
        SystemConfig,
    )
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_vespa["http_port"],
        )
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="video_colpali_smol500_mv_frame",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colsmol-500m",
        ),
        tenant_id="test:unit",
    )

    BackendRegistry._backend_instances.clear()

    try:
        yield {
            "http_port": shared_vespa["http_port"],
            "config_port": shared_vespa["config_port"],
            "base_url": shared_vespa["base_url"],
            "container_name": shared_vespa["container_name"],
        }
    finally:
        # Restore env vars; clear singletons.
        if original_url is not None:
            os.environ["BACKEND_URL"] = original_url
        elif "BACKEND_URL" in os.environ:
            del os.environ["BACKEND_URL"]
        if original_port is not None:
            os.environ["BACKEND_PORT"] = original_port
        elif "BACKEND_PORT" in os.environ:
            del os.environ["BACKEND_PORT"]
        try:
            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
            BackendRegistry._shared_schema_registry = None
        except Exception as cleanup_err:
            logger.warning(f"BackendRegistry cleanup failed: {cleanup_err}")


@pytest.fixture
def test_generator_config():
    """Create test generator configuration with all required optimizer configs"""
    return SyntheticGeneratorConfig(
        tenant_id="test:unit",
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
        },
    )


@pytest.fixture
def test_backend_config():
    """Create test backend configuration"""
    return BackendConfig(profiles={}, tenant_id="test:unit")
