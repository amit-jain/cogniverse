"""
Integration test configuration for runtime integration tests.

Provides shared Vespa Docker instance with metadata schemas deployed,
plus ConfigManager, SchemaLoader, and FastAPI TestClient fixtures
wired with real dependencies including real ColPali query encoder.
"""

import json
import logging
import time
from pathlib import Path

import dspy
import httpx
import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    LLMEndpointConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import health, search
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for runtime integration tests.

    Module-scoped to share across all tests in this module.
    Deploys metadata schemas so VespaConfigStore can be used immediately.

    Yields:
        dict: Vespa connection info with http_port, config_port, base_url, container_name
    """
    manager = VespaDockerManager()

    # Clear stale singleton state from other test modules so the test gets
    # a fresh backend pointing at the isolated Vespa, not a cached one.
    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()

    try:
        # Use unique ports derived from module name hash to avoid collisions
        # with a running Vespa instance on the default 8080/19071 ports.
        container_info = manager.start_container(
            module_name="runtime_integration_tests",
            use_module_ports=True,
        )

        manager.wait_for_config_ready(container_info, timeout=180)

        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata + data schemas in a single application package.
        # Vespa rejects deployments that remove existing schemas, so all
        # schemas must be included together.
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

        # Parse the data schema used by search tests.
        # Rename to include "_default" tenant suffix to match the multi-tenant
        # naming convention (BackendRegistry generates tenant-specific schema names).
        schema_file = SCHEMAS_DIR / "video_colpali_smol500_mv_frame_schema.json"
        with open(schema_file) as f:
            schema_json = json.load(f)
        schema_json["name"] = "video_colpali_smol500_mv_frame_default"
        schema_json["document"]["name"] = "video_colpali_smol500_mv_frame_default"
        parser = JsonSchemaParser()
        data_schema = parser.parse_schema(schema_json)

        # Parse agent_memories schema for Mem0 strategy storage tests.
        memory_schema_file = SCHEMAS_DIR / "agent_memories_schema.json"
        with open(memory_schema_file) as f:
            memory_schema_json = json.load(f)
        memory_schema_json["name"] = "agent_memories_default"
        memory_schema_json["document"]["name"] = "agent_memories_default"
        memory_schema = parser.parse_schema(memory_schema_json)

        all_schemas = metadata_schemas + [data_schema, memory_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)
        logger.info("Deployed metadata + data schemas in single package")

        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info(
            "Vespa initialization complete - ready for runtime integration tests"
        )

        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()

        # Clear singleton state to avoid interference with other test modules
        try:
            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
        except Exception as cleanup_err:
            logger.warning(f"BackendRegistry cleanup failed: {cleanup_err}")


@pytest.fixture(scope="module")
def config_manager(vespa_instance):
    """
    ConfigManager backed by real VespaConfigStore.

    Seeds SystemConfig with backend_url/port pointing at the test Vespa container,
    and adds two test profiles for the default tenant.
    """
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )

    cm = ConfigManager(store=store)

    system_config = SystemConfig(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm.set_system_config(system_config)

    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_colpali",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colsmol-500m",
        ),
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_videoprism",
            type="video",
            schema_name="video_videoprism_base_mv_chunk_30s",
            embedding_model="google/videoprism-base",
        ),
    )

    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="tenant_b_profile",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colsmol-500m",
        ),
        tenant_id="tenant_b",
    )

    return cm


@pytest.fixture(scope="module")
def schema_loader():
    """FilesystemSchemaLoader from configs/schemas/."""
    return FilesystemSchemaLoader(SCHEMAS_DIR)


@pytest.fixture(scope="module")
def memory_manager(vespa_instance, config_manager, schema_loader):
    """Real Mem0MemoryManager backed by test Vespa Docker.

    Follows the same pattern as tests/memory/conftest.py shared_memory_vespa.
    Uses the same Vespa instance as search tests. Requires Ollama
    for Mem0's LLM-based memory extraction.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()

    mm = Mem0MemoryManager(tenant_id="default")
    mm.initialize(
        backend_host="http://localhost",
        backend_port=vespa_instance["http_port"],
        backend_config_port=vespa_instance["config_port"],
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        llm_base_url="http://localhost:11434",
        config_manager=config_manager,
        schema_loader=schema_loader,
        auto_create_schema=False,
    )

    yield mm

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()


@pytest.fixture(scope="module")
def real_telemetry(phoenix_container):
    """Module-scoped real TelemetryManager backed by Phoenix Docker.

    Sets up the global TelemetryManager singleton so that
    get_telemetry_manager() returns a real manager throughout search tests.
    """
    import os

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv("TELEMETRY_OTLP_ENDPOINT", "localhost:4317"),
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(scope="module")
def search_client(vespa_instance, config_manager, schema_loader, real_telemetry):
    """
    FastAPI TestClient with search router wired to real ConfigManager and SchemaLoader.

    Only the search router is mounted (no lifespan needed).
    Dependency overrides point at the real Vespa-backed ConfigManager.
    Real TelemetryManager singleton is set up via real_telemetry fixture.
    """
    test_app = FastAPI()
    test_app.include_router(search.router, prefix="/search")

    test_app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    test_app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="module")
def health_client(vespa_instance, config_manager):
    """
    FastAPI TestClient with health router.

    Mounts the health router for integration testing against real registries.
    """
    test_app = FastAPI()
    test_app.include_router(health.router)

    with TestClient(test_app) as client:
        yield client


def _is_llm_available() -> bool:
    """Check if the configured LLM endpoint is reachable.

    Reads api_base from configs/config.json directly (no ConfigManager
    needed — avoids BACKEND_URL env var requirement at import time).
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        config_path = _Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = _json.load(f)
        api_base = (
            config.get("llm_config", {})
            .get("primary", {})
            .get("api_base", "http://localhost:11434")
        )
        response = httpx.get(f"{api_base}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


skip_if_no_llm = pytest.mark.skipif(
    not _is_llm_available(),
    reason="Configured LLM endpoint not reachable",
)


@pytest.fixture(scope="module")
def _dspy_lm_instance():
    """Module-scoped: create the LM once per module (expensive)."""
    from cogniverse_foundation.config.utils import (
        create_default_config_manager as _cdcm,
    )

    cm = _cdcm()
    config = get_config(tenant_id="default", config_manager=cm)
    llm_cfg = config.get("llm_config", {}).get("primary", {})

    # Disable qwen3 thinking mode — it puts output in a 'thinking' field
    # that DSPy can't read, leaving content empty.
    extra_body = None
    model = llm_cfg["model"]
    if "qwen3" in model or "qwen-3" in model:
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=llm_cfg.get("api_base"),
        temperature=0.1,
        max_tokens=200,
        extra_body=extra_body,
    )
    return create_dspy_lm(endpoint)


@pytest.fixture
def dspy_lm(_dspy_lm_instance):
    """Function-scoped: re-apply dspy.configure before each test.

    The root conftest cleanup_dspy_state clears dspy.settings.lm after
    each test, so we must re-configure before every test that needs an LLM.
    """
    dspy.configure(lm=_dspy_lm_instance)
    return _dspy_lm_instance


@pytest.fixture(scope="module")
def agent_registry(config_manager):
    """AgentRegistry with a search_agent registered."""
    registry = AgentRegistry(tenant_id="default", config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://localhost:8000",
            capabilities=["search", "video_search"],
        )
    )
    return registry


@pytest.fixture(scope="module")
def dispatcher(agent_registry, config_manager, schema_loader):
    """Real AgentDispatcher wired to real registry, config, and schema loader."""
    return AgentDispatcher(
        agent_registry=agent_registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture(scope="module")
def a2a_client(dispatcher):
    """Starlette TestClient wrapping a real A2A server with InMemoryTaskStore.

    This is the production A2A stack: real executor, real task store,
    real request handler. Only the transport is in-process (TestClient).
    """
    executor = CogniverseAgentExecutor(dispatcher=dispatcher)

    card = AgentCard(
        name="Test Cogniverse",
        description="Integration test agent",
        url="http://localhost:9999/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="search_agent",
                name="search_agent",
                description="Search for videos",
                tags=["search", "video_search"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

    with StarletteTestClient(server.build()) as client:
        yield client
